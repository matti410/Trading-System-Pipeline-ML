"""
Le funzioni delle celle del notebook, lavoro B (30/10/2026, docs/02 §B e §F).

  ore(H, timeframe) -> str
  orizzonte_event_study(timeframe) -> int
  barre_train(ds, quota=0.8) -> DataFrame
  event_study_train(ds, direzione, H, quota=0.8, trigger=None) -> EventStudy
  sintesi_event_study(ev, timeframe) -> DataFrame
  controlla_scelta(ds, trigger, H, direzione) -> bool
  feature_lato(ds, trigger, df_btc=None) -> DataFrame
  prepara_lato(ds, direzione, trigger, H, quota=0.8, feat=None) -> Lato
  seleziona_lato(ds, lato, ...) -> (Selezione, Lato con la X scelta)
  confronto_base_selezione(ds, lato, modello, lato_sel, modello_sel, ...) -> DataFrame
  allena(ds, lato, con_unicita=True, peso_rendimento=False, ...) -> (modello, fasce)
  confronti(ds, lato, modello, fasce=5) -> (tabella, percentile)
  guarda_oos(ds, lato, modello, fasce=5, ...) -> tabella
  backtest(ds, lati, modelli, fasce=5, ...) -> (tabella, risultati)

REGOLE
------
- L'event study gira SOLO sulle barre del train (D2): le barre prima del confine
  train/test di split.barra_confine, lo stesso dello split. min_trades del Lab.
- Nessuna scelta automatica: trigger e H del lato li scrive Mattia
  (controlla_scelta li controlla e basta).
- Ogni addestramento e ogni sguardo all'OOS (tabella o backtest) finiscono nel
  registro (registro.py, D4). Dal lavoro D anche ogni selezione e ogni confronto
  base/selezione.
- Lavoro D, regola sui NaN (decisione del revisore): i trigger che cadono nel
  riscaldamento delle feature (le prime barre in cui una feature calcolata e' ancora NaN
  per mancanza di storia) si tolgono dal target del lato PRIMA dello split, cosi' modello
  base e modello con la selezione usano le stesse righe. I NaN sparsi dopo il
  riscaldamento restano NaN: LightGBM li gestisce.
- Lavoro D, regola fissata prima: soglia, confronti D3, OOS e backtest usano SEMPRE il
  modello con la selezione; il modello base si stampa solo come riferimento.
"""
import dataclasses
from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import roc_auc_score

import registro
from backtest_oos import esegui_backtest, tabella_metriche, trigger_in_fascia
from confronti import confronti_test, miglior_filtro_train, selezione_test
from modello import PARAMETRI_BASE, addestra, importanza, rendimenti_trigger, valuta
from oos import FasceNonCalcolabili, indici_oos, soglia_dal_test, valuta_oos
from pesi import pesi_unicita
from split import barra_confine, costruisci_X, split_train_test
from target import costruisci_target

NOMI_LATO = {1: "LONG", -1: "SHORT"}


def ore(H: int, timeframe: str) -> str:
    """H barre in ore, come le scrive il Lab («4 ore», «6,25 ore»)."""
    from engine.vetrina import _ore_di
    return _ore_di(int(H), timeframe)


def orizzonte_event_study(timeframe: str) -> int:
    """
    Fin dove guarda l'event study: l'orizzonte piu' lungo della griglia del Lab
    (1, 4 e 8 ore). La curva mostra tutti gli orizzonti intermedi: non e' una scelta.
    """
    from engine.vetrina import orizzonti_in_barre
    barre, _ = orizzonti_in_barre(timeframe)
    return max(barre)


def barre_train(ds, quota: float = 0.8) -> pd.DataFrame:
    """Le barre del train: df_is fino al confine train/test, escluso."""
    return ds.df_is.iloc[:barra_confine(ds.df_is, quota)]


# Il minimo di occorrenze piu' basso che run_event_study di V4 accetta: 0 («Metti 0 per tenerle
# tutte», docstring di V4). Cosi' nessun trigger esce dall'event study (precisazione di Mattia,
# 30/10/2026): i trigger sotto il minimo del Lab (vetrina.MIN_TRIGGER) restano visibili, con un segno.
MIN_TRADES_EVENT_STUDY = 0


def event_study_train(ds, direzione: int, H: int, quota: float = 0.8, trigger=None):
    """
    L'event study di V4 sulle sole barre del train, con TUTTI i trigger candidati del lato
    (o quelli di `trigger`), senza minimo di occorrenze. Ritorna l'EventStudy di V4.
    """
    from engine.event_study import run_event_study
    if direzione == -1 and ds.solo_long:
        raise ValueError("Sullo spot lo short non esiste: niente event study SHORT.")
    candidati = list(trigger) if trigger is not None else (ds.trigger_long if direzione == 1 else ds.trigger_short)
    ev = run_event_study(barre_train(ds, quota), entry_names=candidati, horizon=int(H),
                         min_trades=MIN_TRADES_EVENT_STUDY, pip=ds.pip, verbose=False)
    # i trigger che nel train non scattano mai non hanno curva: V4 non li mette nella sintesi.
    # Si ricordano qui, cosi' la sintesi e il grafico li mostrano comunque (0 occorrenze).
    ev.candidati_richiesti = candidati
    return ev


def occorrenze_train(ev) -> dict:
    """Le occorrenze di ogni trigger nel train, come le conta l'event study di V4 (dopo la quarantena)."""
    if ev is None:
        return {}
    conta = {nome: 0 for nome in getattr(ev, "candidati_richiesti", [])}
    conta.update(dict(zip(ev.sintesi["candidato"], ev.sintesi["trades"].astype(int))))
    conta.update({nome: int(n) for nome, n in ev.scartate})
    return conta


def sintesi_event_study(ev, timeframe: str, costo_bp=None) -> pd.DataFrame:
    """
    La sintesi di V4, nell'ordine di V4, con le ore accanto alla barra del picco e un segno
    sui trigger con meno occorrenze nel train del minimo del Lab. Non toglie nessun trigger.
    costo_bp (dataset.costo_bp()["totale_bp"]): aggiunge picco_netto_bp = picco in bp - costo.
    """
    from engine.vetrina import MIN_TRIGGER
    colonne = ["candidato", "trades", "barra_picco", "picco_pct", "incertezza_pct", "volte_incertezza",
               "oltre_rumore", "a_fine_pct", "vs_mercato_pct"]
    s = ev.sintesi[[c for c in colonne if c in ev.sintesi.columns]].copy()
    s.insert(s.columns.get_loc("barra_picco") + 1, "ore_picco", [ore(b, timeframe) for b in s["barra_picco"]])
    if costo_bp is not None:
        s.insert(s.columns.get_loc("picco_pct") + 1, "picco_netto_bp", s["picco_pct"] * 100 - float(costo_bp))
    senza = [c for c in getattr(ev, "candidati_richiesti", []) if c not in set(s["candidato"])]
    if senza:
        s = pd.concat([s, pd.DataFrame({"candidato": senza, "trades": 0})], ignore_index=True)
    s.insert(s.columns.get_loc("trades") + 1, f"sotto_minimo_Lab_{MIN_TRIGGER}",
             ["◄ sotto il minimo" + (" (nessuna occorrenza: nessuna curva)" if n == 0 else "")
              if n < MIN_TRIGGER else "" for n in s["trades"]])
    return s


def sintesi_da_mostrare(sintesi: pd.DataFrame, n_mostra: int = 10):
    """
    Solo visualizzazione (passo 3): dalla sintesi completa, i trigger con picco_pct > 0
    (il picco di V4 ha gia' il segno del lato), prima quelli oltre il rumore e poi gli
    altri, dentro ciascun gruppo per picco decrescente; al massimo n_mostra righe.
    Nessun filtro sulle occorrenze. Ritorna (tabella, conteggi per la riga sotto il grafico).
    """
    picco = pd.to_numeric(sintesi["picco_pct"], errors="coerce")
    senza_curva = picco.isna()
    # le colonne d'ordine prima del filtro: su una tabella gia' vuota pandas prenderebbe il loro indice
    positivi = sintesi.assign(_oltre=sintesi["oltre_rumore"].fillna(False).astype(bool), _picco=picco)[picco > 0]
    ordinati = positivi.sort_values(["_oltre", "_picco"], ascending=[False, False], kind="stable")
    vista = ordinati.drop(columns=["_oltre", "_picco"]).head(int(n_mostra)).reset_index(drop=True)
    conteggi = {"mostrati": len(vista), "totale": len(sintesi),
                "picco_non_positivo": int((picco <= 0).sum()),
                "oltre_i_primi": len(ordinati) - len(vista), "senza_curva": int(senza_curva.sum())}
    return vista, conteggi


def riga_mostrati(conteggi: dict, n_mostra: int = 10) -> str:
    """La riga sotto il grafico: quanti trigger si vedono e perche' gli altri no."""
    c = conteggi
    motivi = f"{c['picco_non_positivo']} con picco ≤ 0, {c['oltre_i_primi']} oltre i primi {n_mostra}"
    if c["senza_curva"]:
        motivi += f", {c['senza_curva']} senza curva perché senza occorrenze nel train"
    if c["mostrati"] == 0:
        return (f"nessun trigger mostrato: nessuno dei {c['totale']} trigger ha il picco > 0 ({motivi}). "
                "La sintesi completa resta nella variabile della cella.")
    return f"mostrati {c['mostrati']} di {c['totale']} trigger ({motivi})"


def _plot_con_costo(copia, nomi, costo_bp):
    """Il grafico di V4, con la linea tratteggiata al costo di un trade (in %, come l'asse)."""
    import matplotlib.pyplot as plt
    if costo_bp is None:
        copia.plot(candidati=nomi, figsize=(13, 7.5))       # il grafico di V4: si mostra da se'
        return
    mostra = plt.show
    plt.show = lambda *a, **k: None                          # V4 chiama plt.show: la linea va aggiunta prima
    try:
        copia.plot(candidati=nomi, figsize=(13, 7.5))
    finally:
        plt.show = mostra
    ax = plt.gcf().axes[0]
    ax.axhline(float(costo_bp) / 100, color=COLORE_COSTO, ls="--", lw=1.2,
               label=f"costo di un trade ({float(costo_bp):.2f} bp: commissione andata e ritorno + spread)")
    ax.legend(fontsize=8, loc="best")
    plt.show()


COLORE_COSTO = "#c0392b"


def grafico_event_study(ev, titolo: str = "", candidati=None, costo_bp=None):
    """
    Le curve dell'event study con il grafico di V4 (EventStudy.plot), con le occorrenze nel
    train accanto al nome (nella legenda). candidati=None: TUTTI i trigger misurati; una lista:
    solo quei trigger (piu' la curva del mercato, che V4 disegna sempre). costo_bp: una linea
    orizzontale tratteggiata al costo di un trade. Ritorna le etichette.
    """
    import copy
    n = occorrenze_train(ev)
    etichetta = {c: f"{c} ({n.get(c, 0)} nel train)" for c in ev.curve.columns}
    copia = copy.copy(ev)
    copia.curve = ev.curve.rename(columns=etichetta)
    copia.mercato = ev.mercato.rename(columns=etichetta)
    copia.sintesi = ev.sintesi.assign(candidato=ev.sintesi["candidato"].map(etichetta))
    if candidati is not None:
        nomi = [etichetta[c] for c in candidati]
        if titolo:
            print(titolo)
        if nomi:
            _plot_con_costo(copia, nomi, costo_bp)
        else:
            print("nessuna curva da disegnare: nessun trigger da mostrare.")
        return nomi
    nomi = list(etichetta.values())
    senza = [c for c in getattr(ev, "candidati_richiesti", list(ev.curve.columns))
             if c not in ev.curve.columns or ev.curve[c].isna().all()]
    if titolo:
        print(titolo)
    _plot_con_costo(copia, nomi, costo_bp)
    if senza:
        print(f"senza curva (nessuna occorrenza nel train): {', '.join(senza)}")
    return nomi + [f"{c} (0 nel train, senza curva)" for c in senza if c not in ev.curve.columns]


def controlla_scelta(ds, trigger, H, direzione: int, occorrenze=None) -> bool:
    """
    Controlla la scelta di Mattia per un lato. True = il lato si usa; False = lato
    saltato (nessun trigger scritto, qualunque sia H; oppure lo short sullo spot).
    Si ferma se ci sono trigger ma H manca o non e' un numero di barre >= 1, se un
    nome non e' un trigger del lato, o se si chiede lo short sullo spot.

    occorrenze: le occorrenze nel train di ogni trigger (occorrenze_train). Un trigger
    scelto sotto il minimo del Lab da' un AVVISO ben visibile, ma non ferma niente.
    """
    nome = NOMI_LATO[direzione]
    trigger = list(trigger or [])
    if direzione == -1 and ds.solo_long:
        if trigger:
            raise ValueError("Sullo spot lo short non esiste: lascia vuoto TRIGGER_SHORT.")
        print("SHORT: saltato (spot, lo short non esiste).")
        return False
    if not trigger:
        print(f"{nome}: nessun trigger scelto, lato saltato")
        return False
    if H is None:
        raise ValueError(f"Hai scritto TRIGGER_{nome}: scrivi anche H_{nome} (in barre) nella cella 4, "
                         "guardando l'event study qui sopra. Finché manca, il notebook si ferma qui.")
    if isinstance(H, bool) or not isinstance(H, int) or H < 1:
        raise ValueError(f"H_{nome} deve essere un numero intero di barre >= 1, non {H!r}.")
    validi = ds.trigger_long if direzione == 1 else ds.trigger_short
    estranei = [t for t in trigger if t not in validi]
    if estranei:
        raise ValueError(f"Non sono trigger {nome} del Lab: {estranei}. I nomi sono nella sintesi dell'event study.")
    print(f"{nome}: {len(trigger)} trigger, H = {H} barre ({ore(H, ds.timeframe)})")
    from engine.vetrina import MIN_TRIGGER
    pochi = {t: occorrenze[t] for t in trigger if occorrenze and t in occorrenze and occorrenze[t] < MIN_TRIGGER}
    if pochi:
        riga = "!" * 78
        dettaglio = ", ".join(f"{t}: {n} occorrenze" for t, n in pochi.items())
        print(f"{riga}\n!!! AVVISO · {nome}: trigger con meno occorrenze nel train del minimo del Lab "
              f"({MIN_TRIGGER}): {dettaglio}.\n!!! La scelta resta valida: il notebook va avanti.\n{riga}")
    return True


def lati_scelti(ds, scelta: dict, ev: dict) -> list:
    """
    La cella 4: controlla_scelta per LONG e SHORT, ritorna i lati usati. Un lato si puo'
    lasciare vuoto; se sono vuoti tutti e due, si ferma.
    """
    lati = [d for d in (1, -1) if controlla_scelta(ds, scelta[d][0], scelta[d][1], d,
                                                    occorrenze=occorrenze_train(ev.get(d)))]
    if not lati:
        raise ValueError("Scegli i trigger di almeno un lato nella cella 4 (TRIGGER_LONG o TRIGGER_SHORT, "
                         "con il suo H).")
    return lati


@dataclass
class Lato:
    direzione: int
    trigger: list
    H: int
    target: pd.DataFrame
    idx_train: pd.DatetimeIndex
    idx_test: pd.DatetimeIndex
    idx_oos: pd.DatetimeIndex
    X: pd.DataFrame
    quota: float = 0.8
    extra: dict = field(default_factory=dict)

    @property
    def nome(self):
        return NOMI_LATO[self.direzione]

    @property
    def y(self):
        return self.target["TARGET"]

    @property
    def rend(self):
        return rendimenti_trigger(self.target)


def feature_lato(ds, trigger, df_btc=None) -> pd.DataFrame:
    """Il catalogo del lavoro C sul dataset, con la famiglia 11 calcolata sui trigger del lato."""
    from feature.calcolo import calcola_feature
    trigger_lato = ds.df[list(trigger)].astype(bool).any(axis=1)
    return calcola_feature(ds.df, ds.timeframe, trigger_lato=trigger_lato, df_btc=df_btc)


def prepara_lato(ds, direzione: int, trigger, H: int, quota: float = 0.8, feat=None) -> Lato:
    """
    Target del lato (passo 5), divisione train/test con purge ed embargo sull'H del lato e X
    (passo 6). Con `feat` (lavoro D) si tolgono prima i trigger nel riscaldamento delle feature.
    """
    nome = NOMI_LATO[direzione]
    target = costruisci_target(ds.df, list(trigger), direzione, H, costi=ds.costi)
    print(f"===== {nome} · H = {H} barre ({ore(H, ds.timeframe)}) =====")
    extra = {}
    if feat is not None:
        from feature.calcolo import riscaldamento
        _, barre = riscaldamento(feat)
        inizio = ds.df.index[min(barre, len(ds.df) - 1)]
        tolti = int((target.index < inizio).sum())
        target = target.loc[target.index >= inizio]
        extra = {"feat": feat, "inizio_feature": inizio, "tolti_riscaldamento": tolti}
        print(f"riscaldamento delle feature: {barre} barre, fino al {inizio:%d/%m/%Y %H:%M} UTC · "
              f"trigger tolti dal target (per il modello base e per quello con la selezione): {tolti}")
    print(f"target: {len(target)} trigger · classe 1 {target['TARGET'].mean():.1%} · "
          f"bp netti medi {target['netto_bp'].mean():.2f}")
    idx_train, idx_test = split_train_test(target, ds.df_is, horizon=H, quota=quota)
    X = costruisci_X(ds.df, target, list(trigger), ds.filtri)
    return Lato(direzione=direzione, trigger=list(trigger), H=int(H), target=target, idx_train=idx_train,
                idx_test=idx_test, idx_oos=indici_oos(target, ds.df_is), X=X, quota=quota, extra=extra)


NOTA_TRIGGER_INSIEME = ("Un trigger che scatta insieme a un altro conta per entrambi: la somma delle "
                        "occorrenze può superare la riga «tutti».")


def tabella_trigger_train(lato: Lato) -> pd.DataFrame:
    """
    Passo 5: per ogni trigger scelto, SOLO sulle righe di train del target: occorrenze, bp lordi
    medi, bp netti medi, % classe 1; piu' la riga «tutti». Un trigger che scatta insieme a un
    altro conta per entrambi (NOTA_TRIGGER_INSIEME).
    """
    t = lato.target.loc[lato.idx_train]
    X = lato.X.loc[lato.idx_train]

    def riga(r):
        return {"occorrenze": len(r), "bp_lordi_medi": r["lordo_bp"].mean(), "bp_netti_medi": r["netto_bp"].mean(),
                "classe_1_%": 100 * r["TARGET"].mean()}
    righe = {trig: riga(t[X[trig].astype(bool)]) for trig in lato.trigger}
    righe["tutti"] = riga(t)
    tab = pd.DataFrame(righe).T
    tab.index.name = f"{lato.nome} · train"
    return tab.astype({"occorrenze": int})


def seleziona_lato(ds, lato: Lato, con_unicita: bool = True, peso_rendimento: bool = False, seme: int = 0,
                   collaudo: bool = False, macchina=None):
    """
    Feature selection del lato sul solo train (passo 7b): candidate = le feature del catalogo
    calcolate + i filtri del Lab (F* e VWAP_*); l'identita' del trigger resta sempre. Stampa
    i conteggi e il tempo; registra la selezione. Ritorna (Selezione, Lato con la X scelta:
    stesse righe, stessi train/test/OOS).
    """
    from selezione import selezione
    if "feat" not in lato.extra:
        raise ValueError("prepara_lato senza feat: servono le feature del lato (feature_lato)")
    feat = lato.extra["feat"]
    filtri = [c for c in lato.X.columns if c not in lato.trigger]
    X_cand = pd.concat([lato.X[lato.trigger + filtri], feat.loc[lato.target.index]], axis=1)
    inizio_test = ds.df_is.index[barra_confine(ds.df_is, lato.quota)]
    sel = selezione(X_cand, lato.y, ds.df, lato.idx_train, inizio_test, lato.H, sempre=lato.trigger, seme=seme,
                    con_unicita=con_unicita, peso_rendimento=peso_rendimento, netto_bp=lato.target["netto_bp"])
    print(f"===== {lato.nome} · selezione sul train: {len(lato.idx_train)} trigger, "
          f"{X_cand.shape[1] - len(lato.trigger)} candidate + 10 sonde, K = {sel.K} · {sel.secondi:.1f} s =====")
    print(" · ".join(f"{k}: {v}" for k, v in sel.conteggio.items()))
    if not sel.tenute:
        riga = "!" * 78
        print(f"{riga}\n!!! {lato.nome}: la selezione non ha tenuto nessuna feature, il modello non distingue "
              f"i trigger.\n!!! Ai passi 8 e 9 per questo lato solo «tutti i trigger», senza fasce né soglia.\n{riga}")
    lato_sel = dataclasses.replace(lato, X=X_cand[lato.trigger + sel.tenute],
                                   extra={**lato.extra, "selezione": sel})
    registro.aggiungi(ds.cartella, ds.symbol + "_" + ds.timeframe, {
        "tipo": "selezione", **_base_voce(ds, lato, collaudo), "candidate": X_cand.shape[1] - len(lato.trigger),
        "tenute": sel.tenute, "K": sel.K, "conteggio": sel.conteggio, "secondi": round(sel.secondi, 1),
        "seme": seme, "regola": "B: 3 giri su 4 sopra zero e sopra la sonda migliore del giro",
        "esiti": sel.tabella["esito"].to_dict(),
        "trigger_tolti_riscaldamento": lato.extra.get("tolti_riscaldamento")}, macchina=macchina)
    return sel, lato_sel


def confronto_base_selezione(ds, lato: Lato, modello, lato_sel: Lato, modello_sel, n_fasce: int = 5,
                             collaudo: bool = False, macchina=None) -> pd.DataFrame:
    """
    AUC e fasce sul test del modello base e di quello con la selezione, uno accanto all'altro.
    Solo un riferimento: il modello che va avanti e' sempre quello con la selezione (regola
    fissata prima). Si registra come prova.
    """
    if not (lato.target.index.equals(lato_sel.target.index) and lato.idx_train.equals(lato_sel.idx_train)
            and lato.idx_test.equals(lato_sel.idx_test)):
        raise ValueError("base e selezione devono usare le stesse righe di target, train e test")
    import contextlib
    import io
    tab, auc = {}, {}
    for nome, l, m in (("base", lato, modello), ("selezione", lato_sel, modello_sel)):
        with contextlib.redirect_stdout(io.StringIO()):
            tab[nome] = valuta(m, l.X, l.y, l.rend, l.idx_train, l.idx_test, n_fasce=n_fasce)
        auc[nome] = {n: float(roc_auc_score(l.y.loc[i], m.predict_proba(l.X.loc[i])[:, 1]))
                     for n, i in (("train", l.idx_train), ("test", l.idx_test))}
    fianco = pd.concat({k: v[["trade", "vincenti_%", "bp_medi_netti"]] for k, v in tab.items()}, axis=1)
    print(f"===== {lato.nome} · test: modello base (riferimento) e modello con la selezione =====")
    print(f"AUC base      train {auc['base']['train']:.3f}   test {auc['base']['test']:.3f}   "
          f"({lato.X.shape[1]} colonne)")
    print(f"AUC selezione train {auc['selezione']['train']:.3f}   test {auc['selezione']['test']:.3f}   "
          f"({lato_sel.X.shape[1]} colonne)")
    print(fianco.to_string())
    registro.aggiungi(ds.cartella, ds.symbol + "_" + ds.timeframe, {
        "tipo": "confronto", **_base_voce(ds, lato, collaudo), "auc": auc,
        "colonne": {"base": list(lato.X.columns), "selezione": list(lato_sel.X.columns)},
        "fasce_test": {k: v.reset_index().to_dict(orient="records") for k, v in tab.items()}}, macchina=macchina)
    return fianco


def _base_voce(ds, lato, collaudo):
    import collegamento_v4
    from engine.studio import nome_dataset
    return {"collaudo": bool(collaudo), "v4": collegamento_v4.versione(), "ml": registro.versione_ml(),
            "dataset": nome_dataset(ds.symbol, ds.timeframe), "lato": lato.nome, "H": lato.H,
            "trigger": lato.trigger, "quota_train": lato.quota}


def allena(ds, lato: Lato, con_unicita: bool = True, peso_rendimento: bool = False, parametri=None,
           collaudo: bool = False, macchina=None):
    """
    Modello del lato sul solo train (passo 7): AUC, fasce sul test e feature piu'
    usate stampate; una voce «addestramento» nel registro. Ritorna (modello, fasce).
    """
    pesi = pesi_unicita(lato.idx_train, ds.df, lato.H) if con_unicita else None
    parametri = parametri or PARAMETRI_BASE
    modello = addestra(lato.X, lato.y, lato.idx_train, lato.idx_test, parametri=parametri, pesi=pesi,
                       peso_rendimento=peso_rendimento, netto_bp=lato.target["netto_bp"])
    print(f"===== {lato.nome} =====")
    fasce = valuta(modello, lato.X, lato.y, lato.rend, lato.idx_train, lato.idx_test)
    print(fasce.to_string())
    imp = importanza(modello, lato.X)
    print("\ncolonne piu' usate:")
    print(imp.head(10).to_string())
    auc = {n: float(roc_auc_score(lato.y.loc[i], modello.predict_proba(lato.X.loc[i])[:, 1]))
           for n, i in (("train", lato.idx_train), ("test", lato.idx_test))}
    registro.aggiungi(ds.cartella, ds.symbol + "_" + ds.timeframe, {
        "tipo": "addestramento", **_base_voce(ds, lato, collaudo),
        "colonne_X": list(lato.X.columns), "pesi": {"unicita": con_unicita, "rendimento": peso_rendimento},
        "parametri": parametri, "semi": {"lightgbm": parametri.get("random_state")},
        "righe": {"train": len(lato.idx_train), "test": len(lato.idx_test)},
        "metriche": {"auc": auc, "fasce_test": fasce.reset_index().to_dict(orient="records")},
    }, macchina=macchina)
    return modello, fasce


def nessuna_feature(lato: Lato) -> bool:
    """True se la selezione del lato non ha tenuto nessuna feature (decisione di Mattia sul §I)."""
    sel = lato.extra.get("selezione")
    return sel is not None and len(sel.tenute) == 0


def frase_nessuna_feature(lato: Lato) -> str:
    return f"{lato.nome}: la selezione non ha tenuto nessuna feature, il modello non distingue i trigger"


def _ferma_fasce(lato, modello, idx, fasce, n_fasce):
    """None se fascia e soglia si possono usare; altrimenti il motivo, da stampare."""
    if nessuna_feature(lato):
        return frase_nessuna_feature(lato) + ". Si mostrano solo «tutti i trigger», senza fasce né soglia."
    try:
        soglia_dal_test(modello, lato.X, idx, fasce, n_fasce)
    except FasceNonCalcolabili as e:
        return f"{lato.nome}: {e}. Si mostrano solo «tutti i trigger»."
    return None


def mostra_soglia(lato: Lato, modello, fasce=5, n_fasce: int = 5):
    """La soglia di probabilita' della fascia operativa, dal solo test (passo 8), o il motivo per cui non c'e'."""
    motivo = _ferma_fasce(lato, modello, lato.idx_test, fasce, n_fasce)
    if motivo:
        print(motivo)
        return None
    bassa, alta = soglia_dal_test(modello, lato.X, lato.idx_test, fasce, n_fasce)
    print(f"{lato.nome}: soglia di probabilità da {bassa:.3f} a {alta:.3f} (solo dal test)")
    return bassa, alta


def confronti(ds, lato: Lato, modello, fasce=5, n_fasce: int = 5, min_trade: int = 100,
              n_estrazioni: int = 1000, seme: int = 0):
    """
    I due confronti fissi sul test (D3), stampati. Il miglior filtro si sceglie sul train.
    Senza feature tenute, o senza fasce calcolabili, si stampa solo «tutti i trigger del test».
    """
    print(f"===== {lato.nome} · confronti sul test (solo stampati) =====")
    motivo = _ferma_fasce(lato, modello, lato.idx_test, fasce, n_fasce)
    if motivo:
        print(motivo)
        netto = lato.target.loc[lato.idx_test, "netto_bp"]
        tab = pd.DataFrame([{"": "tutti i trigger del test", "trade": len(netto), "presi_%": 100.0,
                             "vincenti_%": round(100 * float((netto > 0).mean()), 1),
                             "bp_medi_netti": round(float(netto.mean()), 2)}]).set_index("")
        print(tab.to_string())
        return tab, float("nan")
    filtro, tab_filtri = miglior_filtro_train(lato.target, ds.df, lato.idx_train, ds.filtri, min_trade)
    sel = selezione_test(modello, lato.X, lato.idx_test, fasce, n_fasce)
    tab, percentile = confronti_test(lato.target, ds.df, lato.idx_test, sel, filtro,
                                     n_estrazioni=n_estrazioni, seme=seme)
    print(f"miglior filtro sul train (almeno {min_trade} trigger): {filtro} "
          f"({len(tab_filtri)} filtri ammessi)")
    print(tab.to_string())
    print(f"il modello sta al {percentile:.1f}° percentile delle {n_estrazioni} estrazioni a caso (seme {seme})")
    return tab, percentile


MESSAGGIO_OOS_CHIUSO = "OOS non guardato: metti GUARDA_OOS = True quando la scelta è fatta"


def oos_permesso(guarda: bool) -> bool:
    """
    L'interruttore GUARDA_OOS (in cima al notebook). False: il passo 9 non gira, niente OOS,
    niente backtest e nessuno sguardo all'OOS nel registro; si stampa il motivo.
    """
    if guarda is True:
        return True
    riga = "=" * 78
    print(f"{riga}\n{MESSAGGIO_OOS_CHIUSO}.\nIl notebook si ferma qui: nessuno sguardo all'OOS nel registro.\n{riga}")
    return False


def guarda_oos(ds, lato: Lato, modello, fasce=5, n_fasce: int = 5, collaudo: bool = False, macchina=None):
    """
    La tabella OOS con la soglia dal test (passo 9). Si registra come sguardo all'OOS.
    Senza feature tenute, solo «test tutti» e «OOS tutti».
    """
    print(f"===== {lato.nome} · OOS =====")
    solo = nessuna_feature(lato)
    if solo:
        print(frase_nessuna_feature(lato) + ". Si mostrano solo «tutti i trigger», senza fasce né soglia.")
    tab = valuta_oos(modello, lato.X, lato.y, lato.rend, lato.idx_test, lato.idx_oos, fasce=fasce,
                     n_fasce=n_fasce, solo_tutti=solo)
    print(tab.to_string())
    registro.aggiungi(ds.cartella, ds.symbol + "_" + ds.timeframe, {
        "tipo": "oos", "cosa": "tabella", **_base_voce(ds, lato, collaudo), "fasce": fasce,
        "solo_tutti": bool(solo or len(tab) == 2),
        "risultato": tab.reset_index().to_dict(orient="records")}, macchina=macchina)
    return tab


def buy_hold_oos(ds) -> pd.Series:
    """Il buy & hold sull'OOS del backtest (dalla prima barra OOS alla fine), in bp, meno il costo di un trade."""
    from backtest_oos import curva_buy_hold
    return curva_buy_hold(ds.df.iloc[len(ds.df_is):], ds.costo_bp()["totale_bp"])


def backtest(ds, lati: dict, modelli: dict, fasce=5, n_fasce: int = 5, collaudo: bool = False, macchina=None):
    """
    Backtest OOS sul motore veloce di V4 (passo 9): «modello» (trigger nella fascia
    operativa) e «tutti i trigger», per LONG, SHORT e COMBINATO. Si registra.
    Un lato senza feature tenute, o senza fasce calcolabili, ha solo «tutti i trigger»;
    il COMBINATO «modello» si mostra solo se tutti i lati hanno il modello.
    """
    segnali = {"modello": {}, "tutti i trigger": {}}
    senza_modello = []
    for d, lato in lati.items():
        segnali["tutti i trigger"][d] = lato.idx_oos
        motivo = _ferma_fasce(lato, modelli[d], lato.idx_test, fasce, n_fasce)
        if motivo:
            print(f"backtest · {motivo}")
            senza_modello.append(lato.nome)
            continue
        segnali["modello"][d] = trigger_in_fascia(modelli[d], lato.X, lato.idx_test, lato.idx_oos, fasce, n_fasce)
    H = {d: lati[d].H if d in lati else None for d in (1, -1)}
    if len(senza_modello) == len(lati):
        segnali.pop("modello")
    risultati = {nome: esegui_backtest(ds.df, s.get(1), s.get(-1), H[1], H[-1], ds.costi)
                 for nome, s in segnali.items()}
    inizio = len(ds.df_is)
    tab = tabella_metriche(risultati, inizio, len(ds.df), buy_hold=buy_hold_oos(ds))
    if "modello" in risultati and senza_modello:
        via = [(l, "modello") for l in senza_modello + ["COMBINATO"]]
        tab = tab.drop(index=[v for v in via if v in tab.index])
        vuoto = risultati["modello"]["trade"]["LONG"].iloc[0:0]
        for l in senza_modello + ["COMBINATO"]:
            risultati["modello"]["trade"][l] = vuoto
    for d, lato in lati.items():
        registro.aggiungi(ds.cartella, ds.symbol + "_" + ds.timeframe, {
            "tipo": "oos", "cosa": "backtest", **_base_voce(ds, lato, collaudo), "fasce": fasce,
            "solo_tutti": lato.nome in senza_modello,
            "risultato": tab.xs(lato.nome, level="lato").reset_index().to_dict(orient="records")},
            macchina=macchina)
    return tab, risultati
