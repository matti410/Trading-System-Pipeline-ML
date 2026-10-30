"""
Le funzioni delle celle del notebook, lavoro B (30/10/2026, docs/02 §B e §F).

  ore(H, timeframe) -> str
  orizzonte_event_study(timeframe) -> int
  barre_train(ds, quota=0.8) -> DataFrame
  event_study_train(ds, direzione, H, quota=0.8, trigger=None) -> EventStudy
  sintesi_event_study(ev, timeframe) -> DataFrame
  controlla_scelta(ds, trigger, H, direzione) -> bool
  prepara_lato(ds, direzione, trigger, H, quota=0.8) -> Lato
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
  registro (registro.py, D4).
"""
from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import roc_auc_score

import registro
from backtest_oos import esegui_backtest, tabella_metriche, trigger_in_fascia
from confronti import confronti_test, miglior_filtro_train, selezione_test
from modello import PARAMETRI_BASE, addestra, importanza, rendimenti_trigger, valuta
from oos import indici_oos, valuta_oos
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


def sintesi_event_study(ev, timeframe: str) -> pd.DataFrame:
    """
    La sintesi di V4, nell'ordine di V4, con le ore accanto alla barra del picco e un segno
    sui trigger con meno occorrenze nel train del minimo del Lab. Non toglie nessun trigger.
    """
    from engine.vetrina import MIN_TRIGGER
    colonne = ["candidato", "trades", "barra_picco", "picco_pct", "incertezza_pct", "volte_incertezza",
               "oltre_rumore", "a_fine_pct", "vs_mercato_pct"]
    s = ev.sintesi[[c for c in colonne if c in ev.sintesi.columns]].copy()
    s.insert(s.columns.get_loc("barra_picco") + 1, "ore_picco", [ore(b, timeframe) for b in s["barra_picco"]])
    senza = [c for c in getattr(ev, "candidati_richiesti", []) if c not in set(s["candidato"])]
    if senza:
        s = pd.concat([s, pd.DataFrame({"candidato": senza, "trades": 0})], ignore_index=True)
    s.insert(s.columns.get_loc("trades") + 1, f"sotto_minimo_Lab_{MIN_TRIGGER}",
             ["◄ sotto il minimo" + (" (nessuna occorrenza: nessuna curva)" if n == 0 else "")
              if n < MIN_TRIGGER else "" for n in s["trades"]])
    return s


def grafico_event_study(ev, titolo: str = ""):
    """
    Le curve dell'event study con il grafico di V4 (EventStudy.plot), per TUTTI i trigger
    misurati, con le occorrenze nel train accanto al nome (nella legenda). Ritorna le etichette.
    """
    import copy
    n = occorrenze_train(ev)
    etichetta = {c: f"{c} ({n.get(c, 0)} nel train)" for c in ev.curve.columns}
    copia = copy.copy(ev)
    copia.curve = ev.curve.rename(columns=etichetta)
    copia.mercato = ev.mercato.rename(columns=etichetta)
    copia.sintesi = ev.sintesi.assign(candidato=ev.sintesi["candidato"].map(etichetta))
    nomi = list(etichetta.values())
    senza = [c for c in getattr(ev, "candidati_richiesti", list(ev.curve.columns))
             if c not in ev.curve.columns or ev.curve[c].isna().all()]
    if titolo:
        print(titolo)
    copia.plot(candidati=nomi, figsize=(13, 7.5))       # il grafico di V4: si mostra da se'
    if senza:
        print(f"senza curva (nessuna occorrenza nel train): {', '.join(senza)}")
    return nomi + [f"{c} (0 nel train, senza curva)" for c in senza if c not in ev.curve.columns]


def controlla_scelta(ds, trigger, H, direzione: int, occorrenze=None) -> bool:
    """
    Controlla la scelta di Mattia per un lato. True = il lato si usa; False = lato
    saltato (lo short sullo spot, lasciato vuoto). Si ferma se manca la scelta, se un
    nome non e' un trigger del lato, se H non e' un numero di barre >= 1, o se si
    chiede lo short sullo spot.

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
    if not trigger or H is None:
        raise ValueError(f"Scrivi TRIGGER_{nome} (una lista di nomi) e H_{nome} (in barre) nella cella 4, "
                         "guardando l'event study qui sopra. Finché mancano, il notebook si ferma qui.")
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


def prepara_lato(ds, direzione: int, trigger, H: int, quota: float = 0.8) -> Lato:
    """Target del lato (passo 5), divisione train/test con purge ed embargo sull'H del lato e X (passo 6)."""
    nome = NOMI_LATO[direzione]
    target = costruisci_target(ds.df, list(trigger), direzione, H, costi=ds.costi)
    print(f"===== {nome} · H = {H} barre ({ore(H, ds.timeframe)}) =====")
    print(f"target: {len(target)} trigger · classe 1 {target['TARGET'].mean():.1%} · "
          f"bp netti medi {target['netto_bp'].mean():.2f}")
    idx_train, idx_test = split_train_test(target, ds.df_is, horizon=H, quota=quota)
    X = costruisci_X(ds.df, target, list(trigger), ds.filtri)
    return Lato(direzione=direzione, trigger=list(trigger), H=int(H), target=target, idx_train=idx_train,
                idx_test=idx_test, idx_oos=indici_oos(target, ds.df_is), X=X, quota=quota)


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


def confronti(ds, lato: Lato, modello, fasce=5, n_fasce: int = 5, min_trade: int = 100,
              n_estrazioni: int = 1000, seme: int = 0):
    """I due confronti fissi sul test (D3), stampati. Il miglior filtro si sceglie sul train."""
    filtro, tab_filtri = miglior_filtro_train(lato.target, ds.df, lato.idx_train, ds.filtri, min_trade)
    sel = selezione_test(modello, lato.X, lato.idx_test, fasce, n_fasce)
    tab, percentile = confronti_test(lato.target, ds.df, lato.idx_test, sel, filtro,
                                     n_estrazioni=n_estrazioni, seme=seme)
    print(f"===== {lato.nome} · confronti sul test (solo stampati) =====")
    print(f"miglior filtro sul train (almeno {min_trade} trigger): {filtro} "
          f"({len(tab_filtri)} filtri ammessi)")
    print(tab.to_string())
    print(f"il modello sta al {percentile:.1f}° percentile delle {n_estrazioni} estrazioni a caso (seme {seme})")
    return tab, percentile


def guarda_oos(ds, lato: Lato, modello, fasce=5, n_fasce: int = 5, collaudo: bool = False, macchina=None):
    """La tabella OOS con la soglia dal test (passo 9). Si registra come sguardo all'OOS."""
    print(f"===== {lato.nome} · OOS =====")
    tab = valuta_oos(modello, lato.X, lato.y, lato.rend, lato.idx_test, lato.idx_oos, fasce=fasce, n_fasce=n_fasce)
    print(tab.to_string())
    registro.aggiungi(ds.cartella, ds.symbol + "_" + ds.timeframe, {
        "tipo": "oos", "cosa": "tabella", **_base_voce(ds, lato, collaudo), "fasce": fasce,
        "risultato": tab.reset_index().to_dict(orient="records")}, macchina=macchina)
    return tab


def backtest(ds, lati: dict, modelli: dict, fasce=5, n_fasce: int = 5, collaudo: bool = False, macchina=None):
    """
    Backtest OOS sul motore veloce di V4 (passo 9): «modello» (trigger nella fascia
    operativa) e «tutti i trigger», per LONG, SHORT e COMBINATO. Si registra.
    """
    segnali = {"modello": {}, "tutti i trigger": {}}
    for d, lato in lati.items():
        segnali["modello"][d] = trigger_in_fascia(modelli[d], lato.X, lato.idx_test, lato.idx_oos, fasce, n_fasce)
        segnali["tutti i trigger"][d] = lato.idx_oos
    H = {d: lati[d].H if d in lati else None for d in (1, -1)}
    risultati = {nome: esegui_backtest(ds.df, s.get(1), s.get(-1), H[1], H[-1], ds.costi)
                 for nome, s in segnali.items()}
    inizio = len(ds.df_is)
    tab = tabella_metriche(risultati, inizio, len(ds.df))
    for d, lato in lati.items():
        registro.aggiungi(ds.cartella, ds.symbol + "_" + ds.timeframe, {
            "tipo": "oos", "cosa": "backtest", **_base_voce(ds, lato, collaudo), "fasce": fasce,
            "risultato": tab.xs(lato.nome, level="lato").reset_index().to_dict(orient="records")},
            macchina=macchina)
    return tab, risultati
