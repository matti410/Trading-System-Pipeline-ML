"""
Event Study — andamento medio del prezzo dopo il trigger.

DOMANDA A CUI RISPONDE
----------------------
Ogni volta che una condizione di ingresso e' vera, come si muove il prezzo
nelle barre successive? In media, barra per barra, fino a un orizzonte H.

Non e' un backtest di strategia: non c'e' capitale, non c'e' equity, non ci
sono posizioni. E' una MISURA del segnale, pensata per essere puntata su
qualunque asset e qualunque timeframe senza ricalibrature. Ogni singola
occorrenza del trigger viene misurata (nessun diradamento, nessuna
occorrenza scartata perche' "siamo gia' dentro"), perche' la domanda
riguarda l'evento, non la gestione del conto.

CONVENZIONE DI PREZZO
---------------------
Trigger vero sulla barra i (calcolato su Close[i] e indicatori[i]).
  ingresso  -> Open[i+1]     (execution lag 1: non si opera sullo stesso
                              Close che ha generato il segnale)
  a k barre -> Close[i+k]    con k = 1 ... H

  variazione(k) = (Close[i+k] / Open[i+1] - 1) * direction

`direction` viene dal registry (+1 long, -1 short), quindi la curva e'
sempre "a favore del trade": sale = il trade guadagna, per entrambi i lati.

LA LINEA DI RIFERIMENTO
-----------------------
La stessa curva calcolata su TUTTE le barre, senza nessuna condizione. E' il
movimento medio dell'asset su quell'orizzonte — il drift, la "natura dello
strumento". Non viene sottratta: viene disegnata accanto. Su una coppia
valutaria e' quasi piatta; su un asset con drift (crypto, indici) sale, e
senza quel riferimento la curva di una entry long sembrerebbe brava quando
invece sta solo seguendo l'asset.

LE DUE FASCE
------------
  dispersione  quanto ballano i SINGOLI trade (percentili, default 25-75).
               Larga attorno a una media piccola = la media e' fatta da
               pochi casi estremi, non da un comportamento regolare.

  incertezza   quanto balla la MEDIA (errore standard). Risponde alla
               domanda "questo numero e' grande?" senza bisogno di sapere
               in anticipo quali sono le grandezze tipiche dell'asset che
               si sta guardando. Cresce con la radice dell'orizzonte e
               cala con la radice del numero di trigger.
               Dal 24/9/2026 tiene conto dei trigger VICINI, che
               condividono le stesse candele e non valgono come prove
               indipendenti: vedi _errore_standard_sovrapposti.

LA SOGLIA DI RUMORE (24/9/2026)
-------------------------------
La sintesi e' una classifica: prende il picco di ogni curva e ordina le
entry. Provando 52 entry, la prima sembra buona anche su dati casuali. La
riga stampata sotto la tabella, e la colonna `oltre_rumore`, dicono oltre
quale |volte_incertezza| il risultato non e' piu' spiegabile dal caso
(engine/giudizio.py, soglia_rumore_orizzonte). E' la stessa idea della
soglia della ricerca dei filtri, adattata al fatto che qui si sceglie
anche la barra del picco.

AVVERTENZA PERMANENTE (vale su ogni asset)
------------------------------------------
Questo modulo misura un prezzo solo, non bid e ask. Un rientro del prezzo
nelle primissime barre dopo il trigger puo' essere l'oscillazione fra
denaro e lettera, che nel grafico si vede ma NON si incassa: per incassarla
bisognerebbe comprare in lettera e vendere in denaro, cioe' pagare proprio
quel rientro. Su strumenti con spread largo l'effetto e' molto piu' grande
che su una major FX. Un risultato concentrato nelle prime barre e di
ampiezza paragonabile allo spread va trattato come sospetto per default.

Modulo ADDITIVO: non modifica nessun file esistente dell'engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import get_entry, get_entry_direction, list_entries, get_filter
from .giudizio import soglia_rumore_orizzonte


# ========================================================================
# Dimensione del pip
# ========================================================================

def deduci_pip(prezzo_medio: float) -> float:
    """
    Sceglie la dimensione di un pip dall'ordine di grandezza del prezzo.
    Serve solo a rendere leggibile la colonna in pips a colpo d'occhio: e'
    una comodita' di lettura, non entra in nessun calcolo. Sovrascrivibile
    sempre col parametro `pip=` di run_event_study.

        prezzo < 20         -> 0.0001   (EURUSD, GBPUSD, AUDUSD...)
        20 <= prezzo < 5000 -> 0.01     (USDJPY, XAUUSD...)
        prezzo >= 5000      -> 1.0      (BTCUSD, indici)
    """
    if not np.isfinite(prezzo_medio) or prezzo_medio <= 0:
        return 1.0
    if prezzo_medio < 20:
        return 0.0001
    if prezzo_medio < 5000:
        return 0.01
    return 1.0


# ========================================================================
# Incertezza con trigger sovrapposti
# ========================================================================

def _errore_standard_sovrapposti(v: np.ndarray, pos: np.ndarray, k: int) -> float:
    """
    Errore standard della media dei rendimenti a k barre, tenendo conto che
    trigger VICINI condividono le stesse candele (24/9/2026).

    Due trade partiti a d barre di distanza, misurati a k barre, hanno in
    comune k - d candele su k: non sono due prove indipendenti. La formula
    classica (deviazione standard / radice di n) li conta come
    indipendenti e SOTTOSTIMA l'incertezza. Misurato su entry casuali: al
    posto del 5% atteso, falsi allarmi al 15% con un trigger ogni ~25 barre
    e al 91% con trigger a grappolo.

    Qui la covarianza fra due trade e' proporzionale alle candele condivise:

        var(media) = [ somma_i r_i^2 + 2 somma_{i<j} w_ij r_i r_j ] / n^2
        w_ij = max(0, 1 - |pos_i - pos_j| / k)

    dove r sono gli scarti dalla media. Senza sovrapposizioni (trigger
    distanti almeno k barre) tutti i w sono zero e il risultato coincide con
    la formula classica, a meno del fattore n/(n-1).

    v    rendimenti a k barre dei trigger validi, nell'ordine delle posizioni
    pos  posizioni (indici di barra) degli stessi trigger, crescenti
    """
    n_v = v.size
    if n_v < 2:
        return np.nan
    res = v - v.mean()
    tot = float((res ** 2).sum()) * n_v / (n_v - 1)
    d = 1
    while d < n_v:
        dist = pos[d:] - pos[:-d]
        vicini = dist < k
        if not vicini.any():
            break
        w = 1.0 - dist[vicini] / k
        tot += 2.0 * float((w * res[d:][vicini] * res[:-d][vicini]).sum())
        d += 1
    return float(np.sqrt(max(tot, 0.0))) / n_v


# ========================================================================
# Motore
# ========================================================================

def run_event_study(
    df: pd.DataFrame,
    entry_names=None,
    horizon: int = 96,
    min_trades: int = 200,
    pip: float | None = None,
    filters=None,
    close_col: str = "Close",
    open_col: str = "Open",
    banda: tuple[float, float] = (25.0, 75.0),
    verbose: bool = True,
    alpha: float = 0.05,
):
    """
    Calcola l'andamento medio del prezzo dopo il trigger, per ogni entry.

    Parametri
    ---------
    df            DataFrame di mercato (servono almeno Open e Close).
    entry_names   lista di nomi di entry, oppure None = tutte le registrate.
    horizon       quante barre in avanti guardare (H). Tutti gli orizzonti
                  intermedi si leggono sulla curva: non serve una griglia.
    min_trades    scarta le entry con meno trigger di cosi'. Serve: con
                  pochi trigger la media balla tantissimo, e siccome la
                  sintesi ordina per ampiezza del risultato sarebbero
                  proprio quelle a finire in cima. Metti 0 per tenerle
                  tutte (sapendo cosa si sta guardando).
    pip           dimensione di un pip in unita' di prezzo. None = dedotta
                  dal prezzo (vedi `deduci_pip`).
    filters       lista di filtri da applicare in AND all'entry, o None.
    alpha         famiglia della soglia di rumore (default 5%). Vedi sotto.
    banda         percentili bassi/alti della fascia di dispersione.

    Ritorna
    -------
    Un oggetto EventStudy. Vedi .sintesi (la tabella), .plot(),
    .plot_singola(), .diagnosi().
    """
    for col in (close_col, open_col):
        if col not in df.columns:
            raise ValueError(f"Colonna '{col}' non trovata nel DataFrame.")
    if horizon < 1:
        raise ValueError("horizon deve essere >= 1.")

    if entry_names is None:
        entry_names = list_entries()
    entry_names = list(entry_names)
    if not entry_names:
        raise ValueError("Nessuna entry da testare.")
    filters = list(filters or [])

    n = len(df)
    close = df[close_col].to_numpy(dtype=float)
    open_ = df[open_col].to_numpy(dtype=float)

    prezzo_ing = np.full(n, np.nan)
    prezzo_ing[: n - 1] = open_[1:]
    valido_ing = np.isfinite(prezzo_ing) & (prezzo_ing > 0)

    prezzo_medio = float(np.nanmean(close))
    pip_size = float(pip) if pip is not None else deduci_pip(prezzo_medio)
    pips_per_pct = (prezzo_medio / 100.0) / pip_size

    cache_filtri = {f: get_filter(f)(df).to_numpy(dtype=bool) for f in filters}
    maschere, scartate = {}, []
    for nome in entry_names:
        m = get_entry(nome)(df).to_numpy(dtype=bool)
        for f in filters:
            m = m & cache_filtri[f]
        m = m & valido_ing
        if int(m.sum()) < min_trades:
            scartate.append((nome, int(m.sum())))
            continue
        maschere[nome] = m

    if not maschere:
        raise ValueError(
            f"Nessuna entry raggiunge min_trades={min_trades}. "
            f"Massimo osservato: {max((c for _, c in scartate), default=0)} trigger."
        )
    attive = list(maschere)
    direzioni = {nome: int(get_entry_direction(nome)) for nome in attive}
    etichetta = {nome: (nome if not filters else f"{nome} + {' + '.join(filters)}")
                 for nome in attive}

    if verbose:
        print(f"pip = {pip_size:g} unita' di prezzo  (prezzo medio {prezzo_medio:,.2f})")
        if scartate:
            print(f"{len(scartate)} entry scartate sotto min_trades={min_trades}")

    barre = np.arange(1, horizon + 1)
    curva = pd.DataFrame(index=barre, dtype=float)
    mercato = pd.DataFrame(index=barre, dtype=float)
    b_lo = pd.DataFrame(index=barre, dtype=float)
    b_hi = pd.DataFrame(index=barre, dtype=float)
    incert = pd.DataFrame(index=barre, dtype=float)
    for d in (curva, mercato, b_lo, b_hi, incert):
        d.index.name = "barre_dal_trigger"

    acc = {nome: {"media": [], "lo": [], "hi": [], "mkt": [], "se": []}
           for nome in attive}
    n_trade = {nome: 0 for nome in attive}
    prezzo_ing_medio = {nome: float(np.nanmean(prezzo_ing[maschere[nome]]))
                        for nome in attive}

    posizioni = {nome: np.flatnonzero(maschere[nome]) for nome in attive}
    ok = {}
    for k in barre:
        fwd = np.full(n, np.nan)
        limite = n - k
        if limite > 0:
            fwd[:limite] = close[k:] / prezzo_ing[:limite] - 1.0
        mkt_long = float(np.nanmean(fwd)) if np.isfinite(fwd).any() else np.nan

        for nome in attive:
            d = direzioni[nome]
            v = fwd[maschere[nome]]
            ok[nome] = np.isfinite(v)
            v = v[ok[nome]]
            if v.size == 0:
                for chiave in ("media", "lo", "hi", "se"):
                    acc[nome][chiave].append(np.nan)
            else:
                v = v * d
                acc[nome]["media"].append(float(v.mean()))
                lo, hi = np.percentile(v, banda)
                acc[nome]["lo"].append(float(lo))
                acc[nome]["hi"].append(float(hi))
                acc[nome]["se"].append(
                    _errore_standard_sovrapposti(v, posizioni[nome][ok[nome]], k))
                n_trade[nome] = max(n_trade[nome], int(v.size))
            acc[nome]["mkt"].append(mkt_long * d)

        if verbose and (k % max(1, horizon // 8) == 0 or k == horizon):
            print(f"  barra {k:>3}/{horizon}")

    for nome in attive:
        e = etichetta[nome]
        curva[e] = np.asarray(acc[nome]["media"]) * 100.0
        mercato[e] = np.asarray(acc[nome]["mkt"]) * 100.0
        b_lo[e] = np.asarray(acc[nome]["lo"]) * 100.0
        b_hi[e] = np.asarray(acc[nome]["hi"]) * 100.0
        incert[e] = np.asarray(acc[nome]["se"]) * 100.0

    coda = int(np.ceil(horizon * 0.75))
    righe = []
    for nome in attive:
        e = etichetta[nome]
        c, s = curva[e], incert[e]
        if not c.notna().any():
            continue
        i_pk = int(c.abs().idxmax())
        picco = float(c.loc[i_pk])
        se_pk = float(s.loc[i_pk])
        volte = picco / se_pk if (np.isfinite(se_pk) and se_pk > 0) else np.nan
        pips = picco / 100.0 * prezzo_ing_medio[nome] / pip_size

        # Il test "il picco cade in fondo?" si fa sull'ECCESSO sul mercato,
        # non sulla curva grezza. Su un asset con drift forte (crypto,
        # indici) la curva grezza sale fino all'ultima barra per costruzione
        # — e' l'asset che sale, non il segnale — e il test darebbe
        # "deriva casuale" anche davanti a un segnale ottimo. L'eccesso
        # toglie il drift comune e lascia solo cio' che il trigger aggiunge.
        ecc = (c - mercato[e]).abs()
        i_ecc = int(ecc.idxmax()) if ecc.notna().any() else i_pk
        righe.append({
            "candidato": e,
            "direction": direzioni[nome],
            "trades": n_trade[nome],
            "barra_picco": i_pk,
            "picco_pips": pips,
            "picco_pct": picco,
            "incertezza_pct": se_pk,
            "volte_incertezza": volte,
            "a_fine_pct": float(c.iloc[-1]),
            "vs_mercato_pct": picco - float(mercato[e].loc[i_pk]),
            "barra_picco_netto": i_ecc,
            "picco_in_coda": i_ecc >= coda,
        })
    sintesi = (pd.DataFrame(righe)
               .sort_values("volte_incertezza", key=abs, ascending=False)
               .reset_index(drop=True))

    # --- soglia di rumore (24/9/2026) -----------------------------------
    # La classifica sceglie il migliore due volte: la barra del picco
    # sull'orizzonte e l'entry fra le k misurate. La soglia dice oltre
    # quale |volte_incertezza| il primo della classifica non e' piu'
    # spiegabile dal caso. k = entry misurate in QUESTA chiamata: le metro
    # vanno lasciate fuori passando entry_names (sono un righello).
    k = len(sintesi)
    soglia = soglia_rumore_orizzonte(k, horizon, alpha) if k else np.nan
    if k:
        sintesi["oltre_rumore"] = sintesi["volte_incertezza"].abs() > soglia

    if verbose:
        print(f"\n{len(attive)} entry · orizzonte {horizon} barre · "
              f"{sum(n_trade.values()):,} trigger misurati")
        if k:
            print(f"soglia di rumore per {k} entry × picco su {horizon} barre "
                  f"(famiglia {alpha:.0%}): |volte_incertezza| > {soglia:.2f}  —  "
                  f"entry oltre: {int(sintesi['oltre_rumore'].sum())}. "
                  f"Conta solo le prove di questa chiamata, e' un pavimento")

    return EventStudy(curve=curva, mercato=mercato, banda_bassa=b_lo,
                      banda_alta=b_hi, incertezza=incert, sintesi=sintesi,
                      horizon=horizon, banda=banda, pip=pip_size,
                      pips_per_pct=pips_per_pct, scartate=scartate,
                      soglia_rumore=soglia, k=k, alpha=alpha)


# ========================================================================
# Contenitore, diagnostica, grafici
# ========================================================================

class EventStudy:
    """Risultato di run_event_study. Vedi .sintesi, .plot(), .diagnosi()."""

    def __init__(self, curve, mercato, banda_bassa, banda_alta, incertezza,
                 sintesi, horizon, banda, pip, pips_per_pct, scartate,
                 soglia_rumore=np.nan, k=0, alpha=0.05):
        self.curve = curve
        self.mercato = mercato
        self.banda_bassa = banda_bassa
        self.banda_alta = banda_alta
        self.incertezza = incertezza
        self.sintesi = sintesi
        self.horizon = horizon
        self.banda = banda
        self.pip = pip
        self.pips_per_pct = pips_per_pct
        self.scartate = scartate
        # soglia di rumore della classifica (vedi run_event_study)
        self.soglia_rumore = soglia_rumore
        self.k = k
        self.alpha = alpha

    def __repr__(self):
        return (f"<EventStudy: {self.curve.shape[1]} entry, "
                f"orizzonte {self.horizon} barre, pip={self.pip:g}>")

    def in_pips(self, df_pct):
        """Converte una qualunque delle tabelle in % (curve, incertezza,
        bande) nella stessa tabella espressa in pips."""
        return df_pct * self.pips_per_pct

    # ---------------------------------------------------------------
    def diagnosi(self):
        """
        L'orizzonte scelto e' adatto a QUESTO asset e timeframe?

        Il criterio non dipende dall'asset. Se il prezzo dopo il trigger si
        muovesse a caso, la sua distanza dal punto di partenza crescerebbe
        con la radice del tempo: il massimo cadrebbe quasi sempre
        sull'ULTIMA barra della finestra. Un segnale vero fa il contrario —
        esprime l'effetto presto e poi si esaurisce, e il picco cade DENTRO
        la finestra.

        Quindi: molti picchi ammassati nell'ultimo quarto = la finestra e'
        piu' lunga della memoria dell'asset, e si sta misurando deriva
        casuale. Accorciare `horizon` e rileggere.
        """
        s = self.sintesi
        if s.empty:
            print("Nessuna entry da diagnosticare.")
            return
        quota = float(s["picco_in_coda"].mean())
        coda = int(np.ceil(self.horizon * 0.75))
        mediana = float(s["volte_incertezza"].abs().median())
        massimo = float(s["volte_incertezza"].abs().max())

        print(f"orizzonte impostato          : {self.horizon} barre")
        print(f"picchi (netti) nell'ultimo quarto: {quota:.0%} "
              f"({int(s['picco_in_coda'].sum())}/{len(s)}, oltre la barra {coda})")
        print(f"|picco| / incertezza, mediana: {mediana:.2f}")
        print(f"|picco| / incertezza, massimo: {massimo:.2f}")
        if np.isfinite(self.soglia_rumore):
            oltre = int((s["volte_incertezza"].abs() > self.soglia_rumore).sum())
            print(f"soglia di rumore ({self.k} entry × picco su {self.horizon} barre): "
                  f"{self.soglia_rumore:.2f}  —  entry oltre: {oltre}")
        print()
        if quota >= 0.5:
            print("I picchi si ammassano in fondo alla finestra: e' il")
            print("comportamento di una deriva casuale, non di un segnale.")
            print(f"Prova a rileggere con horizon={max(4, self.horizon // 4)}")
            print("e guarda se la forma delle curve cambia.")
        elif quota >= 0.25:
            print("Una parte dei picchi cade in fondo alla finestra: la coda")
            print("del grafico va letta con prudenza, la parte iniziale no.")
        else:
            print("I picchi cadono dentro la finestra: l'orizzonte e'")
            print("adeguato a questo asset e timeframe.")

    # ---------------------------------------------------------------
    def plot(self, candidati=None, top=8, pips=False, figsize=(12, 6.5)):
        """Tutte le curve insieme, per confrontare le FORME."""
        import matplotlib.pyplot as plt

        cols = self._scegli(candidati, top)
        dati = self.in_pips(self.curve) if pips else self.curve
        mkt = self.in_pips(self.mercato) if pips else self.mercato
        unita = "pips" if pips else "%"

        fig, ax = plt.subplots(figsize=figsize)
        for c in cols:
            ax.plot(dati.index, dati[c], lw=1.7, label=c)
        ax.plot(mkt.index, mkt[cols[0]], color="black", ls="--", lw=1.2,
                label="mercato (tutte le barre)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("barre dal trigger")
        ax.set_ylabel(f"variazione media del prezzo  ({unita})")
        ax.set_title("Andamento medio del prezzo dopo il trigger")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------
    def plot_singola(self, candidato, pips=False, dispersione=True,
                     figsize=(9, 5.5)):
        """
        Una sola entry, con due fasce:
          chiara  = dispersione dei singoli trade
          scura   = incertezza sulla media (+/- 2 errori standard)

        La fascia scura e' quella che risponde a "questo numero e' grande?".
        Se comprende lo zero, il movimento medio non e' distinguibile dal
        nulla a quell'orizzonte.
        """
        import matplotlib.pyplot as plt

        if candidato not in self.curve.columns:
            raise KeyError(f"'{candidato}' non presente. "
                           f"Disponibili: {list(self.curve.columns)}")
        f = self.pips_per_pct if pips else 1.0
        unita = "pips" if pips else "%"
        x = self.curve.index
        c = self.curve[candidato] * f
        se = self.incertezza[candidato] * f

        fig, ax = plt.subplots(figsize=figsize)
        if dispersione:
            ax.fill_between(x, self.banda_bassa[candidato] * f,
                            self.banda_alta[candidato] * f,
                            color="tab:blue", alpha=.10,
                            label=f"singoli trade "
                                  f"({self.banda[0]:g}°-{self.banda[1]:g}° pct)")
        ax.fill_between(x, c - 2 * se, c + 2 * se, color="tab:blue", alpha=.30,
                        label="incertezza sulla media (±2 errori standard)")
        ax.plot(x, c, color="tab:blue", lw=2.2, label=candidato)
        ax.plot(x, self.mercato[candidato] * f, color="black", ls="--", lw=1.2,
                label="mercato (tutte le barre)")
        ax.axhline(0, color="black", lw=0.8)

        riga = self.sintesi[self.sintesi["candidato"] == candidato]
        if len(riga):
            r = riga.iloc[0]
            ax.axvline(r["barra_picco"], color="tab:blue", ls=":", lw=1)
            ax.set_title(f"{candidato}  ·  {int(r['trades']):,} trigger  ·  "
                         f"picco {r['picco_pips']:+.2f} pips "
                         f"({r['picco_pct']:+.3f}%) alla barra "
                         f"{int(r['barra_picco'])}")
        ax.set_xlabel("barre dal trigger")
        ax.set_ylabel(f"variazione del prezzo  ({unita})")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------
    def _scegli(self, candidati, top):
        if candidati is not None:
            mancanti = [c for c in candidati if c not in self.curve.columns]
            if mancanti:
                raise KeyError(f"Non presenti: {mancanti}")
            return list(candidati)
        return list(self.sintesi["candidato"].head(top))
