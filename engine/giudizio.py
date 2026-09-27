"""
Giudizio — la colonna che decide, e la scheda KPI di una strategia.

PERCHE' NON PIU' `guadagno_sharpe`
-----------------------------------
Lo `Sharpe Ratio` di backtesting.py e' calcolato sull'EQUITY ricampionata
a frequenza giornaliera (rendimento annualizzato / volatilita'
annualizzata). E' quindi una metrica di CALENDARIO, non di trade:

  - i giorni senza trade entrano come rendimento zero e comprimono la
    volatilita';
  - un filtro che taglia i trade da 995 a 500 cambia numeratore E
    denominatore per ragioni che non hanno a che fare con la qualita'
    dell'edge;
  - dipende dal sizing, che qui e' convenzionale (tutta la cassa).

Verificato: su una serie sintetica, AGGIUNGERE la commissione fa SALIRE
lo Sharpe da 24.49 a 27.16. Una metrica su cui un costo in piu' migliora
il punteggio non e' una misura pulita di bonta' del sistema.

Le metriche qui sotto lavorano sui PIPS NETTI PER TRADE: invarianti
all'esposizione, alla leva e al sizing, e confrontabili direttamente col
costo round-turn.

PERCHE' IL CONFRONTO E' `TENUTI vs SCARTATI`
---------------------------------------------
Verificato sui dati reali (19/9/2026): i trade del sistema filtrato NON
sono un sottoinsieme di quelli della baseline. Su EURUSD M15,
VWAP_SESSION_CLOSE ha 500 trade di cui 58 (12%) la baseline non aveva:
scartando un trade, il filtro libera il posto e rende tradabili segnali
prima bloccati da una posizione aperta (exclusive_orders=True). Succede
con tutti i filtri, dal 3% al 12%.

Conseguenze:
  - un test APPAIATO e' illecito (gli insiemi non coincidono);
  - un test a due campioni fra "filtrato" e "baseline" e' altrettanto
    sbagliato: si sovrappongono all'88-97%, quindi non sono indipendenti.

La via pulita e' partizionare la SOLA baseline nei trade che il filtro
TIENE e in quelli che SCARTA: due gruppi disgiunti per costruzione, su
cui il test di Welch e' valido. Ed e' anche la domanda economica giusta —
i trade che tieni sono migliori di quelli che butti?

Il confronto si fa solo sul LATO su cui il filtro agisce: per un filtro
direction=+1 si confrontano i long tenuti coi long scartati, senza
diluire con gli short che quel filtro non tocca.

PERCHE' IL `t` E' LECITO QUI
-----------------------------
Il test richiede osservazioni indipendenti. Il framework impone trade non
sovrapposti in tutta la pipeline, quindi la condizione e' soddisfatta per
costruzione — e' una delle ragioni per cui quella regola valeva la pena.

LIMITE DA RICORDARE: i pips netti sono asimmetrici (lo stop taglia la
coda sinistra, il take profit e' spento). Con n nelle centinaia il
teorema del limite centrale tiene, ma il `t` va trattato come uno
STRUMENTO DI CLASSIFICA, non come un p-value esatto. Per questo accanto
c'e' `p_ev_negativo`, che e' un bootstrap e non assume normalita'.

IL PROBLEMA DEI TEST MULTIPLI
------------------------------
Provando k filtri, il migliore dei k sembra buono anche su dati casuali.
`soglia_rumore(k)` restituisce il |t| oltre il quale il massimo di k
prove non e' piu' spiegabile dal caso (correzione di Sidak, famiglia al
5%). Verificato contro simulazione Monte Carlo: k=17 -> 2.97 vs 2.97
misurato, k=42 -> 3.23 vs 3.23.

ATTENZIONE: la soglia conta solo le prove di QUELLA chiamata. Le entry
provate nella Sessione 1, e ogni rilancio della griglia, si sommano a
quel conto. E' un PAVIMENTO, non un tetto.

SCARTARE E' OGGETTIVO, APPROVARE NO
-------------------------------------
`ROADMAP_RICERCA.md`, Passo 4: "Nessuna soglia automatica decide se un
setup ha un edge sfruttabile". Resta vero, e questo modulo non lo viola:
automatizza solo la BOCCIATURA, che e' verificabile (un EV negativo, un
t sotto il rumore, dodici trade), e si ferma prima dell'approvazione, che
dipende da quanto drawdown reggi tu. Da cui i tre esiti:

    SCARTATA        almeno un criterio oggettivo fallito
    CAMPIONE CORTO  nessun criterio fallito, ma troppi pochi trade
    DA VALUTARE     tutti i criteri superati — ora decidi tu

Nessuna dipendenza oltre numpy/pandas: l'inversa della normale arriva da
`statistics.NormalDist`, nella libreria standard.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .metriche import pips_per_trade

SCARTATA = "SCARTATA"
CAMPIONE_CORTO = "CAMPIONE CORTO"
DA_VALUTARE = "DA VALUTARE"
RIFERIMENTO = "—"


# ========================================================================
# 1 · Statistiche di base sui pips per trade
# ========================================================================

def t_stat(pips) -> float:
    """
    media / (deviazione standard / radice di n).

    Invariante a esposizione, leva e sizing: dipende solo dai trade.
    Serve a rispondere "questo EV si distingue da zero?".
    """
    x = np.asarray(pips, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    sd = x.std(ddof=1)
    if not sd > 0:
        return np.nan
    return float(x.mean() / (sd / np.sqrt(x.size)))


def t_welch(a, b) -> float:
    """
    t di Welch fra due campioni DISGIUNTI e di varianza diversa.
    Calcolato a mano (nessuna dipendenza da scipy): serve la statistica,
    non il p-value — il confronto e' con `soglia_rumore`.
    """
    x = np.asarray(a, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(b, dtype=float); y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan
    se2 = x.var(ddof=1) / x.size + y.var(ddof=1) / y.size
    if not se2 > 0:
        # varianza nulla in entrambi i gruppi: separazione perfetta (non
        # capita coi trade veri, ma un NaN qui verrebbe SALTATO dal
        # verdetto e la riga passerebbe senza controllo sul rumore).
        d = float(x.mean() - y.mean())
        return 0.0 if d == 0 else float(np.sign(d) * np.inf)
    return float((x.mean() - y.mean()) / np.sqrt(se2))


def p_ev_negativo(pips, n_boot: int = 10_000, seed: int = 0) -> float:
    """
    Probabilita' che il vero EV sia negativo, in PERCENTUALE, stimata per
    bootstrap (ricampionamento con reimmissione dei trade).

    E' la "bootstrap edge probability" del corso (Lezione 9). Soglie che
    il corso indica: sopra il 5% in-sample e' un campanello; fuori
    campione meglio sotto il 10%, sotto il 5% rassicurante, sotto l'1%
    molto buono.

    A differenza del `t`, non assume normalita': su una distribuzione di
    pips asimmetrica e' la stima piu' affidabile delle due.
    """
    x = np.asarray(pips, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    medie = rng.choice(x, size=(int(n_boot), x.size), replace=True).mean(axis=1)
    return float((medie < 0).mean() * 100.0)


def ic_ev(pips, livello: float = 95.0, n_boot: int = 10_000,
          seed: int = 0) -> tuple[float, float]:
    """Intervallo di confidenza bootstrap dell'EV per trade, in pips."""
    x = np.asarray(pips, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    medie = rng.choice(x, size=(int(n_boot), x.size), replace=True).mean(axis=1)
    coda = (100.0 - livello) / 2.0
    lo, hi = np.percentile(medie, [coda, 100.0 - coda])
    return (float(lo), float(hi))


def soglia_rumore(k: int, alpha: float = 0.05) -> float:
    """
    Il |t| oltre cui il MIGLIORE di k prove non e' piu' spiegabile dal
    caso, a famiglia `alpha` (correzione di Sidak).

        soglia = Phi^-1( 1 - (1 - (1-alpha)^(1/k)) / 2 )

    Verificato contro simulazione Monte Carlo su rumore puro:
        k=17 -> 2.97 (simulato 2.97)
        k=42 -> 3.23 (simulato 3.23)

    Conta solo le prove di quella chiamata: e' un pavimento, non un tetto.
    """
    k = max(1, int(k))
    per_test = 1.0 - (1.0 - alpha) ** (1.0 / k)
    return float(NormalDist().inv_cdf(1.0 - per_test / 2.0))


_CACHE_ORIZZONTE: dict[tuple, np.ndarray] = {}


def soglia_rumore_orizzonte(k: int, horizon: int, alpha: float = 0.05,
                            n_sim: int = 1_000_000, seme: int = 0) -> float:
    """
    Come `soglia_rumore`, ma per l'EVENT STUDY, dove si sceglie DUE volte
    il migliore:

      1. per ogni entry, la barra del PICCO fra le `horizon` barre
         dell'orizzonte (`volte_incertezza` e' misurata li');
      2. fra le k entry, quella in cima alla classifica.

    Restituisce il |volte_incertezza| oltre cui il migliore di k entry,
    ciascuna col proprio picco scelto sull'orizzonte, non e' piu'
    spiegabile dal caso (famiglia `alpha`).

    Le barre dell'orizzonte sono molto correlate fra loro (la barra 20 e la
    21 condividono 20 barre di storia), quindi NON valgono `horizon` prove
    indipendenti: su rumore puro 25 barre valgono circa 7 prove, 48 circa 10.
    Per questo la soglia si ottiene per simulazione e non con una formula:

      - rumore puro: rendimenti indipendenti, la curva media dopo il
        trigger e' una passeggiata casuale; alla barra t il rapporto
        picco/incertezza e' S_t / radice(t);
      - per una entry, il massimo di |S_t / radice(t)| su t = 1..horizon;
      - fra k entry indipendenti si combina come Sidak: la soglia e' il
        quantile (1 - alpha)^(1/k) di quel massimo.

    Seme fisso: stesso input, stesso numero. La simulazione di un dato
    orizzonte si fa una volta sola per sessione (cache).

    Con horizon=1 non c'e' nessun picco da scegliere e la funzione
    restituisce esattamente `soglia_rumore(k, alpha)`.

    Vale se i trigger non si sovrappongono. Con trigger vicini (grappoli)
    l'incertezza dell'event study e' sottostimata e la soglia e' un
    PAVIMENTO. Come `soglia_rumore`, conta solo le prove di quella chiamata.
    """
    k = max(1, int(k))
    horizon = max(1, int(horizon))
    if horizon == 1:
        return soglia_rumore(k, alpha)

    chiave = (horizon, int(n_sim), int(seme))
    if chiave not in _CACHE_ORIZZONTE:
        rng = np.random.default_rng(seme)
        radici = np.sqrt(np.arange(1, horizon + 1))
        blocco = max(1, 2_000_000 // horizon)       # memoria contenuta
        massimi, rimasti = [], int(n_sim)
        while rimasti > 0:
            m = min(blocco, rimasti)
            z = np.cumsum(rng.standard_normal((m, horizon)), axis=1) / radici
            massimi.append(np.abs(z).max(axis=1))
            rimasti -= m
        _CACHE_ORIZZONTE[chiave] = np.sort(np.concatenate(massimi))

    livello = (1.0 - alpha) ** (1.0 / k)
    return float(np.quantile(_CACHE_ORIZZONTE[chiave], livello))


# ========================================================================
# 2 · Il confronto tenuti / scartati
# ========================================================================

def tenuti_scartati(trades_baseline: pd.DataFrame, pips_netti,
                    maschera_long=None, maschera_short=None) -> dict:
    """
    Partiziona i trade della BASELINE in quelli che il filtro tiene e
    quelli che scarta, sui soli lati su cui il filtro agisce.

    trades_baseline
        I trade nativi di backtesting.py della riga senza filtro.
    pips_netti
        I pips netti di quegli stessi trade, nello stesso ordine
        (engine.metriche.pips_per_trade).
    maschera_long / maschera_short
        Array booleani sull'INTERO df, il valore del filtro barra per
        barra. None sul lato dove il filtro non si applica: i trade di
        quel lato restano fuori dal confronto, non vengono contati come
        "tenuti" (conterebbero come diluizione).

    Il segnale nasce alla barra `EntryBar - 1` (l'ordine si riempie alla
    barra dopo): e' li' che il filtro viene valutato, esattamente come in
    run_filter_search_bt.

    Ritorna {"tenuti", "scartati", "n_tenuti", "n_scartati",
             "guadagno_pips", "t_guadagno"}.
    """
    vuoto = {"tenuti": np.array([]), "scartati": np.array([]),
             "n_tenuti": 0, "n_scartati": 0,
             "guadagno_pips": np.nan, "t_guadagno": np.nan}
    if trades_baseline is None or len(trades_baseline) == 0:
        return vuoto

    pips = np.asarray(pips_netti, dtype=float)
    lato = np.sign(trades_baseline["Size"].to_numpy())
    barra_segnale = trades_baseline["EntryBar"].to_numpy() - 1

    tenuti, scartati = [], []
    for segno, maschera in ((1, maschera_long), (-1, maschera_short)):
        if maschera is None:
            continue
        m = np.asarray(maschera, dtype=bool)
        sel = (lato == segno) & (barra_segnale >= 0) & (barra_segnale < m.size)
        if not sel.any():
            continue
        acceso = m[barra_segnale[sel]]
        tenuti.append(pips[sel][acceso])
        scartati.append(pips[sel][~acceso])

    if not tenuti:
        return vuoto

    T = np.concatenate(tenuti)
    S = np.concatenate(scartati) if scartati else np.array([])

    guadagno = (float(T.mean() - S.mean())
                if T.size and S.size else np.nan)
    return {"tenuti": T, "scartati": S,
            "n_tenuti": int(T.size), "n_scartati": int(S.size),
            "guadagno_pips": guadagno, "t_guadagno": t_welch(T, S)}


# ========================================================================
# 3 · Il verdetto
# ========================================================================

def verdetto(ev=None, p_neg=None, n_trades=0, t=None, soglia_t=None,
             guadagno=None, p_max: float = 5.0,
             min_trades: int = 100, ev_min: float = 0.0) -> dict:
    """
    Tre esiti. SCARTATA se un criterio OGGETTIVO fallisce; CAMPIONE CORTO
    se nessuno fallisce ma i trade sono pochi; DA VALUTARE se tutto passa
    — e da li' in poi decide chi conduce la ricerca (Passo 4).

    Il drawdown NON e' un criterio: e' tolleranza al rischio, quindi resta
    in scheda come numero da guardare, non come cancello.

    Ogni parametro passato a None disattiva il criterio corrispondente:
    cosi' la stessa funzione serve una strategia da sola (nessun
    `guadagno`, nessuna `soglia_t`) e una riga di griglia filtri.

    Ritorna {"esito", "motivo"} — `motivo` nomina il PRIMO criterio
    fallito, che e' l'informazione utile: dove guardare.
    """
    def num(x):
        return x is not None and np.isfinite(x)

    # Un confronto non calcolabile NON e' un criterio superato: senza
    # questo controllo un t o un guadagno NaN verrebbero saltati e la riga
    # arriverebbe a DA VALUTARE senza aver mai passato il test sul rumore.
    # Capita davvero: un filtro sempre vero non scarta nessun trade, quindi
    # non c'e' un gruppo con cui confrontarlo.
    # (un t infinito e' invece legittimo: separazione perfetta)
    t_valido = t is not None and not np.isnan(t)
    if soglia_t is not None and not (t_valido and num(guadagno)):
        return {"esito": SCARTATA,
                "motivo": "confronto non calcolabile (il filtro non scarta trade?)"}

    if num(ev) and ev <= ev_min:
        return {"esito": SCARTATA, "motivo": f"EV netto {ev:+.3f} pips <= {ev_min:g}"}
    if num(guadagno) and guadagno <= 0:
        return {"esito": SCARTATA, "motivo": f"guadagno {guadagno:+.3f} pips <= 0"}
    if t_valido and num(soglia_t) and abs(t) < soglia_t:
        return {"esito": SCARTATA, "motivo": f"|t| {abs(t):.2f} < soglia rumore {soglia_t:.2f}"}
    if t_valido and num(soglia_t) and t < 0:
        return {"esito": SCARTATA, "motivo": f"t {t:.2f} oltre soglia ma NEGATIVO (peggiora)"}
    if num(p_neg) and p_neg >= p_max:
        return {"esito": SCARTATA, "motivo": f"P(EV<0) {p_neg:.1f}% >= {p_max:g}%"}
    if int(n_trades) < int(min_trades):
        return {"esito": CAMPIONE_CORTO, "motivo": f"{int(n_trades)} trade < {int(min_trades)}"}
    return {"esito": DA_VALUTARE, "motivo": ""}


# ========================================================================
# 4 · La scheda KPI di una strategia
# ========================================================================

def _drawdown_chiusi(pips) -> float:
    """Max drawdown della cumulata dei soli trade chiusi, in pips."""
    x = np.asarray(pips, dtype=float)
    if x.size == 0:
        return np.nan
    eq = np.cumsum(x)
    return float((np.maximum.accumulate(eq) - eq).max())


def drawdown_montecarlo(pips, n_sim: int = 2000, seed: int = 0) -> dict:
    """
    Monte Carlo SHUFFLE: rimescola l'ordine dei trade e misura il
    drawdown di ogni sequenza. Risponde a "quanto male puo' andare una
    sequenza sfortunata dello STESSO identico sistema".

    Il totale non cambia (sono gli stessi trade): cambia il percorso, ed
    e' il percorso a determinare il drawdown.
    """
    x = np.asarray(pips, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {"dd_50": np.nan, "dd_90": np.nan, "dd_99": np.nan}
    rng = np.random.default_rng(seed)
    dds = np.empty(int(n_sim))
    for i in range(int(n_sim)):
        eq = np.cumsum(rng.permutation(x))
        dds[i] = (np.maximum.accumulate(eq) - eq).max()
    p50, p90, p99 = np.percentile(dds, [50, 90, 99])
    return {"dd_50": float(p50), "dd_90": float(p90), "dd_99": float(p99)}


def scheda_strategia(trades: pd.DataFrame, pip_size: float,
                     commission: float = 0.0, n_boot: int = 10_000,
                     montecarlo: bool = True, n_sim: int = 2000,
                     seed: int = 0, min_trades: int = 100,
                     p_max: float = 5.0, verbose: bool = True) -> dict:
    """
    La scheda completa di una strategia: i KPI di scoperta (nostri) e
    quelli di rischio (Lezione 8/9 del corso), su qualunque `.trades()`.

    IL CUSCINETTO SUI COSTI
    ------------------------
    `ev_pips` e' anche la risposta a "quanto attrito in piu' regge questo
    sistema prima di andare a zero". Un EV netto di 0.25 pips su un costo
    round-turn di 0.80 significa che un terzo di giro di costi in piu' di
    slippage lo azzera. E' la versione verificabile di "barely afloat", e
    a differenza di una soglia sul win rate e' portabile su qualunque
    strumento.

    IL WIN RATE DI PAREGGIO
    ------------------------
    `wr_pareggio = 1 / (1 + avg_win/avg_loss)` e' un'identita', non una
    convenzione: il win rate a cui il sistema fa esattamente zero,
    calcolato dai suoi stessi trade. Sta in scheda come DIAGNOSI (e' come
    un trader legge la cosa a colpo d'occhio), non come cancello: win
    rate e reward:risk sono due parametri dello stesso EV, e metterli
    come criterio separato conterebbe due volte lo stesso fatto.

    L'EV IN R MULTIPLI
    -------------------
    Pips diviso la distanza dallo stop di QUEL trade. Qui vale piu' che
    nel corso: lo stop e' adattivo, quindi un trade con stop da 15 pips e
    uno con stop da 40 non sono confrontabili in pips. NaN se lo stop non
    e' attivo (perc_sl=0).
    """
    n_t = int(len(trades)) if trades is not None else 0
    if n_t == 0:
        return {"n_trades": 0, "esito": SCARTATA, "motivo": "nessun trade"}

    lordi, netti = pips_per_trade(trades, pip_size, commission)
    n = netti.to_numpy()
    lato = np.sign(trades["Size"].to_numpy())

    vinc, perd = n[n > 0], n[n <= 0]
    avg_win = float(vinc.mean()) if vinc.size else np.nan
    avg_loss = float(abs(perd.mean())) if perd.size else np.nan
    rr = (avg_win / avg_loss) if (np.isfinite(avg_win) and avg_loss > 0) else np.nan
    wr = float((n > 0).mean() * 100.0)
    wr_par = (100.0 / (1.0 + rr)) if np.isfinite(rr) else np.nan
    pf = (float(vinc.sum() / abs(perd.sum()))
          if perd.size and perd.sum() != 0 else np.nan)

    # EV in R multipli: rischio = distanza dallo stop di quel trade
    ev_r = np.nan
    if "SL" in trades.columns:
        R = np.abs(trades["EntryPrice"].to_numpy() - trades["SL"].to_numpy()) / pip_size
        ok = np.isfinite(R) & (R > 0)
        if ok.any():
            ev_r = float((n[ok] / R[ok]).mean())

    ev = float(n.mean())
    p_neg = p_ev_negativo(n, n_boot=n_boot, seed=seed)
    lo, hi = ic_ev(n, n_boot=n_boot, seed=seed)

    s = {
        "n_trades": n_t,
        "ev_pips": ev,
        "ev_lordo": float(lordi.to_numpy().mean()),
        "ev_R": ev_r,
        "net_pips": float(n.sum()),
        "win_rate": wr,
        "wr_pareggio": wr_par,
        "margine_wr": wr - wr_par if np.isfinite(wr_par) else np.nan,
        "reward_risk": rr,
        "profit_factor": pf,
        "t_stat": t_stat(n),
        "p_ev_neg": p_neg,
        "ic_ev_lo": lo, "ic_ev_hi": hi,
        "dd_chiusi_pips": _drawdown_chiusi(n),
    }
    if montecarlo:
        s.update(drawdown_montecarlo(n, n_sim=n_sim, seed=seed))

    for segno, et in ((1, "long"), (-1, "short")):
        m = lato == segno
        s[f"n_{et}"] = int(m.sum())
        s[f"ev_{et}"] = float(n[m].mean()) if m.sum() else np.nan
        s[f"t_{et}"] = t_stat(n[m]) if m.sum() > 1 else np.nan

    s.update(verdetto(ev=ev, p_neg=p_neg, n_trades=n_t,
                      p_max=p_max, min_trades=min_trades))
    if verbose:
        stampa_scheda(s)
    return s


def stampa_scheda(s: dict, titolo: str = "SCHEDA STRATEGIA") -> None:
    """La scheda in forma leggibile. Le soglie citate sono della Lezione 9."""
    if s.get("n_trades", 0) == 0:
        print(f"{titolo}: nessun trade.")
        return
    r = "=" * 66
    print(f"\n{r}\n  {titolo}  —  {s['n_trades']} trade\n{r}")
    print(f"  EV netto per trade      {s['ev_pips']:>9.3f} pips   "
          f"(lordo {s['ev_lordo']:.3f})")
    print(f"     cuscinetto sui costi: regge {s['ev_pips']:.3f} pips di attrito "
          "in piu' prima di azzerarsi")
    if np.isfinite(s.get("ev_R", np.nan)):
        print(f"  EV in R multipli        {s['ev_R']:>9.3f} R")
    print(f"  t-stat                  {s['t_stat']:>9.2f}")
    print(f"  P(EV<0)                 {s['p_ev_neg']:>9.1f} %      "
          "soglie: <5% ok, <1% molto buono")
    print(f"  IC 95% dell'EV          [{s['ic_ev_lo']:.3f} , {s['ic_ev_hi']:.3f}] pips")
    print(f"  win rate                {s['win_rate']:>9.1f} %      "
          f"pareggio a {s['wr_pareggio']:.1f}%  "
          f"(margine {s['margine_wr']:+.1f})")
    print(f"  reward:risk realizzato  {s['reward_risk']:>9.2f}        "
          f"profit factor {s['profit_factor']:.3f}")
    print(f"  max DD a trade chiusi   {s['dd_chiusi_pips']:>9.1f} pips")
    if "dd_50" in s:
        print(f"     Monte Carlo shuffle: mediano {s['dd_50']:.0f}, "
              f"90° pct {s['dd_90']:.0f}, 99° pct {s['dd_99']:.0f} pips")
    for et in ("long", "short"):
        if s.get(f"n_{et}", 0) > 1:
            print(f"     solo {et:<6} {s[f'n_{et}']:>5} trade   "
                  f"EV {s[f'ev_{et}']:+.3f} pips   t {s[f't_{et}']:+.2f}")
    print(f"  ESITO                   {s['esito']}"
          + (f"   ({s['motivo']})" if s.get("motivo") else ""))
    print(r)
