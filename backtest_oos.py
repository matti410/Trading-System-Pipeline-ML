"""
Backtest OOS dei trigger selezionati dal modello, sul motore veloce di V4 (D5, 30/10/2026).

  trigger_in_fascia(modello, X, idx_test, idx_oos, fasce=5, n_fasce=5)
      -> trigger OOS con probabilita' nella fascia operativa (soglia dal test)
  trade_motore(df, segnali, direction, horizon, costi)
      -> DataFrame dei trade del motore, con lordo_bp, netto_bp e la barra del trigger
  esegui_backtest(df, segnali_long, segnali_short, H_long, H_short, costi)
      -> dict con i trade LONG, SHORT, COMBINATO e i segnali saltati
  tabella_metriche({"nome": risultato, ...}) -> DataFrame
  grafico_equity({"nome": risultato, ...})

REGOLE: le stesse del Lab e del target
  - MotoreVeloce di V4 (engine/motore_veloce.py), exclusive_orders=True: un
    trade alla volta PER LATO. Un segnale che arriva con un trade dello stesso
    lato aperto si salta (e si conta);
  - uscita a tempo n_barre = H del lato: ingresso all'Open dopo il trigger,
    uscita all'Open della barra trigger + H + 1, niente SL/TP;
  - costi con la stessa catena del Lab: costo_extra_pips -> aggiungi_funding ->
    aggiungi_spread_passo -> netti_in_unita(..., "bp").
Il motore gira sullo storico intero (cosi' funding, quarantena e posizioni
delle barre sono quelle del target) con i segnali accesi SOLO sulle barre
passate: nessun trade fuori dal periodo dei segnali.

UNITA': bp per trade, a capitale fisso (nessun reinvestimento). La curva e' la
somma dei bp netti dei trade, in ordine di uscita.

COMBINATO (LONG + SHORT): i due lati girano separati, ciascuno con il suo
capitale, e possono essere aperti insieme (esposizione doppia, nessuna
compensazione). Il COMBINATO e' l'unione dei trade dei due lati, ordinata per
ora d'uscita: trade, vincenti, trade medio e profit factor sui trade uniti;
curva = somma cumulata dei bp di tutti i trade; max drawdown su quella curva;
esposizione = quota di barre con almeno un lato aperto.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oos import soglia_dal_test

COLORE_MODELLO = "#2a78d6"
COLORE_TUTTI = "#8a8a85"


def trigger_in_fascia(modello, X, idx_test, idx_oos, fasce=5, n_fasce=5):
    """Stessa soglia di oos.valuta_oos: quantili delle probabilita' del test."""
    bassa, alta = soglia_dal_test(modello, X, idx_test, fasce, n_fasce)
    p_oo = modello.predict_proba(X.loc[idx_oos])[:, 1]
    return idx_oos[(p_oo >= bassa) & (p_oo < alta)]


def trade_motore(df, segnali, direction, horizon, costi, quarantena=True):
    """
    I trade del motore veloce di V4 per un lato, con i costi del Lab. `segnali`
    sono le barre dei trigger da tradare (gia' pulite dalla quarantena: vengono
    dal target). Colonne in piu': trigger (la barra del segnale), lordo_bp, netto_bp.
    """
    from engine.exchange import aggiungi_funding, aggiungi_spread_passo
    from engine.metriche import netti_in_unita
    from engine.motore_veloce import MotoreVeloce
    from engine.quarantena_trade import costo_extra_pips

    if direction not in (1, -1):
        raise ValueError("direction deve essere 1 (LONG) o -1 (SHORT)")
    df_bt = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    acceso = df.index.isin(pd.DatetimeIndex(segnali))
    if acceso.sum() != len(pd.DatetimeIndex(segnali).unique()):
        raise ValueError("segnali con barre che non sono nel df")
    df_bt["segnale"] = acceso
    for col in ("sl_long", "tp_long", "sl_short", "tp_short"):
        df_bt[col] = np.nan
    lato = {"long_col": "segnale"} if direction == 1 else {"short_col": "segnale"}
    stats = MotoreVeloce(df_bt, cash=costi.get("cash", 10_000_000), spread=costi["spread"],
                         commission=costi["commission"], exclusive_orders=True).run(n_barre=int(horizon), **lato)
    pip = float(costi["pip_size"])
    tr = costo_extra_pips(stats["_trades"], df, quarantena, costi["spread"], costi.get("spread_rollover_pips"), pip)
    tr = aggiungi_funding(tr, df, costi.get("funding"), pip)
    tr = aggiungi_spread_passo(tr, costi.get("spread_passo_pips"))
    lordi, netti = netti_in_unita(tr, pip, float(costi["commission"]), "bp")
    tr = tr.assign(trigger=df.index[tr["EntryBar"].to_numpy(dtype=int) - 1], lordo_bp=lordi, netto_bp=netti,
                   lato="LONG" if direction == 1 else "SHORT")
    return tr


def esegui_backtest(df, segnali_long, segnali_short, H_long, H_short, costi):
    """
    LONG e SHORT separati (un trade alla volta per lato) e il COMBINATO (vedi
    sopra). Un lato senza segnali (None, es. lo short sullo spot) resta vuoto.
    """
    out = {"trade": {}, "saltati": {}, "barre": len(df)}
    for nome, segnali, direction, H in (("LONG", segnali_long, 1, H_long), ("SHORT", segnali_short, -1, H_short)):
        if segnali is None or len(segnali) == 0:
            out["trade"][nome] = pd.DataFrame(columns=["EntryBar", "ExitBar", "ExitTime", "netto_bp", "lordo_bp"])
            out["saltati"][nome] = 0
            continue
        tr = trade_motore(df, segnali, direction, H, costi)
        out["trade"][nome] = tr
        out["saltati"][nome] = int(len(pd.DatetimeIndex(segnali).unique()) - len(tr))
    lati = [t for t in out["trade"].values() if len(t)]
    out["trade"]["COMBINATO"] = (pd.concat(lati).sort_values(["ExitTime", "EntryBar"]).reset_index(drop=True)
                                 if lati else out["trade"]["LONG"])
    out["saltati"]["COMBINATO"] = out["saltati"]["LONG"] + out["saltati"]["SHORT"]
    return out


def _esposizione(trades, inizio, fine):
    """Quota delle barre [inizio, fine) con almeno un trade aperto (ingresso compreso, uscita esclusa)."""
    if fine <= inizio or len(trades) == 0:
        return 0.0
    aperto = np.zeros(fine - inizio + 1, dtype=int)
    eb = np.clip(trades["EntryBar"].to_numpy(dtype=int) - inizio, 0, fine - inizio)
    xb = np.clip(trades["ExitBar"].to_numpy(dtype=int) - inizio, 0, fine - inizio)
    np.add.at(aperto, eb, 1)
    np.add.at(aperto, xb, -1)
    return float((np.cumsum(aperto)[:-1] > 0).mean())


def _metriche(trades, saltati, inizio, fine):
    bp = trades["netto_bp"].astype(float).to_numpy() if len(trades) else np.array([])
    curva = np.cumsum(bp)
    picchi = np.maximum.accumulate(np.r_[0.0, curva])[1:] if len(bp) else np.array([])
    perdite = -bp[bp < 0].sum()
    return {
        "trade": len(bp),
        "saltati": saltati,
        "vincenti_%": 100 * (bp > 0).mean() if len(bp) else np.nan,
        "trade_medio_bp": bp.mean() if len(bp) else np.nan,
        "profit_factor": bp[bp > 0].sum() / perdite if perdite > 0 else np.inf,
        "totale_bp": curva[-1] if len(bp) else 0.0,
        "max_dd_bp": float((picchi - curva).max()) if len(bp) else 0.0,
        "esposizione_%": 100 * _esposizione(trades, inizio, fine),
    }


def tabella_metriche(risultati, inizio_bar, fine_bar):
    """
    Una riga per lato e scenario. inizio_bar / fine_bar: posizioni nel df del
    periodo (per l'esposizione), es. la prima barra OOS e len(df).
    """
    righe = {}
    for nome, r in risultati.items():
        for lato in ("LONG", "SHORT", "COMBINATO"):
            righe[(lato, nome)] = _metriche(r["trade"][lato], r["saltati"][lato], inizio_bar, fine_bar)
    tab = pd.DataFrame(righe).T
    tab.index.names = ["lato", "scenario"]
    return tab.astype(float).round(2)


def grafico_equity(risultati):
    colori = [COLORE_MODELLO, COLORE_TUTTI]
    fig, assi = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, lato in zip(assi, ("LONG", "SHORT", "COMBINATO")):
        for (nome, r), colore in zip(risultati.items(), colori):
            t = r["trade"][lato]
            if len(t):
                ax.plot(pd.DatetimeIndex(t["ExitTime"]), t["netto_bp"].astype(float).cumsum(), color=colore,
                        lw=2 if colore == COLORE_MODELLO else 1.5, label=nome)
        ax.axhline(0, color="#b5b5b0", lw=1, ls=":")
        ax.set_title(lato, loc="left", fontsize=11)
        ax.set_ylabel("bp netti cumulati")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", frameon=False)
    fig.suptitle("Equity OOS — motore veloce di V4, costi del Lab, bp netti a capitale fisso", x=0.01, ha="left")
    fig.tight_layout()
    plt.show()
    return fig
