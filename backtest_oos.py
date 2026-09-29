"""
Backtest (VectorBT) dei trigger selezionati dal modello nel periodo OOS.

Modulo ADDITIVO (29/9/2026): non modifica nessun file esistente.

  trigger_in_fascia(modello, X, idx_test, idx_oos, fasce=5, n_fasce=5)
      -> trigger OOS con probabilita' nella fascia operativa (soglia dal test)
  un_trade_alla_volta(idx, df_periodo, horizon=25)
      -> (trigger tenuti, n. saltati)
  esegui_backtest(df_periodo, trigger_long, trigger_short, horizon=25,
                  unita=100_000, capitale=150_000, costo_rt=0.65e-4)
      -> dict con i portafogli LONG, SHORT, COMBINATO e i trigger saltati
  tabella_metriche({"nome": risultato, ...}) -> DataFrame
  grafico_equity({"nome": risultato, ...}, capitale)

REGOLE (decisioni del 29/9) - le stesse del target:
  - ingresso all'Open della barra successiva al trigger
  - uscita al Close a `horizon` barre dal trigger, niente SL/TP
  - un trade alla volta PER LATO: un trigger che scatta mentre un trade
    dello stesso lato e' aperto viene saltato (e contato)
  - lotto fisso (`unita` del symbol base), nessun reinvestimento:
    il money management e' uno step successivo
  - costo: `costo_rt` in frazione del controvalore, meta' in ingresso e
    meta' in uscita (0.65 bp = commissione IC Markets EU Raw Spread)

VectorBT open-source NON gestisce il margine: `capitale` deve coprire il
controvalore della posizione. Se non basta, il backtest si ferma con un
errore invece di ridurre la size in silenzio.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

from oos import _intervallo

COLORE_MODELLO = "#2a78d6"
COLORE_TUTTI = "#8a8a85"


def trigger_in_fascia(modello, X, idx_test, idx_oos, fasce=5, n_fasce=5):
    """Stessa soglia di oos.valuta_oos: quantili delle probabilita' del test."""
    da, a = _intervallo(fasce, n_fasce)
    p_te = modello.predict_proba(X.loc[idx_test])[:, 1]
    p_oo = modello.predict_proba(X.loc[idx_oos])[:, 1]
    confini = np.quantile(p_te, np.linspace(0, 1, n_fasce + 1))
    bassa = -np.inf if da == 1 else confini[da - 1]
    alta = np.inf if a == n_fasce else confini[a]
    return idx_oos[(p_oo >= bassa) & (p_oo < alta)]


def un_trade_alla_volta(idx, df_periodo, horizon=25):
    """
    Tiene un trigger solo se il trade precedente dello stesso lato e' gia'
    chiuso: nuovo ingresso (pos+1) dopo la barra d'uscita precedente
    (pos_prec + horizon).
    """
    pos = df_periodo.index.get_indexer(idx)
    if (pos < 0).any() or (pos + horizon >= len(df_periodo)).any():
        raise ValueError("trigger fuori dal periodo o senza horizon barre in avanti")
    tenuti, ultima_uscita = [], -1
    for p in np.sort(pos):
        if p + 1 > ultima_uscita:
            tenuti.append(p)
            ultima_uscita = p + horizon
    return df_periodo.index[tenuti], len(pos) - len(tenuti)


def _ordini(df_periodo, trigger, direction, horizon, unita):
    n = len(df_periodo)
    size = np.full(n, np.nan)
    prezzo = np.full(n, np.nan)
    pos = df_periodo.index.get_indexer(trigger)
    size[pos + 1] = unita * direction
    prezzo[pos + 1] = df_periodo["Open"].to_numpy()[pos + 1]
    size[pos + horizon] = -unita * direction
    prezzo[pos + horizon] = df_periodo["Close"].to_numpy()[pos + horizon]
    return size, prezzo


def _portafoglio(df_periodo, size, prezzo, capitale, costo_rt, raggruppa=False):
    return vbt.Portfolio.from_orders(
        close=df_periodo["Close"] if size.ndim == 1 else
        pd.concat([df_periodo["Close"]] * size.shape[1], axis=1, keys=["LONG", "SHORT"]),
        size=size, price=prezzo, size_type="amount", direction="both",
        fees=costo_rt / 2, init_cash=capitale,
        cash_sharing=raggruppa, group_by=True if raggruppa else None,
        allow_partial=False, raise_reject=True, freq="15min",
    )


def esegui_backtest(df_periodo, trigger_long, trigger_short, horizon=25,
                    unita=100_000, capitale=150_000, costo_rt=0.65e-4):
    tl, saltati_l = un_trade_alla_volta(trigger_long, df_periodo, horizon)
    ts, saltati_s = un_trade_alla_volta(trigger_short, df_periodo, horizon)
    sl, pl = _ordini(df_periodo, tl, 1, horizon, unita)
    ss, ps = _ordini(df_periodo, ts, -1, horizon, unita)
    return {
        "LONG": _portafoglio(df_periodo, sl, pl, capitale, costo_rt),
        "SHORT": _portafoglio(df_periodo, ss, ps, capitale, costo_rt),
        "COMBINATO": _portafoglio(df_periodo, np.column_stack([sl, ss]),
                                  np.column_stack([pl, ps]), capitale, costo_rt,
                                  raggruppa=True),
        "saltati": {"LONG": saltati_l, "SHORT": saltati_s,
                    "COMBINATO": saltati_l + saltati_s},
        "trigger": {"LONG": tl, "SHORT": ts},
    }


def _metriche(pf, saltati):
    t = pf.trades.records_readable
    pnl, ret = t["PnL"], t["Return"]
    perdite = -pnl[pnl < 0].sum()
    in_pos = (pf.asset_value(group_by=False) != 0)
    in_pos = in_pos.any(axis=1) if in_pos.ndim == 2 else in_pos
    return {
        "trade": len(t),
        "saltati": saltati,
        "vincenti_%": 100 * (pnl > 0).mean(),
        "trade_medio_bp": 1e4 * ret.mean(),
        "profit_factor": pnl[pnl > 0].sum() / perdite if perdite > 0 else np.inf,
        "pnl_totale": pnl.sum(),
        "rendimento_%": 100 * pf.total_return(),
        "max_dd_%": 100 * pf.max_drawdown(),
        "sharpe": pf.sharpe_ratio(),
        "esposizione_%": 100 * in_pos.mean(),
    }


def tabella_metriche(risultati):
    righe = {}
    for nome, r in risultati.items():
        for lato in ("LONG", "SHORT", "COMBINATO"):
            righe[(lato, nome)] = _metriche(r[lato], r["saltati"][lato])
    tab = pd.DataFrame(righe).T
    tab.index.names = ["lato", "scenario"]
    return tab.astype(float).round(2)


def grafico_equity(risultati, capitale):
    colori = [COLORE_MODELLO, COLORE_TUTTI]
    fig, assi = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, lato in zip(assi, ("LONG", "SHORT", "COMBINATO")):
        for (nome, r), colore in zip(risultati.items(), colori):
            ax.plot(r[lato].value(), color=colore, lw=2 if colore == COLORE_MODELLO else 1.5,
                    label=nome)
        ax.axhline(capitale, color="#b5b5b0", lw=1, ls=":")
        ax.set_title(lato, loc="left", fontsize=11)
        ax.set_ylabel("equity (valuta di quotazione)")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="upper left", frameon=False)
    fig.suptitle("Equity OOS — lotto fisso, commissioni incluse, PnL nella valuta di quotazione (USD su EURUSD)", x=0.01, ha="left")
    fig.tight_layout()
    plt.show()
    return fig
