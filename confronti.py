"""
I due confronti fissi sul test (D3, decisione di Mattia del 30/10/2026): solo stampati,
mai usati per scegliere.

  selezione_test(modello, X, idx_test, fasce=5, n_fasce=5) -> DatetimeIndex
  miglior_filtro_train(target, df, idx_train, filtri, min_trade=100) -> (nome, tabella)
  confronti_test(target, df, idx_test, selezionati, filtro, quota=0.2,
                 n_estrazioni=1000, seme=0) -> DataFrame

  1. «trigger + miglior filtro»: fra i filtri del Lab si sceglie, SOLO sul train,
     quello con i bp netti medi piu' alti fra i trigger in cui e' vero, con almeno
     `min_trade` trigger. Poi lo si misura sul test, senza ritoccarlo.
  2. «20% a caso»: 1.000 estrazioni (seme fisso) del 20% dei trigger del test.
     Si stampa la loro distribuzione e in che percentile cade il modello.
"""
import numpy as np
import pandas as pd

from oos import soglia_dal_test


def selezione_test(modello, X, idx_test, fasce=5, n_fasce=5):
    """I trigger del test nella fascia operativa (stessa soglia di oos e del backtest)."""
    bassa, alta = soglia_dal_test(modello, X, idx_test, fasce, n_fasce)
    p = modello.predict_proba(X.loc[idx_test])[:, 1]
    return idx_test[(p >= bassa) & (p < alta)]


def miglior_filtro_train(target, df, idx_train, filtri, min_trade=100):
    """
    Il filtro con i bp netti medi piu' alti sul TRAIN (almeno min_trade trigger
    in cui e' vero). Ritorna (nome o None, tabella di tutti i filtri ammessi).
    """
    netto = target.loc[idx_train, "netto_bp"]
    righe = []
    for f in filtri:
        vero = df.loc[idx_train, f].astype(bool).to_numpy()
        if vero.sum() >= min_trade:
            righe.append({"filtro": f, "trade_train": int(vero.sum()), "bp_medi_netti_train": netto[vero].mean()})
    tab = pd.DataFrame(righe, columns=["filtro", "trade_train", "bp_medi_netti_train"])
    tab = tab.sort_values("bp_medi_netti_train", ascending=False).reset_index(drop=True)
    return (tab["filtro"].iloc[0] if len(tab) else None), tab


def _riga(nome, netto, quanti, totale):
    return {"": nome, "trade": int(quanti), "presi_%": 100 * quanti / totale,
            "vincenti_%": 100 * (netto > 0).mean() if quanti else np.nan,
            "bp_medi_netti": netto.mean() if quanti else np.nan}


def confronti_test(target, df, idx_test, selezionati, filtro, quota=0.2, n_estrazioni=1000, seme=0):
    """
    La tabella dei confronti sul test: modello, trigger + miglior filtro (dal
    train), tutti i trigger, 20% a caso (media e percentili 5-95 delle
    estrazioni) e il percentile del modello fra le estrazioni.
    """
    netto = target.loc[idx_test, "netto_bp"].to_numpy(dtype=float)
    n = len(netto)
    righe = []
    sel = target.loc[selezionati, "netto_bp"].to_numpy(dtype=float)
    righe.append(_riga("modello (fascia operativa)", sel, len(sel), n))
    if filtro is not None:
        vero = df.loc[idx_test, filtro].astype(bool).to_numpy()
        righe.append(_riga(f"trigger + {filtro} (scelto sul train)", netto[vero], int(vero.sum()), n))
    righe.append(_riga("tutti i trigger del test", netto, n, n))
    k = max(1, round(quota * n))
    rng = np.random.default_rng(seme)
    medie = np.array([netto[rng.choice(n, size=k, replace=False)].mean() for _ in range(n_estrazioni)])
    righe.append({"": f"{quota:.0%} a caso: media di {n_estrazioni} estrazioni", "trade": k,
                  "presi_%": 100 * k / n, "vincenti_%": np.nan, "bp_medi_netti": medie.mean()})
    righe.append({"": f"{quota:.0%} a caso: 5° percentile", "trade": k, "presi_%": 100 * k / n,
                  "vincenti_%": np.nan, "bp_medi_netti": np.quantile(medie, 0.05)})
    righe.append({"": f"{quota:.0%} a caso: 95° percentile", "trade": k, "presi_%": 100 * k / n,
                  "vincenti_%": np.nan, "bp_medi_netti": np.quantile(medie, 0.95)})
    tab = pd.DataFrame(righe).set_index("").round({"presi_%": 1, "vincenti_%": 1, "bp_medi_netti": 2})
    percentile = 100 * (medie < sel.mean()).mean() if len(sel) else np.nan
    return tab, percentile
