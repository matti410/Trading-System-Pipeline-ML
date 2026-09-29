"""
Applicazione del modello all'OOS con soglia fissata sul test.

Modulo ADDITIVO (29/9/2026): non modifica nessun file esistente.

  indici_oos(target, df_is)                       -> DatetimeIndex
  valuta_oos(modello, X, y, rend, idx_test, idx_oos, fasce=5, n_fasce=5)
                                                  -> DataFrame di confronto

REGOLA (decisioni del 29/9)
---------------------------
- Il modello e' quello addestrato sul SOLO train (opzione a), cosi' la
  scala delle probabilita' e' la stessa da cui si ricava la soglia.
- La soglia si ricava SOLO dalle probabilita' del test di df_is: i confini
  delle fasce sono i quantili delle probabilita' del test (fascia 5 su 5 =
  il 20% piu' alto). Poi si applica TALE E QUALE all'OOS: nessun numero
  viene ricalcolato sull'OOS.
- `fasce` puo' essere una fascia (5) o un intervallo ((4, 5)). La fascia
  piu' bassa selezionata non ha limite inferiore se e' la 1, la piu' alta
  non ha limite superiore se e' l'ultima: un trigger OOS con probabilita'
  oltre il massimo visto nel test viene comunque preso.
"""
import numpy as np
import pandas as pd


def indici_oos(target, df_is):
    """Trigger del target successivi all'ultima barra di df_is."""
    return target.index[target.index > df_is.index[-1]]


def _intervallo(fasce, n_fasce):
    if isinstance(fasce, (int, np.integer)):
        fasce = (fasce, fasce)
    da, a = int(fasce[0]), int(fasce[1])
    if not (1 <= da <= a <= n_fasce):
        raise ValueError(f"fasce deve stare fra 1 e {n_fasce}, es. 5 oppure (4, 5)")
    return da, a


def _riga(nome, p, y, bp, presi, totale):
    return {"": nome, "trade": int(presi.sum()),
            "presi_%": 100 * presi.sum() / totale,
            "vincenti_%": 100 * y[presi].mean() if presi.any() else np.nan,
            "bp_medi_lordi": bp[presi].mean() if presi.any() else np.nan}


def valuta_oos(modello, X, y, rend, idx_test, idx_oos, fasce=5, n_fasce=5):
    """
    Ricava la soglia dalle fasce del test e la applica all'OOS.
    Stampa soglia e distribuzione delle probabilita' test vs OOS,
    restituisce la tabella: test (selezionati / tutti), OOS (selezionati / tutti).
    """
    if len(idx_oos) == 0:
        raise ValueError("idx_oos vuoto")
    if idx_oos.min() <= idx_test.max():
        raise ValueError("l'OOS deve venire tutto dopo il test")
    da, a = _intervallo(fasce, n_fasce)

    p_te = modello.predict_proba(X.loc[idx_test])[:, 1]
    p_oo = modello.predict_proba(X.loc[idx_oos])[:, 1]

    confini = np.quantile(p_te, np.linspace(0, 1, n_fasce + 1))
    soglia_bassa = -np.inf if da == 1 else confini[da - 1]
    soglia_alta = np.inf if a == n_fasce else confini[a]

    def seleziona(p):
        return (p >= soglia_bassa) & (p < soglia_alta)

    etichetta = f"fascia {da}" if da == a else f"fasce {da}-{a}"
    print(f"{etichetta} di {n_fasce}: soglia prob. da {soglia_bassa:.3f} a {soglia_alta:.3f}"
          f"  (ricavata solo dal test)")
    q = [0, 0.10, 0.50, 0.90, 1.0]
    for nome, p in (("test", p_te), ("OOS ", p_oo)):
        print(f"prob. {nome}: " + "  ".join(
            f"{n} {v:.3f}" for n, v in zip(["min", "p10", "p50", "p90", "max"], np.quantile(p, q))))
    print(f"OOS: {idx_oos[0]} -> {idx_oos[-1]}")

    righe = []
    for nome, idx, p in (("test", idx_test, p_te), ("OOS", idx_oos, p_oo)):
        yy = y.loc[idx].to_numpy()
        bp = rend.loc[idx].to_numpy() * 1e4
        tutti = np.ones(len(p), dtype=bool)
        righe.append(_riga(f"{nome} {etichetta}", p, yy, bp, seleziona(p), len(p)))
        righe.append(_riga(f"{nome} tutti", p, yy, bp, tutti, len(p)))
    return pd.DataFrame(righe).set_index("").round(
        {"presi_%": 1, "vincenti_%": 1, "bp_medi_lordi": 2})
