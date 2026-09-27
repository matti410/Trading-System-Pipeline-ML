"""
Operatori dal paper "101 Formulaic Alphas" (Kakushadze, WorldQuant, 2015 -
arXiv:1601.00991), usati da `filter_conditions.py`.

Il file originale con lo stesso nome (percorso V3) non e' piu' raggiungibile
per portarlo pari pari: qui ci sono SOLO i tre operatori che
`filter_conditions.py` usa davvero (`delta`, `ts_rank`, `ts_sum`), ricostruiti
dalle definizioni standard del paper e incrociati con l'uso che ne fanno i
docstring dei filtri (es. F10: "il rendimento cumulato sulle ultime lookback
barre" -> ts_sum; F14: "il drift assoluto della media a `window` barre" ->
delta). Non e' l'intera libreria di operatori Alpha101 (rank cross-sezionale,
delay, correlation, ecc.): quelli non servono a nessun filtro qui e non sono
stati aggiunti.

`ts_rank` e' la stessa formula di `helpers._ts_rank` (usata da entry_long.py/
entry_short.py) — tenute due copie indipendenti invece di farle condividere
un import: `helpers.py` e' scope "supporto agli entry", questo file e' scope
"operatori Alpha101 per i filtri". Stessa formula, stesso motivo per cui
`soglie_adattive` vive in due file diversi (vedi `exit_search_bt.py`).
"""
from __future__ import annotations

import pandas as pd


def delta(series: pd.Series, period: int) -> pd.Series:
    """x - x di `period` barre fa. Alpha101: delta(x, d)."""
    return series - series.shift(int(period))


def ts_sum(series: pd.Series, window: int) -> pd.Series:
    """Somma mobile sulle ultime `window` barre. Alpha101: ts_sum(x, d)."""
    return series.rolling(int(window)).sum()


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Percentile della barra corrente rispetto alle ultime `window` barre
    della stessa serie (rolling, mai cross-sezionale). Valori in (0, 1].
    Alpha101: ts_rank(x, d), qui SEMPRE nella variante a singolo asset.
    """
    return series.rolling(int(window)).rank(pct=True)
