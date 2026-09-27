"""
Operatori VWAP condivisi — adattati dal file `vwap_indicator_features.py`
fornito dall'utente. Vivono in engine/ (non in conditions/) per lo stesso
motivo di alpha_ops.py: sono operatori generici sul prezzo, riutilizzabili
da più file di condizioni, non logica di una singola strategia.

`vwap_anchored_daily` calcola la colonna base `vwap`, richiesta da diverse
formule alpha101 (#5, #41, #42, #57, #61, #62 ecc.) — va eseguita UNA VOLTA
nella cella "1.1 Features Engineering" del notebook, non dentro le
condizioni (altrimenti verrebbe ricalcolata ad ogni chiamata).

`regression_trend_fit` e `vwap_gap` sono invece usate solo dai filtri di
regime in conditions/vwap_regime_filter_conditions.py (price_over_under,
gap_trend, last_close_position) — non fanno parte delle 101 formule del
paper, sono un'estensione proposta dall'utente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vwap_anchored_daily(df: pd.DataFrame) -> pd.Series:
    """
    VWAP intraday ancorato al giorno (si azzera e riparte ad ogni nuova
    sessione), calcolato come cumulata di prezzo tipico * volume diviso
    cumulata di volume, entrambe raggruppate per data.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_x_vol = typical_price * df["Volume"]

    date_groups = df.index.date
    cumulative_tp_vol = tp_x_vol.groupby(date_groups).cumsum()
    cumulative_vol = df["Volume"].groupby(date_groups).cumsum()

    return cumulative_tp_vol / cumulative_vol


def regression_trend_fit(close: pd.Series) -> pd.Series:
    """
    Regressione lineare incrementale (O(n)) del prezzo di chiusura,
    ricalcolata da zero ad ogni nuova sessione (stessa logica di
    `regression_vwap_gap` nel file originale, isolata dal calcolo del gap
    vero e proprio).
    """

    def _cumulative_regression_fitted_fast(close_session: pd.Series) -> pd.Series:
        y = close_session.values
        n = len(y)
        fitted = np.empty(n)

        sum_y = 0.0
        sum_xy = 0.0

        for i in range(n):
            sum_y += y[i]
            sum_xy += i * y[i]

            if i == 0:
                fitted[i] = y[i]
            else:
                n_points = i + 1
                sum_x = i * (i + 1) / 2
                sum_x2 = i * (i + 1) * (2 * i + 1) / 6

                denom = n_points * sum_x2 - sum_x * sum_x
                slope = (n_points * sum_xy - sum_x * sum_y) / denom
                intercept = (sum_y - slope * sum_x) / n_points

                fitted[i] = intercept + slope * i

        return pd.Series(fitted, index=close_session.index)

    date_groups = close.index.date
    fitted = close.groupby(date_groups).apply(_cumulative_regression_fitted_fast)
    if isinstance(fitted.index, pd.MultiIndex):
        fitted = fitted.droplevel(0)
    return fitted


def vwap_gap(df: pd.DataFrame, vwap: pd.Series) -> pd.Series:
    """Scarto tra il fair value stimato via regressione intraday e il vwap."""
    fitted = regression_trend_fit(df["Close"])
    return fitted - vwap
