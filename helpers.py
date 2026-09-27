"""
Funzioni di supporto condivise da entry_long.py ed entry_short.py.

Nessuna di queste calcola un trigger: sono mattoncini usati dalle
funzioni nei due file di ingresso.
"""
import pandas as pd


def _evento(mask: pd.Series) -> pd.Series:
    """
    Da STATO a EVENTO: tiene solo la prima barra di ogni blocco
    consecutivo di True. Esempio: [F,T,T,T,F,T] diventa [F,T,F,F,F,T].
    """
    mask = mask.fillna(False).astype(bool)
    return mask & ~mask.shift(1, fill_value=False)


def _cross_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """`fast` attraversa `slow` dal basso verso l'alto (evento puntuale)."""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def _cross_down(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """`fast` attraversa `slow` dall'alto verso il basso (evento puntuale)."""
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Percentile della barra corrente rispetto alle ultime `window` barre
    della stessa serie (rolling, mai cross-sezionale). Valori in (0, 1].
    """
    return series.rolling(int(window)).rank(pct=True)


def _cdl_bull(df: pd.DataFrame, talib_func) -> pd.Series:
    """
    Lato rialzista di una funzione di pattern recognition TA-Lib (CDL*):
    qualunque valore positivo. Evento a singola barra.
    """
    raw = talib_func(
        df["Open"].values.astype(float),
        df["High"].values.astype(float),
        df["Low"].values.astype(float),
        df["Close"].values.astype(float),
    )
    return _evento(pd.Series(raw, index=df.index) > 0)


def _cdl_bear(df: pd.DataFrame, talib_func) -> pd.Series:
    """Lato ribassista della stessa funzione TA-Lib: qualunque valore negativo."""
    raw = talib_func(
        df["Open"].values.astype(float),
        df["High"].values.astype(float),
        df["Low"].values.astype(float),
        df["Close"].values.astype(float),
    )
    return _evento(pd.Series(raw, index=df.index) < 0)


def _confermato(pattern_evento: pd.Series, conferma: pd.Series) -> pd.Series:
    """
    Sposta un evento (vero a T) alla barra T+1 e lo AND-a con una
    condizione valutata sui dati della barra corrente (T+1 stesso, zero
    look-ahead): il pattern avviene a T, la conferma arriva a T+1, il
    segnale finale scatta a T+1.
    """
    pattern_shifted = pattern_evento.shift(1, fill_value=False).astype(bool)
    return pattern_shifted & conferma.fillna(False).astype(bool)


def _conferma_rialzista(df: pd.DataFrame) -> pd.Series:
    """Barra di conferma generica: chiusura sopra l'apertura alla barra corrente."""
    return df["Close"] > df["Open"]


def _conferma_ribassista(df: pd.DataFrame) -> pd.Series:
    """Barra di conferma generica: chiusura sotto l'apertura alla barra corrente."""
    return df["Close"] < df["Open"]
