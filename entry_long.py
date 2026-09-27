"""
Trigger di ingresso LONG.

Ogni funzione `entry_*` prende il DataFrame di mercato (OHLCV +
indicatori: rsi, ema20, ema50, zlema50) e restituisce una Series
booleana della stessa lunghezza: True dove il trigger scatta.

Tutte le funzioni restituiscono EVENTI (una sola barra True per
occorrenza), non stati.

Due funzioni d'uso, in fondo al file:

  aggiungi_trigger_long(df)   calcola tutti i trigger e li aggiunge come
                               colonne al df (usalo quando ti servono le
                               colonne, es. per ispezionarle a mano).

  registra_trigger_long()     registra tutte le funzioni nel motore
                               (engine.registry), cosi' run_event_study
                               le trova da solo. Va chiamata prima di
                               run_event_study.

Per aggiungere un nuovo trigger:
1. scrivere una nuova funzione `entry_*(df) -> pd.Series`
2. aggiungerla al dizionario `TRIGGER_LONG` qui sotto
"""
import pandas as pd
import talib

from helpers import (
    _evento,
    _cross_up,
    _cross_down,
    _ts_rank,
    _cdl_bull,
    _confermato,
    _conferma_rialzista,
)
#from engine.livelli import primo_del_giorno, quarantena_cached, range_finestra
#from engine.sessioni import in_sessione


# ------------------------------------------------------------- E1 --------
def entry_rsi_cross_oversold(df: pd.DataFrame, threshold: float = 30.0) -> pd.Series:
    """L'RSI attraversa la soglia di ipervenduto verso il basso -> long."""
    return _evento(_cross_down(df["rsi"], pd.Series(threshold, index=df.index)))


# ------------------------------------------------------------- E2 --------
def entry_zlema_cross_up(df: pd.DataFrame) -> pd.Series:
    """Il prezzo attraversa la ZLEMA dal basso verso l'alto -> long."""
    return _evento(_cross_up(df["Close"], df["zlema50"]))


# ------------------------------------------------------------- E3 --------
def entry_long_intrabar_weakness_fade(
    df: pd.DataFrame,
    pos_window: float = 10,
    rank_window: float = 20,
    threshold: float = 0.15,
) -> pd.Series:
    """
    Close nella parte bassa del range recente E barra chiusa debole
    rispetto all'apertura -> fade rialzista.
    """
    posizione = _ts_rank(df["Close"], pos_window)
    forza = _ts_rank(df["Close"] / df["Open"], rank_window)
    return _evento((posizione < threshold) & (forza < threshold))


# ------------------------------------------------------------- E4 --------
def entry_ema_cross_up(df: pd.DataFrame) -> pd.Series:
    """EMA20 attraversa EMA50 dal basso verso l'alto -> long."""
    return _evento(_cross_up(df["ema20"], df["ema50"]))


# ------------------------------------------------------------- E5 --------
def entry_quick_pullback(df: pd.DataFrame) -> pd.Series:
    """
    High di 2 barre fa > High di 1 barra fa (pullback nel massimo) E
    Low di 2 barre fa < Low di 1 barra fa (pullback nel minimo) E
    chiusura corrente sopra l'High di 2 barre fa -> rottura rialzista.
    """
    high2, high1 = df["High"].shift(2), df["High"].shift(1)
    low2, low1 = df["Low"].shift(2), df["Low"].shift(1)
    cond = (high2 > high1) & (low2 < low1) & (df["Close"] > high2)
    return _evento(cond)


# ------------------------------------------------------------- E6 --------
def entry_back_in_style(df: pd.DataFrame) -> pd.Series:
    """
    High di 3 barre fa < Low corrente E Low corrente < High di 1 barra
    fa E High corrente < High di 2 barre fa E High di 1 barra fa <
    High di 2 barre fa -> struttura di compressione rialzista.
    """
    h, l = df["High"], df["Low"]
    cond = (
        (h.shift(3) < l)
        & (l < h.shift(1))
        & (h < h.shift(2))
        & (h.shift(1) < h.shift(2))
    )
    return _evento(cond)


# ------------------------------------------------------------- E7 --------
def entry_big_tail_bars(
    df: pd.DataFrame,
    period: float = 10,
    window: float = 2000,
    percentile: float = 0.90,
) -> pd.Series:
    """Prevalenza di bull tail bars nel decile alto della propria storia recente."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    bull_tail = (c > o) & ((o - l) > (c - o)) & (h > h.shift(1))
    bear_tail = (c < o) & ((h - o) > (o - c)) & (l < l.shift(1))
    bull_count = bull_tail.rolling(int(period)).sum()
    bear_count = bear_tail.rolling(int(period)).sum()

    soglia = bull_count.rolling(int(window)).quantile(percentile).shift(1)
    cond = (bull_count > bear_count) & (bull_count > soglia)
    return _evento(cond)


# ------------------------------------------------------------- E8 --------
def entry_closing_pattern_only(df: pd.DataFrame) -> pd.Series:
    """
    Close di 1 barra fa > Close di 3 barre fa E Close corrente > Close
    di 2 barre fa E Close di 2 barre fa > Close di 1 barra fa ->
    struttura di chiusure in accelerazione rialzista.
    """
    c = df["Close"]
    cond = (c.shift(1) > c.shift(3)) & (c > c.shift(2)) & (c.shift(2) > c.shift(1))
    return _evento(cond)


# ------------------------------------------------------------- E9 --------
def entry_closing_pattern_only_ii(df: pd.DataFrame) -> pd.Series:
    """
    Close di 1 barra fa < Close di 2 barre fa < Close di 5 barre fa <
    Close di 3 barre fa < Close di 4 barre fa -> struttura di chiusure
    a "dente di sega" con recupero recente.
    """
    c = df["Close"]
    cond = (
        (c.shift(1) < c.shift(2))
        & (c.shift(2) < c.shift(5))
        & (c.shift(5) < c.shift(3))
        & (c.shift(3) < c.shift(4))
    )
    return _evento(cond)


# ------------------------------------------------------------ E10 --------
def entry_hammer(df: pd.DataFrame) -> pd.Series:
    """Hammer -> reversal rialzista dopo un downtrend."""
    return _cdl_bull(df, talib.CDLHAMMER)


# ------------------------------------------------------------ E11 --------
def entry_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """Inverted Hammer -> reversal rialzista dopo un downtrend."""
    return _cdl_bull(df, talib.CDLINVERTEDHAMMER)


def entry_inverted_hammer_confirmed(df: pd.DataFrame) -> pd.Series:
    """Inverted Hammer + chiusura rialzista sulla barra successiva."""
    return _confermato(_cdl_bull(df, talib.CDLINVERTEDHAMMER), _conferma_rialzista(df))


# ------------------------------------------------------------ E12 --------
def entry_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish Engulfing."""
    return _cdl_bull(df, talib.CDLENGULFING)


def entry_engulfing_confirmed(df: pd.DataFrame) -> pd.Series:
    """Bullish Engulfing + chiusura rialzista sulla barra successiva."""
    return _confermato(_cdl_bull(df, talib.CDLENGULFING), _conferma_rialzista(df))


# ------------------------------------------------------------ E13 --------
def entry_harami(df: pd.DataFrame) -> pd.Series:
    """Bullish Harami."""
    return _cdl_bull(df, talib.CDLHARAMI)


def entry_harami_confirmed(df: pd.DataFrame) -> pd.Series:
    """Bullish Harami + chiusura rialzista sulla barra successiva."""
    return _confermato(_cdl_bull(df, talib.CDLHARAMI), _conferma_rialzista(df))


# ------------------------------------------------------------ E14 --------
def entry_harami_cross(df: pd.DataFrame) -> pd.Series:
    """Bullish Harami Cross."""
    return _cdl_bull(df, talib.CDLHARAMICROSS)


def entry_harami_cross_confirmed(df: pd.DataFrame) -> pd.Series:
    """Bullish Harami Cross + chiusura rialzista sulla barra successiva."""
    return _confermato(_cdl_bull(df, talib.CDLHARAMICROSS), _conferma_rialzista(df))


# ------------------------------------------------------------ E15 --------
def entry_piercing(df: pd.DataFrame) -> pd.Series:
    """Piercing Pattern -> reversal rialzista dopo un downtrend."""
    return _cdl_bull(df, talib.CDLPIERCING)


# ------------------------------------------------------------ E16 --------
def entry_morning_star(df: pd.DataFrame) -> pd.Series:
    """Morning Star -> reversal rialzista dopo un downtrend."""
    return _cdl_bull(df, talib.CDLMORNINGSTAR)


# ------------------------------------------------------------ E17 --------
def entry_three_inside(df: pd.DataFrame) -> pd.Series:
    """Three Inside Up."""
    return _cdl_bull(df, talib.CDL3INSIDE)


# ------------------------------------------------------------ E18 --------
def entry_three_outside(df: pd.DataFrame) -> pd.Series:
    """Three Outside Up."""
    return _cdl_bull(df, talib.CDL3OUTSIDE)


# ------------------------------------------------------------ E19 --------
def entry_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three Advancing White Soldiers."""
    return _cdl_bull(df, talib.CDL3WHITESOLDIERS)


# ------------------------------------------------------------ E20 --------
def entry_marubozu(df: pd.DataFrame) -> pd.Series:
    """Bullish (Closing) Marubozu -> candela di piena convinzione rialzista."""
    return _cdl_bull(df, talib.CDLMARUBOZU)


# ------------------------------------------------------------ E21 --------
def entry_belt_hold(df: pd.DataFrame) -> pd.Series:
    """Bullish Belt-hold."""
    return _cdl_bull(df, talib.CDLBELTHOLD)


def entry_belt_hold_confirmed(df: pd.DataFrame) -> pd.Series:
    """Bullish Belt-hold + chiusura rialzista sulla barra successiva."""
    return _confermato(_cdl_bull(df, talib.CDLBELTHOLD), _conferma_rialzista(df))

# =========================================================================
# Dizionario nome -> funzione. E' l'unica cosa da toccare per aggiungere
# un trigger: scrivere la funzione qui sopra e aggiungere una riga qui.
# =========================================================================
TRIGGER_LONG = {
    "E1_RSI_CROSS_OVERSOLD": entry_rsi_cross_oversold,
    "E2_ZLEMA_CROSS_UP": entry_zlema_cross_up,
    "E3_LONG_INTRABAR_WEAKNESS_FADE": entry_long_intrabar_weakness_fade,
    "E4_EMA_CROSS_UP": entry_ema_cross_up,
    "E5_QUICK_PULLBACK": entry_quick_pullback,
    "E6_BACK_IN_STYLE": entry_back_in_style,
    "E7_BIG_TAIL_BARS": entry_big_tail_bars,
    "E8_CLOSING_PATTERN_ONLY": entry_closing_pattern_only,
    "E9_CLOSING_PATTERN_ONLY_II": entry_closing_pattern_only_ii,
    "E10_HAMMER": entry_hammer,
    "E11_INVERTED_HAMMER": entry_inverted_hammer,
    "E11_INVERTED_HAMMER_CONFIRMED": entry_inverted_hammer_confirmed,
    "E12_ENGULFING": entry_engulfing,
    "E12_ENGULFING_CONFIRMED": entry_engulfing_confirmed,
    "E13_HARAMI": entry_harami,
    "E13_HARAMI_CONFIRMED": entry_harami_confirmed,
    "E14_HARAMI_CROSS": entry_harami_cross,
    "E14_HARAMI_CROSS_CONFIRMED": entry_harami_cross_confirmed,
    "E15_PIERCING": entry_piercing,
    "E16_MORNING_STAR": entry_morning_star,
    "E17_THREE_INSIDE": entry_three_inside,
    "E18_THREE_OUTSIDE": entry_three_outside,
    "E19_THREE_WHITE_SOLDIERS": entry_three_white_soldiers,
    "E20_MARUBOZU": entry_marubozu,
    "E21_BELT_HOLD": entry_belt_hold,
    "E21_BELT_HOLD_CONFIRMED": entry_belt_hold_confirmed,
}


def aggiungi_trigger_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola tutti i trigger long e li aggiunge come colonne al df.
    Non modifica il df originale: ne restituisce una copia.
    """
    df = df.copy()
    for nome_colonna, funzione in TRIGGER_LONG.items():
        df[nome_colonna] = funzione(df)
    return df


def registra_trigger_long():
    """
    Registra tutti i trigger long nel motore (engine.registry), cosi'
    run_event_study (e il resto del framework) li trova da solo.

    Va chiamata una volta prima di run_event_study. E' sicura da
    richiamare piu' volte nella stessa sessione: i nomi gia' registrati
    vengono saltati con un avviso, non sollevano errore.
    """
    from engine.registry import register_entry, list_entries

    gia_presenti = set(list_entries())
    nuovi = 0
    for nome, funzione in TRIGGER_LONG.items():
        if nome in gia_presenti:
            print(f"[registra_trigger_long] '{nome}' gia' registrata, salto.")
            continue
        register_entry(nome, direction=1)(funzione)
        nuovi += 1
    print(f"[registra_trigger_long] {nuovi} entry long registrate "
          f"({len(TRIGGER_LONG) - nuovi} gia' presenti).")
