import numpy as np
import pandas as pd

# =============================================================================
# UTILITY — mappatura colonne ohlcv_* ↔ uppercase (HIMALAYA)
# =============================================================================

def feat_historical_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-return e rolling std su 4 finestre temporali.
    feat_log_return è colonna intermedia — verrà droppata alla fine.
    """
    df['feat_log_return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
    for w in [5, 20, 50, 200]:
        df[f'feat_volatility_{w}'] = df['feat_log_return'].rolling(window=w).std()
    return df


def feat_volatility_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio breve/lungo: misura l'accelerazione della volatilità.
    Ratio > 1 → vol a breve sta esplodendo rispetto alla norma storica.
    """
    df['feat_vol_ratio_5_50']   = df['feat_volatility_5']  / df['feat_volatility_50']
    df['feat_vol_ratio_20_200'] = df['feat_volatility_20'] / df['feat_volatility_200']
    # Derivata della vol: sta accelerando o decelerando?
    df['feat_vol_delta_20'] = df['feat_volatility_20'] - df['feat_volatility_20'].shift(5)
    return df


def feat_arch_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return al quadrato (termine ARCH) e varianza realizzata rolling.
    Sono i predittori fondamentali nei modelli GARCH.
    """
    df['feat_log_return_sq'] = df['feat_log_return'] ** 2
    for w in [10, 20, 50]:
        df[f'feat_realized_var_{w}'] = (
            df['feat_log_return_sq']
            .rolling(window=w, min_periods=w)
            .sum()
        )
    return df


def feat_return_moments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skewness e kurtosis rolling su 50 barre.
    Skewness negativa forte → precursore di alta vol (crash asymmetry).
    Alta kurtosis → tail events più frequenti → regime HIGH imminente.
    """
    df['feat_skewness_50'] = df['feat_log_return'].rolling(window=50, min_periods=50).skew()
    df['feat_kurtosis_50'] = df['feat_log_return'].rolling(window=50, min_periods=50).kurt()
    return df


def feat_high_low_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Range intrabar normalizzato: misura la volatilità realizzata barra per barra,
    indipendente dai return (complementare alla rolling std).
    """
    df['feat_hl_range']          = (df['High'] - df['Low']) / df['Close']
    df['feat_hl_range_mean_20']  = df['feat_hl_range'].rolling(20).mean()
    df['feat_hl_range_mean_50']  = df['feat_hl_range'].rolling(50).mean()
    # Range ratio: barra corrente espansa rispetto alla norma?
    df['feat_hl_range_ratio']    = df['feat_hl_range'] / df['feat_hl_range_mean_50']
    return df


def feat_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anomalie di volume: spike di volume spesso precedono spike di volatilità.
    Le colonne intermedie (mean, std) vengono droppate alla fine.
    """
    vol_mean_50 = df['Volume'].rolling(50).mean()
    vol_std_50  = df['Volume'].rolling(50).std()
    vol_mean_20 = df['Volume'].rolling(20).mean()

    df['feat_volume_zscore'] = (df['Volume'] - vol_mean_50) / vol_std_50
    df['feat_volume_ratio']  = df['Volume'] / vol_mean_20
    return df

# =============================================================================
# UTILITY — LABELS - ETICHETTATURA DEI REGIMI DI VOLATILITA'
# =============================================================================
def build_labels(
    df: pd.DataFrame,
    vol_col: str = 'feat_volatility_5',
    window: int = 500,
    q_low: float = 0.33,
    q_high: float = 0.66,
) -> pd.DataFrame:
    vol = df[vol_col]
    q_low_series  = vol.rolling(window, min_periods=window).quantile(q_low)
    q_high_series = vol.rolling(window, min_periods=window).quantile(q_high)

    conditions = [
        vol <= q_low_series,
        (vol > q_low_series) & (vol <= q_high_series),
        vol > q_high_series,
    ]
    df['vol_regime'] = np.select(conditions, [0, 1, 2], default=np.nan)
    df['vol_regime'] = df['vol_regime'].where(q_low_series.notna(), other=np.nan)

    # Feature di posizione relativa nelle bande
    band_width = q_high_series - q_low_series + 1e-9
    df['feat_pos_in_band']         = (vol - q_low_series) / band_width
    df['feat_dist_to_high_q']      = (q_high_series - vol) / band_width
    df['feat_mid_band_width_norm'] = band_width / vol
    df['feat_approaching_high']    = df['feat_dist_to_high_q'].diff(3)
    df['feat_approaching_low']     = df['feat_pos_in_band'].diff(3)

    return df


def feat_calendar(df):
    idx = df.index
    df['feat_hour']        = idx.hour
    df['feat_day_of_week'] = idx.dayofweek
    df['feat_is_london_open']   = ((idx.hour >= 7) & (idx.hour < 10)).astype(int)
    df['feat_is_ny_open']       = ((idx.hour >= 13) & (idx.hour < 17)).astype(int)
    df['feat_is_overlap']       = ((idx.hour >= 13) & (idx.hour < 16)).astype(int)
    df['feat_is_low_liquidity'] = ((idx.hour >= 22) | (idx.hour < 6)).astype(int)
    return df

# =============================================================================
# UTILITY — mappatura colonne ohlcv_* ↔ uppercase (HIMALAYA)
# =============================================================================
def build_autoregressive_features(df: pd.DataFrame, lags: list = [1, 2, 3, 5, 10, 20],) -> pd.DataFrame:
    """
    Aggiunge le feature autoregressive che catturano la memoria del regime.

    Features aggiunte:
      feat_regime_lag_N      : regime delle ultime N barre (ordinale 0/1/2)
      feat_regime_duration   : quante barre consecutive nel regime corrente
      feat_vol50_lag_N       : livello di vol 50 delle ultime N barre

    La durata del regime è il predittore più diretto della persistenza:
    se siamo in HIGH vol da 30 barre, è più probabile restare in HIGH
    che se ci siamo entrati 2 barre fa.
    """
    # Lag del regime corrente — predittore chiave dell'autoregressività
    for lag in lags:
        df[f'feat_regime_lag_{lag}'] = df['vol_regime'].shift(lag)

    # Durata del regime corrente (quante barre consecutive nello stesso regime)
    regime = df['vol_regime']
    regime_change = regime != regime.shift(1)
    group_id = regime_change.cumsum()
    df['feat_regime_duration'] = group_id.groupby(group_id).cumcount() + 1
    # In build_autoregressive_features:
    df['feat_regime_transition_count'] = (df['vol_regime'].diff().abs().rolling(10).sum())
    df['feat_vol5_trend_short'] = df['feat_volatility_5'].diff(3)
    df['feat_vol5_trend_medium'] = df['feat_volatility_5'].diff(10)

    # Lag della volatilità 50 (livello assoluto, non solo il regime discreto)
    for lag in [1, 3, 5, 10]:
        df[f'feat_vol50_lag_{lag}'] = df['feat_volatility_50'].shift(lag)
    df.dropna(inplace=True)

    return df


# =============================================================================
# UTILITY — PIPELINE DI COSTRUZIONE FEATURES
# =============================================================================
def build_volatility_features(
    df: pd.DataFrame,
    vol_col: str = 'feat_volatility_5',
    label_window: int = 500,
    q_low: float = 0.33,
    q_high: float = 0.66,
) -> pd.DataFrame:
    """
    Entry point del feature engineering. Chiama tutte le funzioni
    nell'ordine corretto e rimuove le colonne intermedie.

    Input:  DataFrame con colonne Open, High, Low, Close, Volume
    Output: DataFrame con sole colonne feat_* (+ OHLCV originali preservati
            per il labeling che viene dopo)

    Parametri del labeling (prima erano hardcoded in build_labels):
      vol_col      : colonna di volatilità su cui calcolare i regimi
      label_window : finestra rolling per i quantili dei regimi
      q_low/q_high : quantili che separano LOW/MID/HIGH (default 0.33/0.66)
    """
    df = df.copy()
    df = feat_historical_volatility(df)   # deve essere prima di tutto
    #df = feat_volatility_ratios(df)
    #df = feat_arch_terms(df)
    #df = feat_return_moments(df)
    #df = feat_high_low_range(df)
    df = feat_volume(df)
    # Feature Labels — ora parametrizzate dal chiamante
    df = build_labels(df, vol_col=vol_col, window=label_window,
                      q_low=q_low, q_high=q_high)
    # Feature autoregressive
    #df = build_autoregressive_features(df)
    #df = feat_calendar(df)

    # Step 7 — Rimuovi colonne OHLCV e intermedie
    cols_to_drop = ['feat_log_return']
    df.drop(columns=cols_to_drop, inplace=True)
    #df.dropna(inplace=True)
    return df
