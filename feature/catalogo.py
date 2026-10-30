"""
Il catalogo delle feature di contesto (lavoro C, 30/10/2026, docs/02 §C e §F).

  CATALOGO : dict  nome -> {famiglia, finestre_ore, finestra_ore, formula, funzione, richiede, conteggio}
  Contesto(df, timeframe, trigger_lato=None, df_btc=None)

REGOLE (docs/02 §C, decise prima di guardare i dati, nessun parametro ottimizzato)
------
- Finestre W = 4, 24, 168 ore; finestra lunga L = 720 ore. Le ore diventano barre
  con indicatori.barre_per_durata di V4 (senza il suo minimo di 20 barre). Se una
  finestra di una feature da' meno di MIN_BARRE = 6 barre, la feature non si calcola.
- r = rendimento logaritmico di una barra (ln C_t - ln C_{t-1}); sigma_W = deviazione
  standard di r sulle ultime n_W barre. Tutte le finestre mobili comprendono la barra
  corrente e solo barre passate: la barra t usa le barre chiuse fino a t compresa.
- Adimensionali: rapporti, logaritmi di rapporti, z-score, percentili. Mai il livello
  del prezzo o del volume.
- Una finestra mobile vale solo se e' piena (min_periods = n): niente valori costruiti
  su meno barre di quelle dichiarate.

CAUSALITA' (precisazioni di Mattia per il lavoro C)
- vol_ora: la mediana della stessa ora usa solo i giorni PRIMA della barra corrente;
- trig_24h e da_trig_prec: escludono il trigger corrente;
- famiglia 10: BTC dello stesso exchange, mercato e timeframe, allineato sullo stesso
  indice SENZA riempire i buchi (barra mancante = NaN, e NaN in ogni finestra che la
  contiene);
- percentili e normalizzazioni: solo sulla finestra passata, barra corrente compresa.

DA V4 (importati, non riscritti): efficienza_kaufman, variance_ratio, barre_per_durata,
il prezzo tipico del VWAP (vwap_ops.vwap_anchored_daily usa (H+L+C)/3). V4 non ha un
VWAP mobile: quello su W si calcola qui con lo stesso prezzo tipico.
"""
from functools import cached_property

import numpy as np
import pandas as pd

W = (4, 24, 168)
L = 720
MIN_BARRE = 6
GIORNI_VOL_ORA = 20
RITARDI = range(1, 7)
CANDELE = (0, 1, 2)
ORA_RIF = 24          # sigma di riferimento per le distanze: sigma_24h


def _pulisci(s: pd.Series) -> pd.Series:
    """Le divisioni per zero diventano NaN, non infinito."""
    return s.replace([np.inf, -np.inf], np.nan)


class Contesto:
    """I pezzi comuni alle feature, calcolati una volta: barre per finestra, r, sigma."""

    def __init__(self, df, timeframe, trigger_lato=None, df_btc=None):
        self.df = df
        self.timeframe = timeframe
        self.trigger = None if trigger_lato is None else (
            pd.Series(trigger_lato).reindex(df.index).fillna(False).astype(bool))
        # BTC sullo stesso indice, SENZA riempire: le barre che mancano restano NaN
        self.btc = None if df_btc is None else df_btc[["Open", "High", "Low", "Close", "Volume"]].reindex(df.index)
        self._n = {}

    def n(self, ore) -> int:
        """Barre del timeframe in `ore` ore (indicatori.barre_per_durata di V4, senza minimo)."""
        if ore not in self._n:
            from engine.indicatori import barre_per_durata
            self._n[ore] = barre_per_durata(self.df.index, ore, minimo=0)
        return self._n[ore]

    def r_di(self, close):
        return np.log(close.astype(float)).diff()

    @cached_property
    def r(self):
        return self.r_di(self.df["Close"])

    @cached_property
    def r_btc(self):
        return self.r_di(self.btc["Close"])

    def sigma(self, ore, btc=False):
        chiave = ("sigma_btc" if btc else "sigma", ore)
        if chiave not in self._n:
            r = self.r_btc if btc else self.r
            self._n[chiave] = r.rolling(self.n(ore)).std()
        return self._n[chiave]


# ---------------------------------------------------------------- famiglia 1
def _ret_lag(k):
    return lambda c: _pulisci(c.r.shift(k - 1) / c.sigma(ORA_RIF))


def _ret_cum(ore, btc=False):
    def f(c):
        close = (c.btc if btc else c.df)["Close"].astype(float)
        n = c.n(ore)
        return _pulisci(np.log(close / close.shift(n)) / (c.sigma(ore, btc) * np.sqrt(n)))
    return f


# ---------------------------------------------------------------- famiglia 2
def _vol(ore, btc=False):
    return lambda c: _pulisci(c.sigma(ore, btc) / c.sigma(L, btc))


def _vol_breve_lunga(c):
    return _pulisci(c.sigma(4) / c.sigma(168))


def _vol_pct(c):
    return c.sigma(24).rolling(c.n(L)).rank(pct=True)


def _parkinson(c):
    hl = np.log(c.df["High"].astype(float) / c.df["Low"].astype(float)) ** 2
    s = np.sqrt(hl.rolling(c.n(24)).mean() / (4 * np.log(2)))
    return _pulisci(s / c.sigma(24))


def _garman_klass(c):
    d = c.df.astype(float)
    termini = 0.5 * np.log(d["High"] / d["Low"]) ** 2 - (2 * np.log(2) - 1) * np.log(d["Close"] / d["Open"]) ** 2
    s = np.sqrt(termini.rolling(c.n(24)).mean().clip(lower=0))
    return _pulisci(s / c.sigma(24))


# ---------------------------------------------------------------- famiglia 3
def _er(ore):
    def f(c):
        from engine.indicatori import efficienza_kaufman
        return efficienza_kaufman(c.df["Close"], c.n(ore))
    return f


def _vr_24h(c):
    # variance_ratio di V4 sulla finestra di 24 h, con l'orizzonte di V4 (default: 4 barre)
    from engine.indicatori import variance_ratio
    return variance_ratio(c.df["Close"], finestra=c.n(24))


def _ac1_168h(c):
    n = c.n(168)
    return c.r.rolling(n).corr(c.r.shift(1))


def _regressione_mobile(y: np.ndarray, n: int):
    """
    Pendenza e R^2 della regressione di y sul tempo, su ogni finestra di n barre che
    finisce in t. Il tempo e' centrato nella finestra (somma zero): la pendenza non
    dipende dal livello di y, quindi ln(10 C) da' la stessa pendenza di ln C anche negli
    arrotondamenti. (La covarianza mobile di pandas su tempo assoluto x ln C accumula
    errori sopra 1e-9: lavoro C, test degli adimensionali.)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    pendenza = np.full(len(y), np.nan)
    r2 = np.full(len(y), np.nan)
    if len(y) < n:
        return pendenza, r2
    tc = np.arange(n, dtype=float) - (n - 1) / 2.0
    stt = float(tc @ tc)
    passo = max(1000, int(2e7 // n))
    for inizio in range(0, len(y) - n + 1, passo):
        blocco = sliding_window_view(y[inizio: inizio + passo + n - 1], n)
        sty = blocco @ tc
        syy = blocco.var(axis=1) * n
        fine = inizio + n - 1
        pendenza[fine: fine + len(blocco)] = sty / stt
        with np.errstate(invalid="ignore", divide="ignore"):
            r2[fine: fine + len(blocco)] = np.where(syy > 0, sty * sty / (stt * syy), np.nan)
    return pendenza, np.clip(r2, 0, 1)


def _regressione(ore):
    """(pendenza in sigma_W per barra, R^2) della regressione di ln C sul tempo, sulle ultime n_W barre."""
    def calcola(c):
        chiave = ("regressione", ore)
        if chiave not in c._n:
            y = np.log(c.df["Close"].to_numpy(dtype=float))
            pendenza, r2 = _regressione_mobile(y, c.n(ore))
            idx = c.df.index
            c._n[chiave] = (_pulisci(pd.Series(pendenza, index=idx) / c.sigma(ore)), pd.Series(r2, index=idx))
        return c._n[chiave]
    return (lambda c: calcola(c)[0]), (lambda c: calcola(c)[1])


# ---------------------------------------------------------------- famiglia 4
def _pos_range(ore):
    def f(c):
        n = c.n(ore)
        alto = c.df["High"].astype(float).rolling(n).max()
        basso = c.df["Low"].astype(float).rolling(n).min()
        return _pulisci((c.df["Close"] - basso) / (alto - basso))
    return f


def _dist_max(ore):
    def f(c):
        alto = c.df["High"].astype(float).rolling(c.n(ore)).max()
        return _pulisci(np.log(alto / c.df["Close"].astype(float)) / c.sigma(ORA_RIF))
    return f


def _dist_min(ore):
    def f(c):
        basso = c.df["Low"].astype(float).rolling(c.n(ore)).min()
        return _pulisci(np.log(c.df["Close"].astype(float) / basso) / c.sigma(ORA_RIF))
    return f


def _barre_da_estremo(valori: np.ndarray, n: int, massimo: bool) -> np.ndarray:
    """Barre trascorse dall'ultimo massimo (minimo) nelle ultime n barre, barra corrente compresa (0 = adesso)."""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(valori), np.nan)
    if len(valori) < n:
        return out
    passo = max(1000, int(2e7 // n))
    for inizio in range(0, len(valori) - n + 1, passo):
        blocco = sliding_window_view(valori[inizio: inizio + passo + n - 1], n)[:, ::-1]   # dalla barra corrente indietro
        out[inizio + n - 1: inizio + n - 1 + len(blocco)] = (
            np.argmax(blocco, axis=1) if massimo else np.argmin(blocco, axis=1))
    return out


def _da_estremo(ore, massimo):
    def f(c):
        n = c.n(ore)
        col = "High" if massimo else "Low"
        v = _barre_da_estremo(c.df[col].to_numpy(dtype=float), n, massimo)
        return pd.Series(v / n, index=c.df.index)
    return f


# ---------------------------------------------------------------- famiglia 5
def _dist_vwap(ore):
    def f(c):
        n = c.n(ore)
        d = c.df.astype(float)
        tipico = (d["High"] + d["Low"] + d["Close"]) / 3            # come vwap_ops.vwap_anchored_daily di V4
        vwap = (tipico * d["Volume"]).rolling(n).sum() / d["Volume"].rolling(n).sum()
        return _pulisci(np.log(d["Close"] / vwap) / c.sigma(ORA_RIF))
    return f


def _dist_ema(ore):
    def f(c):
        n = c.n(ore)
        close = c.df["Close"].astype(float)
        ema = close.ewm(span=n, adjust=False, min_periods=n).mean()
        return _pulisci(np.log(close / ema) / c.sigma(ORA_RIF))
    return f


# ---------------------------------------------------------------- famiglia 6
def _asim(c):
    return c.r.rolling(c.n(168)).skew()


def _curt(c):
    return c.r.rolling(c.n(168)).kurt()


# ---------------------------------------------------------------- famiglia 7
def _vol_ora(c):
    """
    Volume / mediana del volume della stessa barra del giorno (stesso orario UTC) nei
    GIORNI_VOL_ORA giorni precedenti con quella barra. La barra corrente non entra
    nella mediana (shift(1) dentro ogni orario): entra solo nel numeratore.
    """
    v = c.df["Volume"].astype(float)
    orario = c.df.index.strftime("%H:%M")
    mediana = v.groupby(orario).transform(lambda s: s.shift(1).rolling(GIORNI_VOL_ORA).median())
    return _pulisci(v / mediana)


def _vol_z(c):
    lv = np.log(c.df["Volume"].astype(float).where(c.df["Volume"] > 0))
    n = c.n(168)
    return _pulisci((lv - lv.rolling(n).mean()) / lv.rolling(n).std())


def _vol_mov(c):
    n = c.n(24)
    a = _pulisci(c.df["Volume"].astype(float).rolling(n).sum() / c.r.abs().rolling(n).sum())
    return _pulisci(a / a.rolling(c.n(L)).median())


# ---------------------------------------------------------------- famiglia 8
def _candela(parte, k):
    def f(c):
        d = c.df.astype(float)
        rng = d["High"] - d["Low"]
        if parte == "corpo":
            x = d["Close"] - d["Open"]
        elif parte == "su":
            x = d["High"] - d[["Open", "Close"]].max(axis=1)
        else:
            x = d[["Open", "Close"]].min(axis=1) - d["Low"]
        valore = (x / rng).where(rng > 0, 0.0)                     # range nullo -> 0 (docs/02 §C)
        return valore.shift(k)
    return f


# ---------------------------------------------------------------- famiglia 9
def _tempo(cosa, funzione):
    def f(c):
        idx = c.df.index
        if cosa == "ora":
            x = 2 * np.pi * (idx.hour + idx.minute / 60.0) / 24.0
        else:
            x = 2 * np.pi * idx.dayofweek / 7.0
        return pd.Series(funzione(np.asarray(x, dtype=float)), index=idx)
    return f


# ---------------------------------------------------------------- famiglia 10
def _corr_btc(c):
    return c.r.rolling(c.n(168)).corr(c.r_btc)


# ---------------------------------------------------------------- famiglia 11
def _trig_24h(c):
    """Trigger del lato nelle n_24h barre PRIMA della corrente (il trigger corrente escluso)."""
    return c.trigger.astype(float).shift(1).rolling(c.n(24)).sum()


def _da_trig_prec(c):
    """Barre dal trigger precedente (escluso il corrente) / n_24h, con tetto a 1; nessun precedente -> 1."""
    pos = pd.Series(np.where(c.trigger.to_numpy(), np.arange(len(c.trigger), dtype=float), np.nan),
                    index=c.trigger.index)
    ultimo = pos.shift(1).ffill()                                  # l'ultimo trigger PRIMA della barra
    barre = np.arange(len(pos), dtype=float) - ultimo
    return (barre / c.n(24)).clip(upper=1.0).fillna(1.0)


# ======================================================================== il catalogo
CATALOGO = {}


def _metti(nome, famiglia, finestre, formula, funzione, richiede=None, conteggio=False):
    CATALOGO[nome] = {"famiglia": famiglia, "finestre_ore": tuple(finestre),
                      "finestra_ore": max(finestre) if finestre else None, "formula": formula,
                      "funzione": funzione, "richiede": richiede, "conteggio": conteggio}


for k in RITARDI:
    _metti(f"ret_lag_{k}", "1 rendimenti", (24,), f"r(t-{k - 1}) / sigma_24h", _ret_lag(k))
for w in W:
    _metti(f"ret_cum_{w}h", "1 rendimenti", (w,), f"ln(C_t / C_t-n{w}) / (sigma_{w}h * sqrt(n))", _ret_cum(w))
for w in W:
    _metti(f"vol_{w}h", "2 volatilita'", (w, L), f"sigma_{w}h / sigma_{L}h", _vol(w))
_metti("vol_breve_lunga", "2 volatilita'", (4, 168), "sigma_4h / sigma_168h", _vol_breve_lunga)
_metti("vol_pct", "2 volatilita'", (24, L), "percentile di sigma_24h nelle ultime 720 h", _vol_pct)
_metti("park_24h", "2 volatilita'", (24,), "Parkinson su 24 h / sigma_24h", _parkinson)
_metti("gk_24h", "2 volatilita'", (24,), "Garman-Klass su 24 h / sigma_24h", _garman_klass)
for w in W:
    _metti(f"er_{w}h", "3 efficienza", (w,), "efficiency ratio di Kaufman (V4, con segno)", _er(w))
_metti("vr_24h", "3 efficienza", (24,), "variance ratio di V4 su 24 h (orizzonte di V4: 4 barre)", _vr_24h)
_metti("ac1_168h", "3 efficienza", (168,), "autocorrelazione di r a ritardo 1 su 168 h", _ac1_168h)
for w in W:
    pend, r2 = _regressione(w)
    _metti(f"pend_{w}h", "3 efficienza", (w,), "pendenza di ln C sul tempo, in sigma_W per barra", pend)
    _metti(f"r2_{w}h", "3 efficienza", (w,), "R^2 della regressione di ln C sul tempo", r2)
for w in W:
    _metti(f"pos_range_{w}h", "4 range", (w,), "(C - min Low) / (max High - min Low)", _pos_range(w))
    _metti(f"dist_max_{w}h", "4 range", (w, 24), "ln(max High / C) / sigma_24h", _dist_max(w))
    _metti(f"dist_min_{w}h", "4 range", (w, 24), "ln(C / min Low) / sigma_24h", _dist_min(w))
    _metti(f"da_max_{w}h", "4 range", (w,), "barre dall'ultimo massimo / n_W", _da_estremo(w, True))
    _metti(f"da_min_{w}h", "4 range", (w,), "barre dall'ultimo minimo / n_W", _da_estremo(w, False))
for w in W:
    _metti(f"dist_vwap_{w}h", "5 distanze", (w, 24), "ln(C / VWAP mobile su W) / sigma_24h", _dist_vwap(w))
    _metti(f"dist_ema_{w}h", "5 distanze", (w, 24), "ln(C / EMA di n_W barre) / sigma_24h", _dist_ema(w))
_metti("asim_168h", "6 forma", (168,), "asimmetria di r su 168 h", _asim)
_metti("curt_168h", "6 forma", (168,), "curtosi di r su 168 h", _curt)
_metti("vol_ora", "7 volume", (), f"Volume / mediana stessa ora UTC dei {GIORNI_VOL_ORA} giorni prima", _vol_ora)
_metti("vol_z_168h", "7 volume", (168,), "z-score di ln Volume su 168 h", _vol_z)
_metti("vol_mov_24h", "7 volume", (24, L), "(sum Volume / sum |r|) su 24 h / sua mediana su 720 h", _vol_mov)
for k in CANDELE:
    _metti(f"corpo_{k}", "8 candela", (), f"(C-O)/(H-L) di {k} barre fa", _candela("corpo", k))
    _metti(f"ombra_su_{k}", "8 candela", (), f"(H-max(O,C))/(H-L) di {k} barre fa", _candela("su", k))
    _metti(f"ombra_giu_{k}", "8 candela", (), f"(min(O,C)-L)/(H-L) di {k} barre fa", _candela("giu", k))
_metti("ora_sin", "9 tempo", (), "sen(2 pi ora UTC / 24)", _tempo("ora", np.sin))
_metti("ora_cos", "9 tempo", (), "cos(2 pi ora UTC / 24)", _tempo("ora", np.cos))
_metti("giorno_sin", "9 tempo", (), "sen(2 pi giorno / 7)", _tempo("giorno", np.sin))
_metti("giorno_cos", "9 tempo", (), "cos(2 pi giorno / 7)", _tempo("giorno", np.cos))
for w in W:
    _metti(f"btc_ret_cum_{w}h", "10 BTC", (w,), "ret_cum di BTC", _ret_cum(w, btc=True), richiede="btc")
_metti("btc_vol_24h", "10 BTC", (24, L), "sigma_24h / sigma_720h di BTC", _vol(24, btc=True), richiede="btc")
_metti("corr_btc_168h", "10 BTC", (168,), "correlazione di r con r di BTC su 168 h", _corr_btc, richiede="btc")
_metti("trig_24h", "11 trigger", (24,), "trigger del lato nelle 24 h prima (escluso il corrente)", _trig_24h,
       richiede="trigger", conteggio=True)
_metti("da_trig_prec", "11 trigger", (24,), "barre dal trigger precedente / n_24h, tetto 1", _da_trig_prec,
       richiede="trigger")
