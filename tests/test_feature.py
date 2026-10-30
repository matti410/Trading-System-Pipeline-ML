"""
Il catalogo delle feature (lavoro C): lookahead con verifica_lookahead di V4, adimensionali,
ore -> barre, finestre troppo corte, causalita' di vol_ora, trigger e BTC, degenerazione.
"""
import numpy as np
import pandas as pd
import pytest

from feature import calcolo
from feature.catalogo import CATALOGO, L, MIN_BARRE, W

FREQ = {"M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h"}


def mercato(n=4000, tf="M15", seme=0, prezzo=20_000.0):
    rng = np.random.default_rng(seme)
    idx = pd.date_range("2023-01-02", periods=n, freq=FREQ[tf], tz="UTC")
    close = prezzo * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    apertura = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.0005, n))
    alto = np.maximum(apertura, close) * (1 + rng.uniform(0, 0.002, n))
    basso = np.minimum(apertura, close) * (1 - rng.uniform(0, 0.002, n))
    volume = rng.lognormal(3, 0.5, n)
    return pd.DataFrame({"Open": apertura, "High": alto, "Low": basso, "Close": close, "Volume": volume}, index=idx)


def trigger_di(df, quota=0.03, seme=1):
    return pd.Series(np.random.default_rng(seme).random(len(df)) < quota, index=df.index)


def test_catalogo_completo():
    assert len(CATALOGO) == 73
    famiglie = {v["famiglia"].split()[0] for v in CATALOGO.values()}
    assert famiglie == {str(i) for i in range(1, 12)}
    assert W == (4, 24, 168) and L == 720 and MIN_BARRE == 6


def test_ore_in_barre(v4):
    from engine.indicatori import barre_per_durata
    for tf, attese in (("M5", 288), ("M15", 96), ("H1", 24), ("H4", 6)):
        assert calcolo.barre_finestra(24, tf) == attese
        assert barre_per_durata(mercato(50, tf).index, 24, minimo=0) == attese


def test_finestra_sotto_6_barre_non_si_calcola_e_si_stampa(v4, capsys):
    df = mercato(6000, "H4")
    feat = calcolo.calcola_feature(df, "H4", trigger_lato=trigger_di(df), df_btc=mercato(6000, "H4", seme=5))
    uscita = capsys.readouterr().out
    corte = [n for n, v in CATALOGO.items() if 4 in v["finestre_ore"]]       # 4 h su H4 = 1 barra
    assert corte and not set(corte) & set(feat.columns)
    for n in corte:
        assert "sotto 6 barre" in feat.attrs["non_calcolate"][n] and n in uscita
    assert "ret_cum_24h" in feat.columns                                     # 24 h su H4 = 6 barre: si calcola


def test_adimensionali_prezzi_e_volume_per_10(v4):
    df, btc = mercato(), mercato(seme=7)
    tr = trigger_di(df)
    a = calcola_tutto(df, tr, btc)
    df10 = df.copy()
    df10[["Open", "High", "Low", "Close"]] *= 10
    df10["Volume"] *= 10
    b = calcola_tutto(df10, tr, btc)
    assert list(a.columns) == list(b.columns) and a.shape[1] == len(CATALOGO)
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), rtol=0, atol=1e-9, equal_nan=True)


def calcola_tutto(df, tr, btc):
    return calcolo.calcola_feature(df, "M15", trigger_lato=tr, df_btc=btc, verbose=False)


def test_vol_ora_mediana_senza_la_barra_corrente(v4):
    df = mercato(3000)
    t = 2500
    prima = calcolo.calcola_feature(df, "M15", nomi=["vol_ora"], verbose=False)["vol_ora"]
    df2 = df.copy()
    df2.iloc[t, df2.columns.get_loc("Volume")] *= 1000                      # volume della barra t stravolto
    dopo = calcolo.calcola_feature(df2, "M15", nomi=["vol_ora"], verbose=False)["vol_ora"]
    # la mediana alla barra t non cambia: cambia solo il numeratore
    mediana_prima = df["Volume"].iloc[t] / prima.iloc[t]
    mediana_dopo = df2["Volume"].iloc[t] / dopo.iloc[t]
    assert mediana_dopo == pytest.approx(mediana_prima, rel=1e-12)
    pd.testing.assert_series_equal(prima.iloc[:t], dopo.iloc[:t])
    # la mediana usa i 20 giorni prima, stesso orario
    stesso = df["Volume"].iloc[t - 96 * 20: t: 96]
    assert mediana_prima == pytest.approx(stesso.median(), rel=1e-12)


def test_trigger_corrente_escluso(v4):
    df = mercato(300)
    tr = pd.Series(False, index=df.index)
    tr.iloc[[100, 150, 160]] = True
    f = calcolo.calcola_feature(df, "M15", trigger_lato=tr, nomi=["trig_24h", "da_trig_prec"], verbose=False)
    assert f["trig_24h"].iloc[100] == 0            # il trigger della barra 100 non conta per se stesso
    assert f["trig_24h"].iloc[101] == 1
    assert f["trig_24h"].iloc[160] == 2            # conta 100 e 150, non 160 stesso
    assert f["trig_24h"].iloc[161] == 3
    assert f["trig_24h"].iloc[197] == 2            # il 100 esce dalle 96 barre prima della 197
    assert f["da_trig_prec"].iloc[160] == pytest.approx(10 / 96)   # dal 150, non dal 160 stesso
    assert f["da_trig_prec"].iloc[150] == pytest.approx(50 / 96)
    assert f["da_trig_prec"].iloc[99] == 1.0                       # nessun trigger prima: tetto
    assert f["da_trig_prec"].iloc[299] == 1.0                      # oltre 24 h: tetto a 1


def test_btc_con_un_buco_da_nan_senza_riempire(v4):
    df, btc = mercato(), mercato(seme=7)
    t = 3500
    nomi_btc = [n for n, v in CATALOGO.items() if v["richiede"] == "btc"]
    pieno = calcolo.calcola_feature(df, "M15", df_btc=btc, verbose=False)
    bucato = calcolo.calcola_feature(df, "M15", df_btc=btc.drop(btc.index[t]), verbose=False)
    assert pieno[nomi_btc].iloc[t].notna().all()
    assert bucato[nomi_btc].iloc[t].isna().all()                   # barra mancante = NaN, nessun riempimento
    altre = [c for c in pieno.columns if c not in nomi_btc]
    pd.testing.assert_frame_equal(pieno[altre], bucato[altre])     # le feature della moneta non cambiano


def test_percentile_sulla_finestra_passata(v4):
    df = mercato(4000)
    f = calcolo.calcola_feature(df, "M15", nomi=["vol_pct"], verbose=False)["vol_pct"]
    sigma = np.log(df["Close"]).diff().rolling(96).std()
    t, n = 3990, calcolo.barre_finestra(720, "M15")
    finestra = sigma.iloc[t - n + 1: t + 1]                        # ultime 720 h, barra corrente compresa
    assert f.iloc[t] == pytest.approx(finestra.rank(pct=True).iloc[-1])


def test_lookahead_inventato_e_controllo_al_contrario(v4, monkeypatch):
    df, btc = mercato(3600), mercato(3600, seme=7)
    tr = trigger_di(df)
    rep = calcolo.collaudo_lookahead(df, "M15", trigger_lato=tr, df_btc=btc, riscaldamento_tagli=3000)
    assert set(rep.index) >= set(CATALOGO)
    assert (rep.loc[list(CATALOGO), "lookahead"] == "OK").all()
    # una feature finta che guarda avanti di una barra deve uscire LOOKAHEAD
    monkeypatch.setitem(CATALOGO, "finta_futuro", {**CATALOGO["ret_lag_1"],
                        "funzione": lambda c: c.r.shift(-1)})
    rep = calcolo.collaudo_lookahead(df, "M15", trigger_lato=tr, df_btc=btc, riscaldamento_tagli=3000)
    assert rep.loc["finta_futuro", "lookahead"] == "LOOKAHEAD"
    assert (rep.loc[[n for n in CATALOGO if n != "finta_futuro"], "lookahead"] == "OK").all()


def test_lookahead_dati_veri_tutte_le_colonne(ds_vero, cartella_dati):
    """BTC perpetual M15 con il primo trigger long (famiglia 11) ed ETH con BTC (famiglia 10)."""
    from engine import vetrina
    tr = ds_vero.df[ds_vero.trigger_long[0]]
    _, risc = calcolo.riscaldamento(calcolo.calcola_feature(ds_vero.df.iloc[:6000], "M15", trigger_lato=tr,
                                                            verbose=False))
    pezzo = ds_vero.df.iloc[: risc + 1500]
    rep = calcolo.collaudo_lookahead(pezzo, "M15", trigger_lato=tr, riscaldamento_tagli=risc)
    nomi = [n for n, v in CATALOGO.items() if v["richiede"] != "btc"]
    assert (rep.loc[nomi, "lookahead"] == "OK").all(), rep[rep["lookahead"] != "OK"]
    eth = vetrina.carica_storico("BINANCE_PERP_ETHUSDT", cartella_dati, "M15")
    btc = vetrina.carica_storico("BINANCE_PERP_BTCUSDT", cartella_dati, "M15")
    pezzo_eth = eth.iloc[: risc + 1500]
    rep = calcolo.collaudo_lookahead(pezzo_eth, "M15", df_btc=btc, riscaldamento_tagli=risc)
    nomi = [n for n, v in CATALOGO.items() if v["richiede"] != "trigger"]
    assert (rep.loc[nomi, "lookahead"] == "OK").all(), rep[rep["lookahead"] != "OK"]


def test_adimensionali_dati_veri(ds_vero):
    pezzo = ds_vero.df.iloc[:5000][["Open", "High", "Low", "Close", "Volume"]]
    a = calcolo.calcola_feature(pezzo, "M15", verbose=False)
    p10 = pezzo.copy()
    p10[["Open", "High", "Low", "Close"]] *= 10
    p10["Volume"] *= 10
    b = calcolo.calcola_feature(p10, "M15", verbose=False)
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), rtol=0, atol=1e-9, equal_nan=True)


def test_degenerazione_continue():
    idx = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    feat = pd.DataFrame({"normale": rng.normal(size=200), "costante": 1.0, "vuota": np.nan,
                         "trig_24h": np.r_[np.zeros(190), np.ones(10)]}, index=idx)
    feat.iloc[:20, 0] = np.nan
    esito = calcolo.degenerazione_continue(feat)["degenerazione"]
    assert esito.to_dict() == {"normale": "OK", "costante": "QUASI_COSTANTE", "vuota": "MAI_CALCOLATA",
                               "trig_24h": "RARA"}


def test_riscaldamento_e_elenco(v4):
    df = mercato(4000)
    feat = calcolo.calcola_feature(df, "M15", verbose=False)
    nan_iniziali, totale = calcolo.riscaldamento(feat, riscaldamento_v4=300)
    assert nan_iniziali["vol_pct"] == 96 + calcolo.barre_finestra(720, "M15") - 1   # sigma_24h poi 720 h
    assert nan_iniziali["ora_sin"] == 0 and nan_iniziali["corpo_2"] == 2
    assert totale == 300 + nan_iniziali.max()
    el = calcolo.elenco_catalogo("M15", feat)
    assert len(el) == len(CATALOGO)
    assert el.loc["trig_24h", "esito"] == "non calcolata"
    assert el.loc["ret_cum_24h", "finestre_barre"] == "96"


def test_regressione_uguale_a_polyfit(v4):
    df = mercato(1000)
    f = calcolo.calcola_feature(df, "M15", nomi=["pend_24h", "r2_24h"], verbose=False)
    t, n = 900, 96
    y = np.log(df["Close"].to_numpy())[t - n + 1: t + 1]
    pendenza, intercetta = np.polyfit(np.arange(n), y, 1)
    r2 = np.corrcoef(np.arange(n), y)[0, 1] ** 2
    sigma = np.log(df["Close"]).diff().rolling(n).std().iloc[t]
    assert f["pend_24h"].iloc[t] == pytest.approx(pendenza / sigma, rel=1e-9)
    assert f["r2_24h"].iloc[t] == pytest.approx(r2, rel=1e-9)


def test_vr_24h_stesso_passo_in_ore_su_M5_e_M15(v4, monkeypatch, capsys):
    import engine.indicatori as ind
    passi = {}
    originale = ind.variance_ratio

    def spia(close, orizzonte=4, finestra=192):
        passi[tf] = (orizzonte, finestra)
        return originale(close, orizzonte=orizzonte, finestra=finestra)
    monkeypatch.setattr(ind, "variance_ratio", spia)
    minuti = {"M5": 5, "M15": 15}
    for tf in ("M5", "M15"):
        f = calcolo.calcola_feature(mercato(4000, tf), tf, nomi=["vr_24h"], verbose=False)
        assert f["vr_24h"].notna().any()
    assert passi["M5"][0] * minuti["M5"] == passi["M15"][0] * minuti["M15"] == 60     # 1 ora su tutti e due
    assert passi["M15"] == (4, 96) and passi["M5"] == (12, 288)
    # su H1 un'ora e' 1 barra (< 2): non si calcola e si stampa
    capsys.readouterr()
    f = calcolo.calcola_feature(mercato(500, "H1"), "H1", nomi=["vr_24h"], verbose=False)
    assert "vr_24h" not in f.columns and "passo sotto 2 barre" in f.attrs["non_calcolate"]["vr_24h"]
    assert "vr_24h" in capsys.readouterr().out
