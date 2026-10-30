"""costruisci_X con i filtri del Lab (VWAP_* compresi) e l'interruttore D1 dei pesi per |netto_bp|."""
import numpy as np
import pandas as pd
import pytest

from modello import addestra
from pesi import pesi_rendimento
from split import costruisci_X


def test_costruisci_X_usa_l_elenco_dei_filtri():
    idx = pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "Close": 1.0, "E1_A": [True, False] * 3, "F1_X": True,
                       "VWAP_GAP": [False, True] * 3, "F9_NON_IN_ELENCO": False}, index=idx)
    target = pd.DataFrame({"TARGET": [1, 0, 1]}, index=idx[[0, 2, 4]])
    X = costruisci_X(df, target, ["E1_A"], ["F1_X", "VWAP_GAP"])
    assert list(X.columns) == ["E1_A", "F1_X", "VWAP_GAP"]          # identita' del trigger + filtri dell'elenco
    assert X.index.equals(target.index) and (X.dtypes == int).all()
    with pytest.raises(KeyError):
        costruisci_X(df, target, ["E1_A"], ["NON_C_E"])
    with pytest.raises(ValueError):
        costruisci_X(df, target, ["E1_A"], ["Close"])               # mai colonne di prezzo


def test_costruisci_X_contiene_i_vwap_sui_dati_veri(ds_vero):
    from target import costruisci_target
    trig = ds_vero.trigger_long[:1]
    t = costruisci_target(ds_vero.df, trig, 1, 16, costi=ds_vero.costi)
    X = costruisci_X(ds_vero.df, t, trig, ds_vero.filtri)
    vwap = [c for c in ds_vero.filtri if c.startswith("VWAP_")]
    assert vwap and set(vwap) <= set(X.columns)
    assert set(X.columns) == set(trig) | set(ds_vero.filtri)


def test_pesi_rendimento_tetto_dal_train():
    netto = pd.Series(np.r_[np.arange(1, 101, dtype=float), 10_000.0])   # un trade estremo
    p = pesi_rendimento(netto)
    tetto = np.quantile(netto.abs(), 0.95)
    grezzo = np.minimum(netto.abs(), tetto)
    assert p.to_numpy() == pytest.approx((grezzo / grezzo.mean()).to_numpy())
    assert p.mean() == pytest.approx(1.0)
    assert p.iloc[-1] == pytest.approx(tetto / grezzo.mean())        # il trade estremo pesa quanto il tetto


def _dati_modello(n=600, seme=0):
    rng = np.random.default_rng(seme)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    X = pd.DataFrame({"a": rng.integers(0, 2, n), "b": rng.integers(0, 2, n)}, index=idx)
    y = pd.Series((rng.random(n) < 0.3 + 0.4 * X["a"]).astype(int), index=idx)
    netto = pd.Series(rng.normal(0, 50, n), index=idx)
    return X, y, netto, idx[:400], idx[400:]


def test_interruttore_pesi_rendimento_spento_di_default():
    X, y, netto, tr, te = _dati_modello()
    base = addestra(X, y, tr, te)
    spento = addestra(X, y, tr, te, netto_bp=netto)                     # netto passato ma interruttore spento
    acceso = addestra(X, y, tr, te, peso_rendimento=True, netto_bp=netto)
    p_base, p_spento, p_acceso = (m.predict_proba(X.loc[te])[:, 1] for m in (base, spento, acceso))
    assert np.array_equal(p_base, p_spento)
    assert not np.array_equal(p_base, p_acceso)
    with pytest.raises(ValueError):
        addestra(X, y, tr, te, peso_rendimento=True)                    # senza netto_bp si ferma


def test_pesi_rendimento_usano_solo_il_train():
    X, y, netto, tr, te = _dati_modello()
    a = addestra(X, y, tr, te, peso_rendimento=True, netto_bp=netto)
    netto2 = netto.copy()
    netto2.loc[te] = 1e6                                                # il test cambia: il modello no
    b = addestra(X, y, tr, te, peso_rendimento=True, netto_bp=netto2)
    assert np.array_equal(a.predict_proba(X)[:, 1], b.predict_proba(X)[:, 1])
