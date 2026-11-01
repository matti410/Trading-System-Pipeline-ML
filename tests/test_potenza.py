"""
Test di potenza della selezione (collaudo dello strumento, decisione di Mattia sul §I).

Dataset inventati come quelli del §F: 5.000 trigger, 1 feature vera e 20 di rumore. La forza
della vera e' il coefficiente FORZA nella probabilita' del target, P(y=1) = 1 / (1 + e^(-FORZA x vera)),
con la vera normale standard. Gradini e semi fissati PRIMA di guardare i risultati.
Per ogni gradino: in quanti semi la vera viene tenuta e l'AUC della sola vera sui blocchi di
validazione (media dei 4 giri). La tabella e' in docs/02 §I. Si vede con: pytest -s tests/test_potenza.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import selezione as S

pytestmark = pytest.mark.lento          # circa 3,5 minuti: python -m pytest -q -m lento

GRADINI = (0.50, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.00)
SEMI = (0, 1, 2)
N_TRIGGER = 5000
N_RUMORE = 20
H = 4


def caso(forza, seme, n=N_TRIGGER, rumore=N_RUMORE):
    rng = np.random.default_rng([seme, int(round(forza * 1000))])
    idx = pd.date_range("2023-01-01", periods=n * 3, freq="15min", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "Close": 1.0}, index=idx)
    ti = idx[np.sort(rng.choice(n * 3 - 10, n, replace=False))]
    X = pd.DataFrame({f"rumore_{i}": rng.normal(size=n) for i in range(1, rumore + 1)}, index=ti)
    X.insert(0, "vera", rng.normal(size=n))
    X.insert(0, "E1_TRIG", 1)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-forza * X["vera"]))).astype(int), index=ti)
    return X, y, df, ti


def potenza():
    righe = []
    for forza in GRADINI:
        tenuta, auc, rumore_passato = [], [], 0
        for seme in SEMI:
            X, y, df, ti = caso(forza, seme)
            sel = S.selezione(X, y, df, ti, df.index[-1], H, sempre=["E1_TRIG"], con_unicita=False, peso_rendimento=False)
            tenuta.append("vera" in sel.tenute)
            rumore_passato += sum(t.startswith("rumore_") for t in sel.tenute)
            giri = S.blocchi_interni(ti, df, H, df.index[-1])
            auc.append(np.mean([roc_auc_score(y.loc[v], X.loc[v, "vera"]) for _, v in giri]))
        righe.append({"forza": forza, "vera_tenuta": f"{sum(tenuta)} su {len(SEMI)}",
                      "auc_vera_validazione": round(float(np.mean(auc)), 3), "rumore_passato": rumore_passato})
    return pd.DataFrame(righe).set_index("forza")


@pytest.fixture(scope="module")
def tabella():
    t = potenza()
    print("\n" + t.to_string())
    return t


def test_potenza(tabella):
    assert list(tabella.index) == list(GRADINI)
    assert tabella.loc[GRADINI[0], "vera_tenuta"] == f"{len(SEMI)} su {len(SEMI)}"     # la piu' forte si vede
    assert tabella.loc[0.0, "vera_tenuta"] == f"0 su {len(SEMI)}"                     # forza zero: e' rumore
    assert abs(tabella.loc[0.0, "auc_vera_validazione"] - 0.5) < 0.03
    # l'AUC della sola vera scende con la forza
    assert tabella["auc_vera_validazione"].is_monotonic_decreasing or \
        np.all(np.diff(tabella["auc_vera_validazione"].to_numpy()) < 0.01)
