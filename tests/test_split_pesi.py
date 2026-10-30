"""Purge, embargo e pesi con l'uscita all'Open della barra trigger + H + 1."""
import pandas as pd
import pytest

from pesi import pesi_unicita
from split import split_train_test


def _barre(n):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"Open": 1.0, "Close": 1.0}, index=idx)


def test_purge_ed_embargo():
    df = _barre(120)
    df_is = df.iloc[:100]                       # confine train/test alla barra 80
    H = 5
    pos = [10, 73, 74, 80, 93, 94, 110]
    target = pd.DataFrame({"TARGET": [1, 0, 1, 0, 1, 0, 1]}, index=df.index[pos])
    tr, te = split_train_test(target, df_is, horizon=H, quota=0.8)
    # 73: uscita alla barra 79 (< 80) -> train; 74: uscita alla 80 -> embargo
    assert list(df.index.get_indexer(tr)) == [10, 73]
    # 93: uscita alla 99 (< 100) -> test; 94: uscita alla 100, fuori dall'IS -> purge; 110: OOS
    assert list(df.index.get_indexer(te)) == [80, 93]


def test_horizon_obbligatorio():
    df = _barre(50)
    target = pd.DataFrame({"TARGET": [1]}, index=df.index[[5]])
    with pytest.raises(TypeError):
        split_train_test(target, df)            # noqa: horizon mancante


def test_pesi_finestre_sovrapposte():
    df = _barre(40)
    p = pesi_unicita(df.index[[2, 3, 20]], df, horizon=4)
    # 2 e 3 condividono 3 barre su 4 (peso 0,625 prima della media 1), 20 e' isolato (1)
    assert p.iloc[0] == pytest.approx(p.iloc[1])
    assert p.iloc[0] / p.iloc[2] == pytest.approx(0.625)
    assert p.mean() == pytest.approx(1.0)


def test_pesi_rifiutano_trigger_senza_uscita():
    df = _barre(10)
    with pytest.raises(ValueError):
        pesi_unicita(df.index[[5]], df, horizon=4)      # serve la barra 5 + 4 + 1 = 10, che non c'e'
