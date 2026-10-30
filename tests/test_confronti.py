"""I confronti fissi sul test (D3): il miglior filtro si sceglie sul solo train; estrazioni a seme fisso."""
import numpy as np
import pandas as pd

from confronti import confronti_test, miglior_filtro_train


def _dati(n=1000, seme=0):
    rng = np.random.default_rng(seme)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    tr, te = idx[:600], idx[600:]
    netto = pd.Series(rng.normal(0, 10, n), index=idx)
    df = pd.DataFrame({"F_BUONO_NEL_TRAIN": False, "F_BUONO_NEL_TEST": False, "F_RARO": False}, index=idx)
    df.loc[tr[::3], "F_BUONO_NEL_TRAIN"] = True
    netto.loc[tr[::3]] += 20
    df.loc[te[::3], "F_BUONO_NEL_TEST"] = True
    df.loc[tr[::3], "F_BUONO_NEL_TEST"] = True
    netto.loc[te[::3]] += 50                                  # vale solo nel test: non va scelto
    df.loc[tr[:50], "F_RARO"] = True
    netto.loc[tr[:50]] += 1000                                # altissimo ma < 100 trade: escluso
    return pd.DataFrame({"TARGET": (netto > 0).astype(int), "netto_bp": netto}), df, tr, te


def test_miglior_filtro_scelto_sul_train():
    target, df, tr, te = _dati()
    nome, tab = miglior_filtro_train(target, df, tr, list(df.columns), min_trade=100)
    assert nome == "F_BUONO_NEL_TRAIN"
    assert "F_RARO" not in set(tab["filtro"])
    # il test non entra nella scelta: cambiarlo non cambia il filtro
    target2 = target.copy()
    target2.loc[te, "netto_bp"] = -1e6
    assert miglior_filtro_train(target2, df, tr, list(df.columns), min_trade=100)[0] == nome


def test_estrazioni_a_caso_riproducibili():
    target, df, tr, te = _dati()
    sel = te[:20]
    a, pa = confronti_test(target, df, te, sel, "F_BUONO_NEL_TRAIN", n_estrazioni=200, seme=0)
    b, pb = confronti_test(target, df, te, sel, "F_BUONO_NEL_TRAIN", n_estrazioni=200, seme=0)
    pd.testing.assert_frame_equal(a, b)
    assert pa == pb and 0 <= pa <= 100
    assert a.loc["20% a caso: media di 200 estrazioni", "trade"] == round(0.2 * len(te))
    assert a.loc["tutti i trigger del test", "trade"] == len(te)
