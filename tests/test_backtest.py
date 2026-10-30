"""
Backtest sul motore veloce di V4: i trade hanno gli stessi bp netti delle righe del
target (identita' esatta), un trade alla volta per lato, COMBINATO = unione dei lati.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_oos import _esposizione, esegui_backtest, tabella_metriche
from target import costruisci_target
from test_identita_motore import _mercato_inventato

COSTI = {"spread": 0.0, "spread_passo_pips": 0.1, "commission": 0.0005, "pip_size": 1.0,
         "spread_rollover_pips": None}


def _identita(bt, target_l, target_s):
    for lato, t in (("LONG", target_l), ("SHORT", target_s)):
        tr = bt["trade"][lato]
        assert len(tr) > 10
        riga = t.loc[pd.DatetimeIndex(tr["trigger"])]
        assert np.array_equal(riga["netto_bp"].to_numpy(), tr["netto_bp"].to_numpy())
        assert np.array_equal(riga["uscita"].to_numpy(), tr["ExitPrice"].to_numpy())


def test_identita_backtest_target_inventato(v4):
    df, funding = _mercato_inventato()
    costi = {**COSTI, "funding": funding}
    tl = costruisci_target(df, ["E_L"], 1, 16, costi=costi, quarantena=False)
    ts = costruisci_target(df, ["E_S"], -1, 32, costi=costi, quarantena=False)
    oos = df.index[3000]
    bt = esegui_backtest(df, tl.index[tl.index >= oos], ts.index[ts.index >= oos], 16, 32, costi)
    _identita(bt, tl, ts)
    # nessun trade prima dei segnali passati: il periodo e' solo quello dei segnali
    assert (pd.DatetimeIndex(bt["trade"]["COMBINATO"]["trigger"]) >= oos).all()


def test_identita_backtest_target_dati_veri(ds_vero):
    tl = costruisci_target(ds_vero.df, ds_vero.trigger_long[:1], 1, 16, costi=ds_vero.costi)
    ts = costruisci_target(ds_vero.df, ds_vero.trigger_short[:1], -1, 32, costi=ds_vero.costi)
    fine_is = ds_vero.df_is.index[-1]
    bt = esegui_backtest(ds_vero.df, tl.index[tl.index > fine_is], ts.index[ts.index > fine_is], 16, 32,
                         ds_vero.costi)
    _identita(bt, tl, ts)


def test_un_trade_alla_volta_e_combinato(v4):
    idx = pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC")
    o = pd.Series(100.0 + np.arange(40), index=idx)
    df = pd.DataFrame({"Open": o, "High": o + 1, "Low": o - 1, "Close": o + 0.5, "Volume": 1.0})
    # long H=4: segnali a 2, 4 (trade di 2 aperto: saltato), 10; short H=3: 3, 20
    bt = esegui_backtest(df, idx[[2, 4, 10]], idx[[3, 20]], 4, 3, COSTI)
    assert list(bt["trade"]["LONG"]["EntryBar"]) == [3, 11]
    assert list(bt["trade"]["LONG"]["ExitBar"]) == [7, 15]
    assert bt["saltati"] == {"LONG": 1, "SHORT": 0, "COMBINATO": 1}
    comb = bt["trade"]["COMBINATO"]
    # in ordine d'uscita: long 3->7, short 4->7, long 11->15, short 21->24
    assert list(comb["lato"]) == ["LONG", "SHORT", "LONG", "SHORT"]
    assert list(comb["ExitBar"]) == [7, 7, 15, 24]
    # esposizione nelle barre 0..40: long aperto 3-6 e 11-14, short 4-6 e 21-23 -> barre 3-6, 11-14, 21-23 = 11
    assert _esposizione(comb, 0, 40) == 11 / 40
    tab = tabella_metriche({"prova": bt}, 0, 40)
    assert tab.loc[("COMBINATO", "prova"), "trade"] == 4
    assert tab.loc[("COMBINATO", "prova"), "totale_bp"] == round(float(comb["netto_bp"].sum()), 2)


def test_spot_senza_short(v4):
    df, _ = _mercato_inventato(n=800)
    tl = costruisci_target(df, ["E_L"], 1, 8, costi=COSTI, quarantena=False)
    bt = esegui_backtest(df, tl.index, None, 8, None, COSTI)
    assert len(bt["trade"]["SHORT"]) == 0
    assert len(bt["trade"]["COMBINATO"]) == len(bt["trade"]["LONG"])


def test_vectorbt_tolto():
    radice = Path(__file__).resolve().parent.parent
    assert "vectorbt" not in (radice / "requirements.txt").read_text(encoding="utf-8")
    assert "vectorbt" not in sys.modules
    for f in radice.glob("*.py"):
        assert "vectorbt" not in f.read_text(encoding="utf-8"), f.name
