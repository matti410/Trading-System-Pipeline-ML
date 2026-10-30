"""Event study solo sul train, controllo della scelta di Mattia, split con H diversi per lato."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pipeline
from split import barra_confine, split_train_test


def test_event_study_del_train_non_vede_il_test(ds_vero, monkeypatch):
    """Le barre passate all'event study finiscono prima del test; cambiare il test non cambia niente."""
    import engine.event_study as es
    visti = []
    originale = es.run_event_study

    def spia(df, **kw):
        visti.append(df)
        return originale(df, **kw)
    monkeypatch.setattr(es, "run_event_study", spia)

    trigger = ds_vero.trigger_long[:3]
    ev = pipeline.event_study_train(ds_vero, 1, 32, trigger=trigger)
    primo_test = ds_vero.df_is.index[barra_confine(ds_vero.df_is)]
    assert visti[0].index[-1] < primo_test
    assert len(visti[0]) == barra_confine(ds_vero.df_is)

    # stessi risultati con i prezzi del test (e dell'OOS) stravolti
    df_is = ds_vero.df_is.copy()
    dopo = df_is.index >= primo_test
    for c in ("Open", "High", "Low", "Close"):
        df_is.loc[dopo, c] = df_is.loc[dopo, c] * np.random.default_rng(1).uniform(0.5, 1.5, dopo.sum())
    ev2 = pipeline.event_study_train(dataclasses.replace(ds_vero, df_is=df_is), 1, 32, trigger=trigger)
    pd.testing.assert_frame_equal(ev.sintesi, ev2.sintesi)


def test_event_study_short_rifiutato_sullo_spot():
    with pytest.raises(ValueError):
        pipeline.event_study_train(SimpleNamespace(solo_long=True), -1, 16)


def _ds_finto(solo_long=False):
    return SimpleNamespace(solo_long=solo_long, timeframe="M15", trigger_long=["E1_A", "E2_B"],
                           trigger_short=[] if solo_long else ["E1_SHORT_A"])


def test_controlla_scelta(v4):
    ds = _ds_finto()
    assert pipeline.controlla_scelta(ds, ["E1_A"], 16, 1) is True
    assert pipeline.controlla_scelta(ds, ["E1_SHORT_A"], 32, -1) is True
    with pytest.raises(ValueError, match="TRIGGER_LONG"):
        pipeline.controlla_scelta(ds, [], 16, 1)                 # scelta mancante: si ferma
    with pytest.raises(ValueError, match="TRIGGER_SHORT"):
        pipeline.controlla_scelta(ds, ["E1_SHORT_A"], None, -1)
    with pytest.raises(ValueError, match="Non sono trigger"):
        pipeline.controlla_scelta(ds, ["E1_SHORT_A"], 16, 1)     # trigger dell'altro lato
    for H in (0, 2.5, True, "16"):
        with pytest.raises(ValueError):
            pipeline.controlla_scelta(ds, ["E1_A"], H, 1)


def test_spot_short(v4):
    spot = _ds_finto(solo_long=True)
    assert pipeline.controlla_scelta(spot, [], None, -1) is False  # saltato
    with pytest.raises(ValueError, match="spot"):
        pipeline.controlla_scelta(spot, ["E1_SHORT_A"], 16, -1)


def test_ore(v4):
    assert pipeline.ore(16, "M15") == "4 ore"
    assert pipeline.ore(4, "M15") == "1 ora"
    assert pipeline.orizzonte_event_study("M15") == 32            # 8 ore, la griglia del Lab
    assert pipeline.orizzonte_event_study("H1") == 8


def test_split_con_H_diversi_per_lato():
    idx = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
    df_is = pd.DataFrame({"Open": 1.0}, index=idx[:150])          # confine train/test alla barra 120
    pos = [100, 108, 112, 115, 125, 135, 140, 145]
    target = pd.DataFrame({"TARGET": [0, 1] * 4}, index=idx[pos])
    tr4, te4 = split_train_test(target, df_is, horizon=4)
    tr12, te12 = split_train_test(target, df_is, horizon=12)
    # H=4: embargo per uscita >= 120 -> via 115 (uscita 120); purge per uscita >= 150 -> via 145
    assert list(df_is.index.get_indexer(tr4)) == [100, 108, 112]
    assert list(df_is.index.get_indexer(te4)) == [125, 135, 140]
    # H=12: embargo via 108 (uscita 121), 112, 115; purge via 140 (uscita 153) e 145
    assert list(df_is.index.get_indexer(tr12)) == [100]
    assert list(df_is.index.get_indexer(te12)) == [125, 135]
