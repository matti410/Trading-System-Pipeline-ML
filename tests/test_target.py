"""
Il target su barre inventate: etichette e bp netti calcolati a mano.

Regola: trigger sulla barra s, ingresso all'Open di s+1, uscita all'Open di
s+H+1. Costo in pips = commissione * (ingresso + uscita) / pip + spread di un
passo (+ funding pagato); bp = pips * pip * 10.000 / ingresso.
"""
import pandas as pd
import pytest

from target import costruisci_target

APERTURE = [100, 100, 101, 102, 103, 99, 100, 100, 100, 100, 100, 100]
COSTI = {"spread": 0.0, "spread_passo_pips": 1.0, "commission": 0.0005, "pip_size": 0.1,
         "spread_rollover_pips": None}


def barre(aperture, **trigger):
    idx = pd.date_range("2024-01-01", periods=len(aperture), freq="15min", tz="UTC")
    o = pd.Series(aperture, index=idx, dtype=float)
    df = pd.DataFrame({"Open": o, "High": o + 1, "Low": o - 1, "Close": o, "Volume": 1.0})
    for nome, barre_vere in trigger.items():
        df[nome] = False
        df.iloc[list(barre_vere), df.columns.get_loc(nome)] = True
    return df


def test_long_etichette_e_bp_a_mano():
    # trigger long alle barre 0 (esclusa: il motore non la guarda), 1, 3, 4, 8,
    # 9 (esclusa: manca la barra 9+2+1 = 12)
    df = barre(APERTURE, E1=[0, 1, 3, 4, 8, 9])
    t = costruisci_target(df, ["E1"], 1, 2, costi=COSTI, quarantena=False)
    assert list(df.index.get_indexer(t.index)) == [1, 3, 4, 8]
    # s=1: 101 -> 103. lordo 20 pips; commissione 0,0005*204/0,1 = 1,02; spread 1 -> netto 17,98 pips
    # s=3: 103 -> 100. lordo -30 pips; commissione 1,015; spread 1 -> netto -32,015 pips
    # s=4: 99 -> 100. lordo 10 pips; commissione 0,995; spread 1 -> netto 8,005 pips
    # s=8: 100 -> 100. lordo 0; commissione 1,0; spread 1 -> netto -2 pips
    attesi_netti = [17.98 * 1000 / 101, -32.015 * 1000 / 103, 8.005 * 1000 / 99, -2.0 * 1000 / 100]
    attesi_lordi = [20 * 1000 / 101, -30 * 1000 / 103, 10 * 1000 / 99, 0.0]
    assert t["TARGET"].tolist() == [1, 0, 1, 0]
    assert t["netto_bp"].tolist() == pytest.approx(attesi_netti, abs=1e-9)
    assert t["lordo_bp"].tolist() == pytest.approx(attesi_lordi, abs=1e-9)
    assert t["ingresso"].tolist() == [101, 103, 99, 100]
    assert t["uscita"].tolist() == [103, 100, 100, 100]


def test_short_a_mano():
    df = barre(APERTURE, S1=[2])
    t = costruisci_target(df, ["S1"], -1, 2, costi=COSTI, quarantena=False)
    # s=2: vende a 102, ricompra a 99. lordo 30 pips; commissione 0,0005*201/0,1 = 1,005; netto 27,995
    assert t["TARGET"].tolist() == [1]
    assert t["netto_bp"].iloc[0] == pytest.approx(27.995 * 1000 / 102, abs=1e-9)


def test_pareggio_esatto_vale_zero():
    # pip 0,25 e prezzi rappresentabili esatti: lordo 1 pip, spread 1 pip, commissione 0 -> netto 0
    costi = {**COSTI, "commission": 0.0, "pip_size": 0.25}
    df = barre([100, 100, 100, 100, 100.25, 100], E1=[1])
    t = costruisci_target(df, ["E1"], 1, 2, costi=costi, quarantena=False)
    assert t["netto_bp"].iloc[0] == 0.0
    assert t["TARGET"].iloc[0] == 0


def test_funding_pagato_fra_ingresso_e_uscita():
    # liquidazioni all'inizio delle barre 2 (= ingresso, non si paga), 3 e 4 (= uscita, si paga)
    df = barre(APERTURE, E1=[1])
    funding = pd.DataFrame({"tasso": [0.001, 0.001, -0.002], "intervallo_ore": 8.0},
                           index=df.index[[2, 3, 4]])
    t = costruisci_target(df, ["E1"], 1, 2, costi={**COSTI, "funding": funding}, quarantena=False)
    # pagato: 0,001*102/0,1 = 1,02 pips; incassato: 0,002*103/0,1 = 2,06 pips -> funding -1,04
    netto_pips = 17.98 - 1.02 + 2.06
    assert t["netto_bp"].iloc[0] == pytest.approx(netto_pips * 1000 / 101, abs=1e-9)


def test_quarantena_scarta_segnale_e_ingresso():
    df = barre(APERTURE, E1=[1, 3, 4, 8])
    q = pd.Series(False, index=df.index)
    q.iloc[4] = True            # barra 4 in quarantena: via s=4 (segnale) e s=3 (ingresso)
    t = costruisci_target(df, ["E1"], 1, 2, costi=COSTI, quarantena=q)
    assert list(df.index.get_indexer(t.index)) == [1, 8]


def test_direzione_e_horizon_controllati():
    df = barre(APERTURE, E1=[1])
    with pytest.raises(ValueError):
        costruisci_target(df, ["E1"], 0, 2, costi=COSTI, quarantena=False)
    with pytest.raises(ValueError):
        costruisci_target(df, ["E1"], 1, 0, costi=COSTI, quarantena=False)
    with pytest.raises(KeyError):
        costruisci_target(df, ["NON_C_E"], 1, 2, costi=COSTI, quarantena=False)
