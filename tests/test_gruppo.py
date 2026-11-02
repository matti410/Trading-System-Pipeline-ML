"""
Scelta automatica del gruppo di trigger e di H (docs/04 §6, decisioni del 01/11/2026 in docs/05).

Barre inventate con costi a zero: con H = 1 il trigger sulla barra s entra all'Open di s+1 (100)
e esce all'Open di s+2, dove si mette il prezzo che decide il trade.
"""
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gruppo
import pipeline
from split import barra_confine, costruisci_X, split_train_test
from target import costruisci_target

COSTI_ZERO = {"spread": 0.0, "spread_passo_pips": 0.0, "commission": 0.0, "pip_size": 0.01,
              "spread_rollover_pips": None}
N = 400                 # train = 320 barre (quota 0,8), meta' alla barra 160


def _ds(aperture, trigger: dict):
    idx = pd.date_range("2024-01-01", periods=len(aperture), freq="15min", tz="UTC")
    o = pd.Series(aperture, index=idx, dtype=float)
    df = pd.DataFrame({"Open": o, "High": o, "Low": o, "Close": o, "Volume": 1.0})
    for nome, barre in trigger.items():
        df[nome] = False
        df.iloc[list(barre), df.columns.get_loc(nome)] = True
    return SimpleNamespace(df=df, df_is=df, costi=COSTI_ZERO, trigger_long=list(trigger), trigger_short=[],
                           solo_long=False, timeframe="M15", symbol="BINANCE_PERP_TEST", cartella=None)


def _tre_trigger(**altri):
    """
    Ogni 12 barre (k = 1..25): A a 12k, sempre +100 bp; B a 12k+3, +300 bp nella prima meta' e
    -100 bp nella seconda (media > 0, ma non nella seconda meta'); C a 12k+6, sempre -100 bp.
    """
    aperture = np.full(N, 100.0)
    k = np.arange(1, 26)
    A, B, C = 12 * k, 12 * k + 3, 12 * k + 6
    aperture[A + 2] = 101.0
    aperture[B + 2] = np.where(B < 160, 103.0, 99.0)
    aperture[C + 2] = 99.0
    trigger = {"A": A, "B": B, "C": C}
    for nome, (barre, prezzo) in altri.items():
        aperture[np.asarray(barre) + 2] = prezzo
        trigger[nome] = barre
    return _ds(aperture, trigger)


def test_tre_trigger_solo_A():
    p = gruppo.proponi(_tre_trigger(), 1, [1], min_per_meta=10, quarantena=False)
    amm = p.ammessi[1]
    assert amm.loc["A", "ammesso"] and not amm.loc["B", "ammesso"] and not amm.loc["C", "ammesso"]
    assert amm.loc["B", "media_netta_bp"] > 0 and amm.loc["B", "media_meta_2_bp"] < 0
    assert p.gruppo == ["A"] and p.H == 1
    assert list(p.composizione[1].index) == ["A"]
    assert p.per_H.loc[1, "occorrenze"] == 25
    assert p.per_H.loc[1, "media_netta_bp"] == pytest.approx(100.0)
    assert p.per_H.loc[1, "Totale_bp"] == pytest.approx(2500.0)       # 25 trade isolati, peso 1


def test_stesse_barre_non_aggiunge_niente():
    ds = _tre_trigger()
    ds.df["D"] = ds.df["A"]
    ds.trigger_long.append("D")
    for minimo in (10, 0):
        p = gruppo.proponi(ds, 1, [1], min_per_meta=minimo, quarantena=False)
        assert p.ammessi[1].loc["D", "ammesso"]
        assert p.gruppo == ["A"]
        assert p.composizione[1].loc["D", "nuove"] == 0
        assert p.composizione[1].loc["D", "esito"] != gruppo.ENTRA


def test_minimo_di_nuove_per_meta():
    # E positivo e stabile, ma con 5 occorrenze per meta': fuori con il minimo 10, dentro con 0
    k = np.r_[1:6, 14:19]
    ds = _tre_trigger(E=(12 * k + 9, 101.0))
    p10 = gruppo.proponi(ds, 1, [1], min_per_meta=10, quarantena=False)
    assert p10.gruppo == ["A"] and "sotto 10" in p10.composizione[1].loc["E", "esito"]
    p0 = gruppo.proponi(ds, 1, [1], min_per_meta=0, quarantena=False)
    assert set(p0.gruppo) == {"A", "E"}


def test_H_scelto_totale_piu_alto_a_parita_il_piu_corto():
    def per_H(righe):
        return pd.DataFrame(righe, columns=["H", "trigger_nel_gruppo", "Totale_bp"]).set_index("H")
    assert gruppo.scegli_H(per_H([(4, 1, 10.0), (8, 1, 30.0), (12, 2, 30.0), (16, 0, 0.0)])) == 8
    assert gruppo.scegli_H(per_H([(4, 1, 10.0), (8, 1, 20.0)])) == 8
    assert gruppo.scegli_H(per_H([(4, 0, 0.0), (8, 0, 0.0)])) is None      # nessun gruppo: lato vuoto
    assert gruppo.scegli_H(per_H([(4, 1, -5.0), (8, 0, 0.0)])) == 4        # solo gli H con un gruppo


def test_totale_non_cresce_con_la_sovrapposizione():
    """Prezzi a crescita costante, costi a zero, un trigger su 3 barre consecutive (due volte, una per meta')."""
    H = 4
    aperture = 100 * 1.001 ** np.arange(200)
    ds = _ds(aperture, {"T": [30, 31, 32, 110, 111, 112]})     # train = 160 barre, meta' a 80
    p = gruppo.proponi(ds, 1, [H], min_per_meta=0, quarantena=False)
    r = (1.001 ** H - 1) * 10_000                              # ogni trade: H barre di crescita
    riga = p.per_H.loc[H]
    assert riga["occorrenze"] == 6
    # ogni blocco copre H + 2 barre: ogni barra coperta conta una volta, r / H
    assert riga["effettive"] == pytest.approx(2 * (H + 2) / H)
    assert riga["Totale_bp"] == pytest.approx(2 * (H + 2) * r / H)
    assert riga["Totale_bp"] < 6 * r                           # la somma semplice conterebbe 6 trade


def test_griglia_non_valida():
    ds = _tre_trigger()
    for griglia in ([], [0, 4], [4, 4], [2.5], [True]):
        with pytest.raises(ValueError):
            gruppo.proponi(ds, 1, griglia, quarantena=False)


def test_spot_niente_short():
    ds = _tre_trigger()
    ds.solo_long = True
    with pytest.raises(ValueError, match="spot"):
        gruppo.proponi(ds, -1, [1], quarantena=False)


def _voci(cartella):
    return [json.loads(r) for f in sorted(cartella.rglob("*.jsonl")) for r in f.read_text(encoding="utf-8").splitlines()]


def test_scelta_a_mano_e_proposta_nel_registro(v4, tmp_path, capsys):
    ds = _tre_trigger()
    ds.cartella = str(tmp_path)
    p = gruppo.proponi(ds, 1, [1], quarantena=False)
    assert pipeline.scelta_lato(ds, p, None, None, 1, macchina="pc") == (["A"], 1, "proposta automatica")
    assert pipeline.scelta_lato(ds, p, ["B", "C"], 4, 1, collaudo=True, macchina="pc") == \
        (["B", "C"], 4, "scelta a mano")
    assert pipeline.scelta_lato(ds, p, [], None, 1, macchina="pc") == ([], None, "scelta a mano")
    uscita = capsys.readouterr().out
    assert "LONG: proposta automatica" in uscita and "LONG: scelta a mano" in uscita
    voci = _voci(tmp_path)
    assert [v["tipo"] for v in voci] == ["gruppo"] * 3
    assert [v["origine"] for v in voci] == ["proposta automatica", "scelta a mano", "scelta a mano"]
    assert voci[1]["proposta"] == {"trigger": ["A"], "H": 1}
    assert voci[1]["scelta"] == {"trigger": ["B", "C"], "H": 4} and voci[1]["collaudo"] is True
    assert voci[0]["griglia"] == [1] and voci[0]["min_nuove_per_meta"] == 10
    assert voci[0]["gruppi"] == {"1": ["A"]} and voci[0]["ammessi"] == {"1": ["A"]}
    # scritto solo uno dei due: si ferma
    for trigger, H in ((["A"], None), (None, 8)):
        with pytest.raises(ValueError, match="scrivi trigger e H insieme, oppure lasciali tutti e due a None"):
            pipeline.scelta_lato(ds, p, trigger, H, 1, macchina="pc")
    assert len(_voci(tmp_path)) == 3


def test_proposta_vuota_lato_saltato(v4, tmp_path, capsys):
    ds = _tre_trigger()
    ds.cartella = str(tmp_path)
    ds.trigger_long = ["B", "C"]                               # nessuno ammesso
    p = gruppo.proponi(ds, 1, [1], quarantena=False)
    assert p.H is None and p.gruppo == []
    assert pipeline.scelta_lato(ds, p, None, None, 1, macchina="pc") == ([], None, "proposta automatica")
    assert "nessun trigger ammesso: lato saltato" in capsys.readouterr().out
    assert pipeline.controlla_scelta(ds, [], None, 1) is False


def test_verifica_gruppo_avvisa_solo_quando_serve(capsys):
    idx = pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC")
    netto = pd.Series(np.r_[np.tile([20.0, 0.0], 10), np.tile([1.0, -1.0], 10)], index=idx)
    lato = SimpleNamespace(nome="LONG", target=pd.DataFrame({"netto_bp": netto}), idx_train=idx[:20],
                           idx_test=idx[20:])
    tab = pipeline.verifica_gruppo(lato)
    assert list(tab.index) == ["train", "test"] and list(tab.columns) == ["trigger", "media_netta_bp", "±bp"]
    assert tab.loc["train", "media_netta_bp"] == 10.0 and tab.loc["test", "media_netta_bp"] == 0.0
    assert pipeline.AVVISO_VERIFICA in capsys.readouterr().out          # 10 bp sotto, ± circa 4,6
    lato.target.loc[idx[20:], "netto_bp"] = np.tile([19.0, 1.0], 10)
    pipeline.verifica_gruppo(lato)
    assert pipeline.AVVISO_VERIFICA not in capsys.readouterr().out


# ---- dati veri (BINANCE_PERP_BTCUSDT M15): si saltano senza i dati del Lab ----

def _ds_ridotto(ds_vero, n=6):
    return dataclasses.replace(ds_vero, trigger_long=ds_vero.trigger_long[:n])


def test_netti_come_il_target(ds_vero):
    """I netti del train sono quelli di costruisci_target su tutto il df, e le uscite cadono nel train."""
    ds = _ds_ridotto(ds_vero)
    netti = gruppo.netti_train(ds, 1, 16)
    tutto = costruisci_target(ds_vero.df, ds.trigger_long, 1, 16, costi=ds_vero.costi)
    assert netti.index.isin(tutto.index).all()
    np.testing.assert_allclose(netti["netto_bp"], tutto.loc[netti.index, "netto_bp"], rtol=1e-12)
    confine = barra_confine(ds_vero.df_is)
    assert (ds_vero.df.index.get_indexer(netti.index) + 16 + 1 < confine).all()
    assert set(netti["meta"]) == {1, 2}


def test_solo_train(ds_vero):
    """Cambiando i prezzi del test (e dell'OOS), gruppo e H non cambiano."""
    ds = _ds_ridotto(ds_vero)
    p = gruppo.proponi(ds, 1, [8, 16])
    primo_test = ds_vero.df_is.index[barra_confine(ds_vero.df_is)]
    df_is, df = ds_vero.df_is.copy(), ds_vero.df.copy()
    for d in (df_is, df):
        dopo = d.index >= primo_test
        for c in ("Open", "High", "Low", "Close"):
            d.loc[dopo, c] = d.loc[dopo, c] * np.random.default_rng(1).uniform(0.5, 1.5, dopo.sum())
    p2 = gruppo.proponi(dataclasses.replace(ds, df_is=df_is, df=df), 1, [8, 16])
    pd.testing.assert_frame_equal(p.per_H, p2.per_H)
    assert (p.H, p.gruppo) == (p2.H, p2.gruppo)


def test_scelta_a_mano_nessun_numero_cambia(ds_vero, tmp_path, capsys):
    """
    Con la scelta a mano degli stessi trigger e H, il lato e' quello di sempre: stesso target,
    stessa divisione, stessa X, stessi pesi del modello. Cambia solo la riga «risoluzione».
    """
    ds = dataclasses.replace(_ds_ridotto(ds_vero), cartella=str(tmp_path))     # il registro va in tmp
    trig, H = ds_vero.trigger_long[:2], 16
    p = gruppo.proponi(ds, 1, [H])
    scelta, H_scelto, origine = pipeline.scelta_lato(ds, p, trig, H, 1, macchina="pc")
    assert (scelta, H_scelto, origine) == (trig, H, "scelta a mano")
    lato = pipeline.prepara_lato(ds_vero, 1, scelta, H_scelto)
    target = costruisci_target(ds_vero.df, trig, 1, H, costi=ds_vero.costi)
    pd.testing.assert_frame_equal(lato.target, target)
    tr, te = split_train_test(target, ds_vero.df_is, horizon=H)
    assert lato.idx_train.equals(tr) and lato.idx_test.equals(te)
    pd.testing.assert_frame_equal(lato.X, costruisci_X(ds_vero.df, target, trig, ds_vero.filtri))
    from pesi import pesi_unicita
    grezzi = pesi_unicita(tr, ds_vero.df, H, normalizza=False)
    pd.testing.assert_series_equal(pesi_unicita(tr, ds_vero.df, H), grezzi / grezzi.mean())
    uscita = capsys.readouterr().out
    assert "risoluzione:" in uscita and f"target: {len(target)} trigger" in uscita
