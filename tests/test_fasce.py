"""
Decisione di Mattia sul §I: fasce solo se le probabilita' del test hanno abbastanza valori distinti;
un lato senza feature tenute mostra ai passi 8 e 9 solo «tutti i trigger», senza fasce ne' soglia.
"""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oos import FasceNonCalcolabili, controlla_fasce, soglia_dal_test, valuta_oos


class Costante:
    """Un modello che da' a tutti la stessa probabilita' (o pochi valori)."""

    def __init__(self, valori=(0.4,)):
        self.valori = np.asarray(valori)

    def predict_proba(self, X):
        p = self.valori[np.arange(len(X)) % len(self.valori)]
        return np.column_stack([1 - p, p])


def _dati(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.integers(0, 2, n), index=idx)
    rend = pd.Series(rng.normal(0, 0.001, n), index=idx)
    return X, y, rend, idx[:200], idx[200:]


def test_controlla_fasce():
    controlla_fasce(np.arange(5) / 10, 5)                             # 5 valori distinti: va
    with pytest.raises(FasceNonCalcolabili, match="1 valore distinto"):
        controlla_fasce(np.full(100, 0.3), 5)
    with pytest.raises(FasceNonCalcolabili, match="4 valori distinti"):
        controlla_fasce(np.tile([0.1, 0.2, 0.3, 0.4], 25), 5)


def test_soglia_e_oos_senza_fasce(capsys):
    X, y, rend, te, oo = _dati()
    with pytest.raises(FasceNonCalcolabili):
        soglia_dal_test(Costante(), X, te)
    tab = valuta_oos(Costante(), X, y, rend, te, oo)
    assert list(tab.index) == ["test tutti", "OOS tutti"]               # niente «fascia 5» = 100%
    assert "fasce e soglia non si calcolano" in capsys.readouterr().out
    tab = valuta_oos(Costante(np.linspace(0.1, 0.9, 50)), X, y, rend, te, oo)
    assert "test fascia 5" in tab.index                                # con abbastanza valori si calcola


def test_valuta_senza_fasce(capsys):
    from modello import valuta
    X, y, rend, tr, te = _dati()
    tab = valuta(Costante(), X, y, rend, tr, te)
    assert list(tab.index) == ["tutti"]
    assert "fasce del test non calcolate" in capsys.readouterr().out


def test_confronti_e_backtest_senza_fasce(v4):
    from backtest_oos import trigger_in_fascia
    from confronti import selezione_test
    X, y, rend, te, oo = _dati()
    with pytest.raises(FasceNonCalcolabili):
        selezione_test(Costante(), X, te)
    with pytest.raises(FasceNonCalcolabili):
        trigger_in_fascia(Costante(), X, te, oo)


def test_lato_senza_feature_passi_8_e_9(ds_vero, tmp_path, capsys):
    import pipeline
    ds = dataclasses.replace(ds_vero, cartella=str(tmp_path))
    trig = ds.trigger_short[:1]
    lato = pipeline.prepara_lato(ds, -1, trig, 32)
    # un lato con la selezione che non ha tenuto nessuna feature: X = sola identita' del trigger
    lato_sel = dataclasses.replace(lato, X=lato.X[trig], extra={"selezione": SimpleNamespace(tenute=[])})
    assert pipeline.nessuna_feature(lato_sel) and not pipeline.nessuna_feature(lato)
    modello, _ = pipeline.allena(ds, lato_sel, collaudo=True)
    capsys.readouterr()
    frase = "SHORT: la selezione non ha tenuto nessuna feature, il modello non distingue i trigger"
    assert pipeline.mostra_soglia(lato_sel, modello) is None
    tab, perc = pipeline.confronti(ds, lato_sel, modello)
    assert list(tab.index) == ["tutti i trigger del test"] and np.isnan(perc)
    oos = pipeline.guarda_oos(ds, lato_sel, modello, collaudo=True)
    assert list(oos.index) == ["test tutti", "OOS tutti"]
    tab_bt, ris = pipeline.backtest(ds, {-1: lato_sel}, {-1: modello}, collaudo=True)
    assert set(tab_bt.index.get_level_values("scenario")) == {"tutti i trigger"}
    assert "modello" not in ris
    uscita = capsys.readouterr().out
    assert uscita.count(frase) >= 4                                    # soglia, confronti, OOS, backtest
    assert "fascia 5" not in uscita


def test_backtest_un_lato_senza_feature(ds_vero, tmp_path):
    """Un lato con il modello e uno senza: il lato senza ha solo «tutti i trigger», e il COMBINATO «modello» non c'e'."""
    import pipeline
    ds = dataclasses.replace(ds_vero, cartella=str(tmp_path))
    lato_l = pipeline.prepara_lato(ds, 1, ds.trigger_long[:1], 16)
    lato_s = pipeline.prepara_lato(ds, -1, ds.trigger_short[:1], 32)
    lato_s = dataclasses.replace(lato_s, X=lato_s.X[ds.trigger_short[:1]],
                                 extra={"selezione": SimpleNamespace(tenute=[])})
    m_l, _ = pipeline.allena(ds, lato_l, collaudo=True)
    m_s, _ = pipeline.allena(ds, lato_s, collaudo=True)
    tab, ris = pipeline.backtest(ds, {1: lato_l, -1: lato_s}, {1: m_l, -1: m_s}, collaudo=True)
    righe = set(tab.index)
    assert ("LONG", "modello") in righe and ("SHORT", "modello") not in righe
    assert ("COMBINATO", "modello") not in righe and ("COMBINATO", "tutti i trigger") in righe
    assert len(ris["modello"]["trade"]["SHORT"]) == 0 and len(ris["modello"]["trade"]["COMBINATO"]) == 0


@pytest.mark.parametrize("d", [1, -1])
def test_un_lato_solo_passi_8_e_9(ds_vero, tmp_path, d, monkeypatch):
    """Un lato solo (l'altro lasciato vuoto nella cella 4), fuori dallo spot: soglia, confronti, OOS e backtest."""
    import pipeline
    ds = dataclasses.replace(ds_vero, cartella=str(tmp_path))
    trig = (ds.trigger_long if d == 1 else ds.trigger_short)[:1]
    lato = pipeline.prepara_lato(ds, d, trig, 16)
    modello, _ = pipeline.allena(ds, lato, collaudo=True)
    pipeline.mostra_soglia(lato, modello)
    pipeline.confronti(ds, lato, modello)
    pipeline.guarda_oos(ds, lato, modello, collaudo=True)
    tab, ris = pipeline.backtest(ds, {d: lato}, {d: modello}, collaudo=True)
    altro = "SHORT" if d == 1 else "LONG"
    for scenario in ("modello", "tutti i trigger"):
        assert tab.loc[(altro, scenario), "trade"] == 0
        assert tab.loc[("COMBINATO", scenario), "trade"] == tab.loc[(lato.nome, scenario), "trade"] > 0
    from backtest_oos import grafico_equity
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    grafico_equity(ris)
    plt.close("all")
