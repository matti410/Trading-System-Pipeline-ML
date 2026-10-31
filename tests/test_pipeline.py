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
    with pytest.raises(ValueError, match="H_SHORT"):
        pipeline.controlla_scelta(ds, ["E1_SHORT_A"], None, -1)  # trigger senza H: si ferma
    with pytest.raises(ValueError, match="H_LONG"):
        pipeline.controlla_scelta(ds, ["E1_A"], None, 1)
    with pytest.raises(ValueError, match="Non sono trigger"):
        pipeline.controlla_scelta(ds, ["E1_SHORT_A"], 16, 1)     # trigger dell'altro lato
    for H in (0, 2.5, True, "16"):
        with pytest.raises(ValueError):
            pipeline.controlla_scelta(ds, ["E1_A"], H, 1)


def test_lato_vuoto_saltato(v4, capsys):
    ds = _ds_finto()
    for trigger in (None, []):
        for H in (None, 16, 0, "x"):                              # qualunque sia H
            assert pipeline.controlla_scelta(ds, trigger, H, 1) is False
            assert pipeline.controlla_scelta(ds, trigger, H, -1) is False
    uscita = capsys.readouterr().out
    assert "LONG: nessun trigger scelto, lato saltato" in uscita
    assert "SHORT: nessun trigger scelto, lato saltato" in uscita


def test_lati_scelti(v4, capsys):
    ds = _ds_finto()
    assert pipeline.lati_scelti(ds, {1: (["E1_A"], 16), -1: ([], None)}, {}) == [1]           # SHORT vuoto
    assert pipeline.lati_scelti(ds, {1: ([], None), -1: (["E1_SHORT_A"], 32)}, {}) == [-1]    # LONG vuoto
    assert pipeline.lati_scelti(ds, {1: (["E1_A"], 16), -1: (["E1_SHORT_A"], 32)}, {}) == [1, -1]
    with pytest.raises(ValueError, match="Scegli i trigger di almeno un lato"):
        pipeline.lati_scelti(ds, {1: ([], None), -1: (None, 8)}, {})
    with pytest.raises(ValueError, match="H_LONG"):
        pipeline.lati_scelti(ds, {1: (["E1_A"], None), -1: ([], None)}, {})                   # trigger senza H
    spot = _ds_finto(solo_long=True)
    assert pipeline.lati_scelti(spot, {1: (["E1_A"], 16), -1: ([], None)}, {}) == [1]
    with pytest.raises(ValueError, match="Scegli i trigger di almeno un lato"):
        pipeline.lati_scelti(spot, {1: ([], None), -1: ([], None)}, {})


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


def _con_trigger_finti(ds):
    """Due trigger LONG registrati solo per il test: uno raro (37 nel train) e uno che non scatta mai."""
    from engine import registry

    def raro(df):
        s = pd.Series(False, index=df.index)
        s.iloc[5000::4000] = True
        return s
    registry.register_entry("E98_MAI_TEST", 1)(lambda df: pd.Series(False, index=df.index))
    registry.register_entry("E99_RARO_TEST", 1)(raro)
    df = ds.df.assign(E99_RARO_TEST=raro(ds.df), E98_MAI_TEST=False)
    return dataclasses.replace(ds, df=df, df_is=df.loc[ds.df_is.index],
                               trigger_long=ds.trigger_long + ["E99_RARO_TEST", "E98_MAI_TEST"])


def test_trigger_raro_visibile_e_avviso(ds_vero, capsys, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from engine import registry
    from engine.vetrina import MIN_TRIGGER
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    ds = _con_trigger_finti(ds_vero)
    try:
        ev = pipeline.event_study_train(ds, 1, 32, trigger=[ds.trigger_long[0], "E99_RARO_TEST", "E98_MAI_TEST"])
        s = pipeline.sintesi_event_study(ev, "M15")
        segno = s.set_index("candidato")[f"sotto_minimo_Lab_{MIN_TRIGGER}"]
        n = pipeline.occorrenze_train(ev)
        assert 0 < n["E99_RARO_TEST"] < MIN_TRIGGER and n["E98_MAI_TEST"] == 0
        assert segno["E99_RARO_TEST"].startswith("◄") and segno["E98_MAI_TEST"].startswith("◄")
        assert segno[ds.trigger_long[0]] == ""
        etichette = pipeline.grafico_event_study(ev)
        legenda = [t.get_text() for t in plt.gcf().axes[0].get_legend().get_texts()]
        assert f"E99_RARO_TEST ({n['E99_RARO_TEST']} nel train)" in legenda          # nel grafico, con il numero
        assert any(e.startswith("E98_MAI_TEST") for e in etichette)
        assert "E98_MAI_TEST" in capsys.readouterr().out                             # senza curva, ma nominato
        plt.close("all")
        # controlla_scelta avvisa e non si ferma
        assert pipeline.controlla_scelta(ds, ["E99_RARO_TEST"], 16, 1, occorrenze=n) is True
        uscita = capsys.readouterr().out
        assert "!!! AVVISO" in uscita and f"E99_RARO_TEST: {n['E99_RARO_TEST']} occorrenze" in uscita
        assert pipeline.controlla_scelta(ds, [ds.trigger_long[0]], 16, 1, occorrenze=n) is True
        assert "AVVISO" not in capsys.readouterr().out
    finally:
        registry.remove_entries("E98_MAI_TEST")
        registry.remove_entries("E99_RARO_TEST")


def test_event_study_senza_minimo():
    assert pipeline.MIN_TRADES_EVENT_STUDY == 0


def _ev_inventato():
    """Un EventStudy di V4 con una sintesi inventata: 13 trigger con curva e uno senza occorrenze."""
    from engine.event_study import EventStudy
    picchi = {"A_OLTRE_1": (0.30, True), "B_OLTRE_2": (0.10, True), "C_OLTRE_3": (0.50, True),
              "D_OLTRE_NEG": (-0.40, True), "E_SOTTO_1": (0.90, False), "F_SOTTO_2": (0.05, False),
              "G_SOTTO_NEG": (-0.10, False), "H_ZERO": (0.0, False)}
    picchi.update({f"R_SOTTO_{i}": (0.01 * i, False) for i in range(1, 6)})
    nomi = list(picchi)
    sintesi = pd.DataFrame({"candidato": nomi, "direction": 1, "trades": [300] * len(nomi), "barra_picco": 4,
                            "picco_pct": [p for p, _ in picchi.values()], "incertezza_pct": 0.1,
                            "volte_incertezza": 1.0, "oltre_rumore": [o for _, o in picchi.values()],
                            "a_fine_pct": 0.0, "vs_mercato_pct": 0.0})
    curve = pd.DataFrame({n: np.linspace(0, p, 9) for n, (p, _) in picchi.items()})
    ev = EventStudy(curve, curve * 0, curve, curve, curve * 0 + 0.1, sintesi, 8, (10, 90), 1.0, 1.0, [])
    ev.candidati_richiesti = nomi + ["Z_MAI"]
    return ev


def test_passo3_solo_i_migliori(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    ev = _ev_inventato()
    tutta = pipeline.sintesi_event_study(ev, "M15")
    assert len(tutta) == 14                                                  # la sintesi completa resta intera
    vista, conti = pipeline.sintesi_da_mostrare(tutta, 6)
    # oltre il rumore per picco decrescente, poi gli altri per picco decrescente; picchi <= 0 esclusi
    assert list(vista["candidato"]) == ["C_OLTRE_3", "A_OLTRE_1", "B_OLTRE_2", "E_SOTTO_1", "F_SOTTO_2", "R_SOTTO_5"]
    assert list(vista.columns) == list(tutta.columns)                        # le colonne di oggi, tutte
    assert conti == {"mostrati": 6, "totale": 14, "picco_non_positivo": 3, "oltre_i_primi": 4, "senza_curva": 1}
    assert len(pipeline.sintesi_da_mostrare(tutta, 10)[0]) == 10
    assert len(pipeline.sintesi_da_mostrare(tutta, 99)[0]) == 10            # solo i 10 con picco > 0
    # il grafico contiene solo quei nomi, piu' il mercato
    pipeline.grafico_event_study(ev, candidati=list(vista["candidato"]))
    legenda = [t.get_text() for t in plt.gcf().axes[0].get_legend().get_texts()]
    assert legenda == [f"{c} (300 nel train)" for c in vista["candidato"]] + ["mercato (tutte le barre)"]
    plt.close("all")
    riga = pipeline.riga_mostrati(conti, 6)
    assert riga.startswith("mostrati 6 di 14 trigger (3 con picco ≤ 0, 4 oltre i primi 6")


def test_passo3_nessuno_da_mostrare(capsys):
    import matplotlib.pyplot as plt
    ev = _ev_inventato()
    tutta = pipeline.sintesi_event_study(ev, "M15")
    tutta["picco_pct"] = -tutta["picco_pct"].abs()
    vista, conti = pipeline.sintesi_da_mostrare(tutta, 10)
    assert len(vista) == 0 and conti["mostrati"] == 0
    n_figure = len(plt.get_fignums())
    assert pipeline.grafico_event_study(ev, candidati=[]) == []
    assert len(plt.get_fignums()) == n_figure                               # nessun grafico vuoto
    assert "nessuna curva" in capsys.readouterr().out
    assert pipeline.riga_mostrati(conti, 10).startswith("nessun trigger mostrato")


def test_passo3_costo_e_picco_netto(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    ev = _ev_inventato()
    costo = 10.03
    tutta = pipeline.sintesi_event_study(ev, "M15", costo_bp=costo)
    assert list(tutta.columns).index("picco_netto_bp") == list(tutta.columns).index("picco_pct") + 1
    np.testing.assert_allclose(tutta["picco_netto_bp"], tutta["picco_pct"] * 100 - costo)
    # ordinamento e filtro invariati
    senza = pipeline.sintesi_event_study(ev, "M15")
    assert list(pipeline.sintesi_da_mostrare(tutta, 6)[0]["candidato"]) == \
        list(pipeline.sintesi_da_mostrare(senza, 6)[0]["candidato"])
    vista, _ = pipeline.sintesi_da_mostrare(tutta, 6)
    pipeline.grafico_event_study(ev, candidati=list(vista["candidato"]), costo_bp=costo)
    ax = plt.gcf().axes[0]
    legenda = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(e.startswith("costo di un trade (10.03 bp") for e in legenda)
    linea = [l for l in ax.get_lines() if l.get_label().startswith("costo di un trade")][0]
    assert linea.get_linestyle() == "--" and np.allclose(linea.get_ydata(), costo / 100)   # in %, come l'asse
    assert sum(e.endswith("nel train)") for e in legenda) == 6                         # le curve, invariate
    plt.close("all")


def test_tabella_trigger_solo_train(ds_vero):
    trig = ds_vero.trigger_long[:2]
    lato = pipeline.prepara_lato(ds_vero, 1, trig, 16)
    tab = pipeline.tabella_trigger_train(lato)
    assert list(tab.index) == trig + ["tutti"]
    assert list(tab.columns) == ["occorrenze", "bp_lordi_medi", "bp_netti_medi", "classe_1_%"]
    t = lato.target.loc[lato.idx_train]
    assert tab.loc["tutti", "occorrenze"] == len(lato.idx_train)
    assert np.isclose(tab.loc["tutti", "bp_netti_medi"], t["netto_bp"].mean())
    uno = lato.X.loc[lato.idx_train, trig[0]].astype(bool)
    assert tab.loc[trig[0], "occorrenze"] == int(uno.sum())
    assert np.isclose(tab.loc[trig[0], "classe_1_%"], 100 * t.loc[uno, "TARGET"].mean())
    # cambiare il test (e l'OOS) non cambia la tabella
    fuori = lato.target.index.difference(lato.idx_train)
    target = lato.target.copy()
    target.loc[fuori, ["lordo_bp", "netto_bp"]] = 9999.0
    target.loc[fuori, "TARGET"] = 1
    X = lato.X.copy()
    X.loc[fuori, trig] = 1
    altro = dataclasses.replace(lato, target=target, X=X)
    pd.testing.assert_frame_equal(pipeline.tabella_trigger_train(altro), tab)
    assert "conta per entrambi" in pipeline.NOTA_TRIGGER_INSIEME


def _cella_passo_9():
    import json
    from pathlib import Path
    nb = json.loads((Path(__file__).resolve().parent.parent / "TS-Pipeline-ML.ipynb").read_text(encoding="utf-8"))
    celle = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return [c for c in celle if "pipeline.backtest(" in c][0]


def test_guarda_oos_spento(v4, tmp_path, monkeypatch, capsys):
    import registro

    def vietato(*a, **k):
        raise AssertionError("con GUARDA_OOS = False non si guarda l'OOS")
    monkeypatch.setattr(pipeline, "guarda_oos", vietato)
    monkeypatch.setattr(pipeline, "backtest", vietato)
    ds = SimpleNamespace(cartella=str(tmp_path), symbol="BINANCE_PERP_BTCUSDT", timeframe="M15")
    g = {"pipeline": pipeline, "registro": registro, "ds": ds, "GUARDA_OOS": False, "lati_sel": {1: object()},
         "modelli": {1: object()}, "FASCE_OPERATIVE": 5, "COLLAUDO": True, "display": print}
    exec(_cella_passo_9(), g)
    uscita = capsys.readouterr().out
    assert "OOS non guardato: metti GUARDA_OOS = True quando la scelta è fatta" in uscita
    assert "tab_bt" not in g
    assert not any('"oos"' in f.read_text(encoding="utf-8") for f in tmp_path.rglob("*.jsonl"))
    # solo True apre l'OOS (nemmeno un valore «vero» qualunque)
    assert pipeline.oos_permesso(True) is True
    assert pipeline.oos_permesso(1) is False and pipeline.oos_permesso("True") is False
