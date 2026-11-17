"""
I trigger generati di V4 (docs/83 di V4) come candidati del gruppo automatico (TRIGGER_GENERATI).

- Con TRIGGER_GENERATI = False tutto e' come prima: stessi trigger, stessa proposta.
- Con True il gruppo riceve anche i generati, con le stesse regole, e la proposta e' deterministica.
- Le tabelle mostrano dei generati solo quelli del gruppo, piu' una riga di riepilogo; i provati vanno
  nel registro.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import dataset
import gruppo
import pipeline

COSTI_ZERO = {"spread": 0.0, "spread_passo_pips": 0.0, "commission": 0.0, "pip_size": 0.01,
              "spread_rollover_pips": None}
RICETTE = 3             # le prime ricette del catalogo: bastano per i test e restano veloci


def _catalogo_piccolo():
    """I trigger generati delle prime RICETTE ricette del catalogo di V4, registrati nel motore."""
    from engine.generatore.catalogo import carica_catalogo, registra_trigger_generati, trigger_del_catalogo
    cat = carica_catalogo()
    cat = {**cat, "ricette": cat["ricette"][:RICETTE]}
    registra_trigger_generati(cat)
    return trigger_del_catalogo(cat)


def _ds_sintetico(generati: bool):
    """
    Il mercato sintetico fisso di V4 (6.000 barre M15), costi a zero, due trigger «del catalogo»
    (A ogni 7 barre, B ogni 11) e, con generati=True, i generati delle prime ricette.
    """
    from engine.generatore.catalogo import dataset_sintetico
    df = dataset_sintetico()
    df["A"] = np.arange(len(df)) % 7 == 0
    df["B"] = np.arange(len(df)) % 11 == 0
    lunghi, corti = _catalogo_piccolo() if generati else ({}, {})
    df = dataset._colonne_condizioni(df, list(lunghi) + list(corti), [])      # come dataset.apri
    return SimpleNamespace(df=df, df_is=df, df_oos=df.iloc[:0], costi=COSTI_ZERO, trigger_long=["A", "B"] + list(lunghi),
                           trigger_short=["A", "B"] + list(corti), generati_long=list(lunghi),
                           generati_short=list(corti), solo_long=False, timeframe="M15",
                           symbol="BINANCE_PERP_TEST", cartella=None)


# ---- i generati di V4 ----

def test_trigger_generati_di_v4(v4):
    lunghi, corti, info = dataset.trigger_generati(solo_long=False)
    assert len(lunghi) == len(corti) == info["trigger"] // 2
    assert info["versione"] == 1 and info["ricette"] > 0
    assert all("_SHORT_" in n for n in corti) and not any("_SHORT_" in n for n in lunghi)
    from engine import registry
    assert set(lunghi + corti) <= set(registry.list_entries())
    solo_l, solo_c, _ = dataset.trigger_generati(solo_long=True)
    assert solo_l == lunghi and solo_c == []                     # spot: niente short
    # registrati, i generati non diventano trigger «del catalogo» (vetrina elenca tutto il registro)
    assert not set(lunghi) & set(dataset.trigger_candidati(1))
    assert not set(corti) & set(dataset.trigger_candidati(-1))


def test_colonne_dei_generati_come_le_altre(v4):
    """Le colonne dei generati sono quelle di get_entry(nome)(df), anche calcolate tutte insieme."""
    ds = _ds_sintetico(generati=True)
    from engine import registry
    from engine.generatore import svuota_cache
    for n in ds.generati_long[:3] + ds.generati_short[-3:]:
        svuota_cache()
        pd.testing.assert_series_equal(ds.df[n], registry.get_entry(n)(ds.df).astype(bool), check_names=False)


# ---- il gruppo automatico ----

def test_spento_come_prima(v4):
    """Senza generati la proposta non cambia (stessi trigger del catalogo, stesse regole)."""
    ds = _ds_sintetico(generati=False)
    p = gruppo.proponi(ds, 1, [1, 4], min_per_meta=10)
    assert pipeline.trigger_provati(ds, 1) == {"catalogo": 2, "generati": 0}
    assert list(p.ammessi[1].index) == ["A", "B"]
    if p.H is not None:
        assert pipeline.tabella_composizione(ds, p).equals(p.composizione[p.H])


def test_spento_dati_veri_come_prima(ds_vero):
    """dataset.apri di base: nessun generato, gli stessi trigger del Lab e l'event study su tutti."""
    assert ds_vero.generati_long == ds_vero.generati_short == [] and ds_vero.catalogo_generati is None
    assert ds_vero.trigger_long == dataset.trigger_candidati(1)
    assert ds_vero.trigger_short == dataset.trigger_candidati(-1)
    for d in (1, -1):
        assert pipeline.catalogo_del_lato(ds_vero, d) == (ds_vero.trigger_long if d == 1 else ds_vero.trigger_short)
    assert not any(c.startswith("G0") for c in ds_vero.df.columns)
    assert "generati" not in ds_vero.riepilogo().loc["Trigger"].iloc[0]


def test_acceso_il_gruppo_riceve_i_generati_ed_e_deterministico(v4):
    ds = _ds_sintetico(generati=True)
    for d, nomi in ((1, ds.generati_long), (-1, ds.generati_short)):
        p = gruppo.proponi(ds, d, [1, 4, 8], min_per_meta=10)
        assert pipeline.trigger_provati(ds, d) == {"catalogo": 2, "generati": len(nomi)}
        for H in p.griglia:
            assert list(p.ammessi[H].index) == ["A", "B"] + nomi          # tutti i candidati, nell'ordine
        p2 = gruppo.proponi(_ds_sintetico(generati=True), d, [1, 4, 8], min_per_meta=10)
        pd.testing.assert_frame_equal(p.per_H, p2.per_H)
        assert (p.H, p.gruppo) == (p2.H, p2.gruppo)
        for H in p.griglia:
            pd.testing.assert_frame_equal(p.ammessi[H], p2.ammessi[H])
            pd.testing.assert_frame_equal(p.composizione[H], p2.composizione[H])


def test_i_generati_non_cambiano_i_numeri_del_catalogo(v4):
    """Le medie e le occorrenze di A e B sono le stesse con e senza i generati accanto."""
    spento = gruppo.proponi(_ds_sintetico(generati=False), 1, [1, 4], min_per_meta=10)
    acceso = gruppo.proponi(_ds_sintetico(generati=True), 1, [1, 4], min_per_meta=10)
    for H in (1, 4):
        pd.testing.assert_frame_equal(spento.ammessi[H], acceso.ammessi[H].loc[["A", "B"]])


def test_notebook_interruttore_spento_di_base():
    from pathlib import Path
    nb = json.loads((Path(__file__).resolve().parent.parent / "TS-Pipeline-ML.ipynb").read_text(encoding="utf-8"))
    celle = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    avvio = [c for c in celle if c.startswith("#@title Avvio")][0]
    assert 'TRIGGER_GENERATI = False  #@param {type:"boolean"}' in avvio
    assert any("dataset.apri(" in c and "generati=TRIGGER_GENERATI" in c for c in celle)


# ---- tabelle e registro ----

def _proposta_finta():
    comp = pd.DataFrame({"nuove": [40, 30, 25, 20], "esito": [gruppo.ENTRA, gruppo.ENTRA, "nuove sotto 10 in una metà",
                                                              "media delle nuove ≤ 0 nel train o in una metà"]},
                        index=pd.Index(["G0001_P10_SU", "E10", "G0002_P50_GIU", "E11"], name="trigger"))
    amm = pd.DataFrame({"ammesso": [True, True, True, True, False, False]},
                       index=["G0001_P10_SU", "E10", "G0002_P50_GIU", "E11", "G0003_P90_SU", "E12"])
    return SimpleNamespace(direzione=1, H=8, gruppo=["G0001_P10_SU", "E10"], composizione={8: comp}, ammessi={8: amm})


def test_tabella_composizione_con_i_generati():
    ds = SimpleNamespace(generati_long=["G0001_P10_SU", "G0002_P50_GIU", "G0003_P90_SU"])
    tab = pipeline.tabella_composizione(ds, _proposta_finta())
    assert list(tab.index) == ["G0001_P10_SU", "E10", "E11", pipeline.RIGA_GENERATI]
    assert tab.loc[pipeline.RIGA_GENERATI, "esito"] == "provati 3 · ammessi 2 · nel gruppo 1"
    assert tab.index.name == "trigger"


def test_riga_generati_event_study():
    ds = SimpleNamespace(generati_long=["G1", "G2"], generati_short=[])
    assert "2, non misurati qui" in pipeline.riga_generati_event_study(ds, 1)
    assert pipeline.riga_generati_event_study(ds, -1) == ""
    ds = SimpleNamespace(trigger_long=["E1", "G1", "G2"], generati_long=["G1", "G2"])
    assert pipeline.catalogo_del_lato(ds, 1) == ["E1"]


def test_provati_nel_registro(v4, tmp_path):
    ds = _ds_sintetico(generati=True)
    ds.cartella = str(tmp_path)
    ds.catalogo_generati = {"versione": 1, "ricette": RICETTE, "trigger": 2 * len(ds.generati_long)}
    p = gruppo.proponi(ds, 1, [1], min_per_meta=10)
    pipeline.scelta_lato(ds, p, None, None, 1, collaudo=True, macchina="pc")
    voce = json.loads(next(tmp_path.rglob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    assert voce["trigger_provati"] == {"catalogo": 2, "generati": len(ds.generati_long)}
    assert voce["catalogo_generati"] == ds.catalogo_generati


def test_event_study_con_gruppo_aggiunge_solo_i_mancanti(v4):
    ds = _ds_sintetico(generati=True)
    ds.pip = 0.01
    primi, altro = ds.generati_long[:2], ds.generati_long[2]       # l'event study di V4 legge il registro
    ev = pipeline.event_study_train(ds, 1, 8, quota=0.8, trigger=primi)
    p = SimpleNamespace(direzione=1, gruppo=primi[:1])
    assert pipeline.event_study_con_gruppo(ds, ev, p, 8) is ev
    p = SimpleNamespace(direzione=1, gruppo=[primi[0], altro])
    ev2 = pipeline.event_study_con_gruppo(ds, ev, p, 8)
    assert ev2.candidati_richiesti == primi + [altro]
    pd.testing.assert_series_equal(ev2.curve[primi[0]], ev.curve[primi[0]])
