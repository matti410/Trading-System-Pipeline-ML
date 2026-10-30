"""
Il dataset preso da V4: costi uguali a quelli del Lab, studio di V4 mai toccato.
Servono i dati veri (BINANCE_PERP_BTCUSDT M15) e lo studio: senza, si saltano.
"""
import os
from pathlib import Path

import pandas as pd

from conftest import SYMBOL_VERO, TF_VERO


def _fotografia(cartella):
    """(percorso, dimensione, ora di modifica) di ogni file sotto studio/."""
    radice = Path(cartella) / "studio"
    return sorted((str(p.relative_to(radice)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in radice.rglob("*") if p.is_file())


def test_studio_di_v4_intatto(cartella_dati):
    import dataset
    prima = _fotografia(cartella_dati)
    ds = dataset.apri(SYMBOL_VERO, TF_VERO, cartella_dati, condizioni=False, verbose=False)
    assert _fotografia(cartella_dati) == prima
    assert ds.impronta == "ok"


def test_costi_identici_al_lab(ds_vero, cartella_dati):
    """Stessi costi che la schermata 1 del Lab passa ai motori, ricalcolati qui da V4."""
    from engine import vetrina
    from engine.exchange import costi_exchange
    from engine.studio import passo
    from engine.splitting import split_al_confine
    d = vetrina.apri_dati(SYMBOL_VERO, TF_VERO, cartella_dati, confine=ds_vero.confine, verbose=False)
    df_is = split_al_confine(d.df, ds_vero.confine, passo(TF_VERO))[0]
    costi_lab, _ = vetrina.costi_storico(SYMBOL_VERO, d.df, funding=d.funding, df_prezzo=df_is)
    assert set(costi_lab) == set(ds_vero.costi)
    for k, v in costi_lab.items():
        if isinstance(v, pd.DataFrame):
            pd.testing.assert_frame_equal(v, ds_vero.costi[k])
        else:
            assert ds_vero.costi[k] == v, k
    listino = costi_exchange(SYMBOL_VERO, df_is)
    assert ds_vero.costi["commission"] == listino["taker"] == 0.0005     # Binance perpetual, taker
    assert ds_vero.costi["spread"] == 0.0                                 # lo spread e' un passo a parte
    assert ds_vero.costi["spread_passo_pips"] == listino["spread_pips"]
    assert "funding" in ds_vero.costi


def test_confine_e_divisione(ds_vero):
    from engine.studio import passo
    p = passo(TF_VERO)
    assert ds_vero.df_is.index[-1] + p <= ds_vero.confine < ds_vero.df_oos.index[0] + p
    assert len(ds_vero.df_is) + len(ds_vero.df_oos) == len(ds_vero.df)
    assert ds_vero.df.index.tz is not None and str(ds_vero.df.index.tz) == "UTC"


def test_colonne_trigger_e_filtri(ds_vero):
    assert ds_vero.trigger_long and ds_vero.trigger_short and ds_vero.filtri
    colonne = ds_vero.trigger_long + ds_vero.trigger_short + ds_vero.filtri
    assert all(ds_vero.df[c].dtype == bool for c in colonne)
    # i filtri del Lab non si chiamano tutti F*: ci sono anche i VWAP_* (vwap_regime_filter_conditions)
    assert any(c.startswith("VWAP_") for c in ds_vero.filtri)
    assert not set(ds_vero.filtri) & set(ds_vero.trigger_long + ds_vero.trigger_short)
    assert not ds_vero.solo_long


def test_riepilogo(ds_vero):
    r = ds_vero.riepilogo()
    assert "Confine dello studio" in r.index and "Costo per trade" in r.index
    assert os.path.isdir(ds_vero.cartella)


def _fotografia_v4(cartella_v4):
    """Ogni file di V4 (senza .venv e senza .git, tranne il suo indice) con dimensione e ora di modifica."""
    radice = Path(cartella_v4)
    voci = []
    for p in radice.rglob("*"):
        parti = p.relative_to(radice).parts
        if parti[0] in (".venv", ".git") or not p.is_file():
            continue
        voci.append((str(p.relative_to(radice)), p.stat().st_size, p.stat().st_mtime_ns))
    indice = radice / ".git" / "index"
    voci.append((".git/index", indice.stat().st_size, indice.stat().st_mtime_ns))
    return sorted(voci)


def test_v4_intatto(v4, cartella_dati):
    """Collegamento, dataset e target non scrivono niente nella cartella di V4 (neanche le cache)."""
    import collegamento_v4
    import dataset
    from target import costruisci_target
    prima = _fotografia_v4(v4["cartella"])
    collegamento_v4.collega(verbose=False)
    ds = dataset.apri(SYMBOL_VERO, TF_VERO, cartella_dati, verbose=False)
    costruisci_target(ds.df, ds.trigger_long[:1], 1, 16, costi=ds.costi)
    assert _fotografia_v4(v4["cartella"]) == prima
