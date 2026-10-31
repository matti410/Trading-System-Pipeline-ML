"""
La feature selection (lavoro D): regola B con le sonde, gruppi, blocchi, solo train, semi fissi,
stesse righe per modello base e modello con la selezione.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

import selezione as S


def caso(seme_dati, n=5000, rumore=20):
    """I dataset inventati del docs/02 §F: 1 feature vera, `rumore` colonne di rumore, 5.000 trigger."""
    rng = np.random.default_rng(seme_dati)
    idx = pd.date_range("2023-01-01", periods=n * 3, freq="15min", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "Close": 1.0}, index=idx)
    pos = np.sort(rng.choice(n * 3 - 10, n, replace=False))
    ti = idx[pos]
    X = pd.DataFrame({f"rumore_{i}": rng.normal(size=n) for i in range(1, rumore + 1)}, index=ti)
    X.insert(0, "vera", rng.normal(size=n))
    X.insert(0, "E1_TRIG", 1)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-1.0 * X["vera"]))).astype(int), index=ti)
    return X, y, df, ti


@pytest.mark.parametrize("seme_dati", range(8))
def test_una_vera_e_venti_di_rumore(seme_dati):
    """Controllo del comportamento sugli 8 casi del §F (non una garanzia): con la regola B passa solo la vera."""
    X, y, df, ti = caso(seme_dati)
    sel = S.selezione(X, y, df, ti, df.index[-1], 4, sempre=["E1_TRIG"], con_unicita=False)
    assert sel.tenute == ["vera"]
    t = sel.tabella
    assert t.loc["E1_TRIG", "esito"].startswith("sempre")
    assert any(i.startswith("SONDE (10)") for i in t.index)            # la riga delle sonde
    assert all(not e.startswith("tenuta") for c, e in t["esito"].items() if c.startswith("rumore_"))


def test_sonde_a_seme_fisso():
    X, y, df, ti = caso(0, n=2000, rumore=5)
    a = S.selezione(X, y, df, ti, df.index[-1], 4, sempre=["E1_TRIG"], con_unicita=False)
    b = S.selezione(X, y, df, ti, df.index[-1], 4, sempre=["E1_TRIG"], con_unicita=False)
    pd.testing.assert_frame_equal(a.tabella, b.tabella)
    pd.testing.assert_frame_equal(a.importanze, b.importanze)
    s1 = S.aggiungi_sonde(X, seme=0)
    s2 = S.aggiungi_sonde(X, seme=0)
    pd.testing.assert_frame_equal(s1, s2)
    assert [c for c in s1.columns if c.startswith("SONDA_")] == [f"SONDA_{i}" for i in range(1, 11)]
    assert not np.allclose(s1["SONDA_1"], s1["SONDA_2"])               # ogni sonda ha il suo seme


def test_copie_nello_stesso_gruppo_e_rappresentante_senza_target():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    X["a_copia"] = X["a"] * 3 + 1                                      # stessa feature (Spearman 1)
    X["a_rumore"] = X["a"] + rng.normal(scale=0.2, size=500)
    X.loc[::7, "b"] = np.nan                                           # NaN: Spearman a coppie
    g = S.gruppi_correlati(X)
    assert g.loc["a", "gruppo"] == g.loc["a_copia", "gruppo"] == g.loc["a_rumore", "gruppo"]
    assert g.loc["b", "gruppo"] != g.loc["a", "gruppo"]
    assert g.loc[["a", "a_copia", "a_rumore"], "e_rappresentante"].sum() == 1
    assert g.loc["b", "e_rappresentante"]
    # la rappresentante non dipende dal target: gruppi_correlati non lo riceve nemmeno
    assert "y" not in S.gruppi_correlati.__code__.co_varnames


def test_blocchi_senza_sovrapposizioni_con_purge_ed_embargo():
    idx = pd.date_range("2024-01-01", periods=2000, freq="15min", tz="UTC")
    df = pd.DataFrame({"Open": 1.0}, index=idx)
    ti = idx[np.arange(0, 2000, 3)]
    H = 10
    giri = S.blocchi_interni(ti, df, H, idx[-1])
    assert len(giri) == 4
    pos = {t: i for i, t in enumerate(df.index)}
    validazioni = [set(v) for _, v in giri]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not validazioni[i] & validazioni[j]                 # blocchi di validazione disgiunti
    for k, (fit, val) in enumerate(giri, 1):
        assert not set(fit) & set(val)
        inizio_val = pos[val[0]]
        assert all(pos[t] + H + 1 < inizio_val for t in fit)           # purge ed embargo sull'H
        assert max(fit) < min(val)                                     # il fit viene tutto prima
    # ad allargamento: ogni giro ha piu' righe di fit del precedente
    assert all(len(giri[i][0]) < len(giri[i + 1][0]) for i in range(3))


def test_si_ferma_con_una_data_del_test_o_dell_oos():
    X, y, df, ti = caso(0, n=1000, rumore=3)
    inizio_test = ti[800]
    with pytest.raises(ValueError, match="test e OOS non entrano"):
        S.selezione(X, y, df, ti, inizio_test, 4, sempre=["E1_TRIG"], con_unicita=False)
    with pytest.raises(ValueError, match="test e OOS non entrano"):
        S.blocchi_interni(ti, df, 4, inizio_test)
    S.blocchi_interni(ti[:800], df, 4, inizio_test)                    # solo train: va


def test_regola_b_giro_per_giro():
    """Tenuta solo se in 3 giri su 4 sopra zero e sopra la sonda migliore DI QUEL GIRO; poi K."""
    giri = ["giro_1", "giro_2", "giro_3", "giro_4"]
    imp = pd.DataFrame({
        "buona":       [5, 5, 5, -1],
        "mediana_alta_ma_2_giri": [9, 9, 0.5, 0.5],      # mediana sopra le sonde, ma sopra la sonda solo in 2 giri
        "instabile":   [1, -1, -1, 1],
        "SONDA_1":     [1, 1, 1, 1],
        "SONDA_2":     [0.2, 2, 0.2, 0.2],
    }, index=giri).T.astype(float)
    gruppi = pd.DataFrame({"gruppo": range(1, 6), "rappresentante": imp.index, "e_rappresentante": True},
                          index=imp.index)
    tenute, tab = S.seleziona(imp, gruppi, n_train=5000)
    assert tenute == ["buona"]
    assert tab.loc["mediana_alta_ma_2_giri", "esito"].startswith("tolta: sotto la sonda migliore del giro")
    assert tab.loc["instabile", "esito"].startswith("tolta: instabile")
    sonde = tab.loc[[i for i in tab.index if i.startswith("SONDE")][0]]
    assert list(sonde[giri]) == [1, 2, 1, 1]                           # la sonda migliore di ogni giro
    # K = min(15, trigger di train / 100): con 150 trigger K = 1
    imp2 = imp.copy()
    imp2.loc["anche_buona"] = [6, 6, 6, 6]
    gruppi2 = pd.DataFrame({"gruppo": range(1, 7), "rappresentante": imp2.index, "e_rappresentante": True},
                           index=imp2.index)
    tenute2, tab2 = S.seleziona(imp2, gruppi2, n_train=150)
    assert tenute2 == ["anche_buona"] and tab2.loc["buona", "esito"] == "tolta: oltre K = 1"


def test_base_e_selezione_stesse_righe(ds_vero, tmp_path):
    import pipeline
    ds = dataclasses.replace(ds_vero, cartella=str(tmp_path))      # registro della prova in una cartella a parte
    trig = ds.trigger_short[:1]
    feat = pipeline.feature_lato(ds, trig)
    lato = pipeline.prepara_lato(ds, -1, trig, 32, feat=feat)
    from feature.calcolo import riscaldamento
    _, barre = riscaldamento(feat)
    assert lato.target.index.min() >= ds.df.index[barre]                 # niente trigger nel riscaldamento
    assert lato.extra["tolti_riscaldamento"] > 0
    m_base, _ = pipeline.allena(ds, lato, collaudo=True)
    sel, lato_sel = pipeline.seleziona_lato(ds, lato, collaudo=True)
    assert lato_sel.target.index.equals(lato.target.index)
    assert lato_sel.idx_train.equals(lato.idx_train) and lato_sel.idx_test.equals(lato.idx_test)
    assert lato_sel.idx_oos.equals(lato.idx_oos)
    assert list(lato_sel.X.columns) == trig + sel.tenute
    m_sel, _ = pipeline.allena(ds, lato_sel, collaudo=True)
    pipeline.confronto_base_selezione(ds, lato, m_base, lato_sel, m_sel, collaudo=True)
    import registro
    tipi = [v["tipo"] for v in registro.leggi(ds.cartella, f"{ds.symbol}_{ds.timeframe}")]
    assert tipi.count("selezione") == 1 and tipi.count("confronto") == 1 and tipi.count("addestramento") == 2
    # con righe diverse il confronto si ferma
    altro = dataclasses.replace(lato_sel, target=lato_sel.target.iloc[1:])
    with pytest.raises(ValueError, match="stesse righe"):
        pipeline.confronto_base_selezione(ds, lato, m_base, altro, m_sel)


class _Spia:
    """Registra righe e pesi di ogni fit di LightGBM (selezione e modelli passano tutti da modello.addestra)."""

    def __init__(self, monkeypatch):
        import modello
        self.fit = []
        spia = self
        originale = modello.LGBMClassifier

        class Spiato(originale):
            def fit(self, X, y, sample_weight=None, **k):
                spia.fit.append((X.index, None if sample_weight is None else np.asarray(sample_weight).copy()))
                return super().fit(X, y, sample_weight=sample_weight, **k)
        monkeypatch.setattr(modello, "LGBMClassifier", Spiato)


def _pesi_attesi(idx, df, H, netto, peso_rendimento):
    """La regola dei pesi, scritta una volta: unicita' x (peso per rendimento), tetto e media sulle sole righe idx."""
    from pesi import pesi_rendimento, pesi_unicita
    w = pesi_unicita(idx, df, H).to_numpy()
    if peso_rendimento:
        w = w * pesi_rendimento(netto.loc[idx]).to_numpy()
        w = w / w.mean()
    return w


def _caso_pesi(n=1500):
    X, y, df, ti = caso(3, n=n, rumore=3)
    netto = pd.Series(np.random.default_rng(7).normal(0, 50, n), index=ti)
    netto.iloc[::97] = 5000.0                                        # qualche trade estremo: il tetto conta
    return X, y, df, ti, netto


@pytest.mark.parametrize("acceso", [True, False])
def test_selezione_e_modelli_stessi_pesi(monkeypatch, acceso):
    from modello import addestra
    X, y, df, ti, netto = _caso_pesi()
    H, fine = 4, df.index[-1]
    spia = _Spia(monkeypatch)
    S.selezione(X, y, df, ti, fine, H, sempre=["E1_TRIG"], peso_rendimento=acceso, netto_bp=netto)
    giri = S.blocchi_interni(ti, df, H, fine)
    assert len(spia.fit) == len(giri)
    for (righe, w), (idx_fit, _) in zip(spia.fit, giri):
        assert righe.equals(idx_fit)
        assert np.array_equal(w, _pesi_attesi(idx_fit, df, H, netto, acceso))     # al bit
    # il modello finale (passi 7 e 7b) riceve i pesi con la stessa regola, sulle righe di train
    from pesi import pesi_unicita
    tr, te = ti[:1200], ti[1200:]
    m = addestra(X, y, tr, te, pesi=pesi_unicita(tr, df, H), peso_rendimento=acceso, netto_bp=netto)
    assert np.array_equal(spia.fit[-1][1], _pesi_attesi(tr, df, H, netto, acceso))
    assert m.peso_rendimento_ is acceso


def test_tetto_solo_sulle_righe_di_fit(monkeypatch):
    X, y, df, ti, netto = _caso_pesi()
    H, fine = 4, df.index[-1]
    giri = S.blocchi_interni(ti, df, H, fine)
    Xs = S.aggiungi_sonde(X, n=2)
    gruppi = S.gruppi_correlati(Xs.drop(columns=["E1_TRIG"]))
    pesi_giro = []
    for prova in range(2):
        spia = _Spia(monkeypatch)
        n2 = netto.copy()
        if prova:
            n2.loc[n2.index.difference(giri[0][0])] = 1e6            # cambia tutto fuori dal fit del giro 1
        S.importanza_gruppi(Xs, y, df, H, giri[:1], gruppi, fine, ripetizioni=1, peso_rendimento=True, netto_bp=n2)
        pesi_giro.append(spia.fit[0][1])
    assert np.array_equal(pesi_giro[0], pesi_giro[1])


def test_spento_identico(monkeypatch):
    """PESO_RENDIMENTO = False: i pesi sono quelli di unicita' e la selezione e' la stessa al bit di prima."""
    X, y, df, ti, netto = _caso_pesi()
    H, fine = 4, df.index[-1]
    prima = S.selezione(X, y, df, ti, fine, H, sempre=["E1_TRIG"])               # come oggi: senza netto_bp
    spento = S.selezione(X, y, df, ti, fine, H, sempre=["E1_TRIG"], peso_rendimento=False, netto_bp=netto)
    pd.testing.assert_frame_equal(prima.importanze, spento.importanze, check_exact=True)
    assert prima.tenute == spento.tenute


def test_validazione_spento_come_prima(monkeypatch):
    """(1) PESO_RENDIMENTO = False: la log-loss di validazione e' quella di prima (nessun peso), importanze al bit."""
    X, y, df, ti, netto = _caso_pesi()
    H, fine = 4, df.index[-1]
    oggi = S.selezione(X, y, df, ti, fine, H, sempre=["E1_TRIG"], peso_rendimento=False, netto_bp=netto)
    from sklearn.metrics import log_loss

    def di_prima(y, p, pesi=None):                                  # la funzione com'era prima della revisione
        assert pesi is None
        return log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])
    monkeypatch.setattr(S, "_log_loss", di_prima)
    prima = S.selezione(X, y, df, ti, fine, H, sempre=["E1_TRIG"], peso_rendimento=False, netto_bp=netto)
    pd.testing.assert_frame_equal(oggi.importanze, prima.importanze, check_exact=True)


def test_tetto_validazione_solo_dal_fit():
    """(2) Il tetto dei pesi di validazione viene dalle righe di fit: cambiare la validazione non lo cambia."""
    from pesi import tetto_rendimento
    X, y, df, ti, netto = _caso_pesi()
    idx_fit, idx_val = S.blocchi_interni(ti, df, 4, df.index[-1])[0]
    w, tetto = S.pesi_validazione(netto, idx_fit, idx_val)
    assert tetto == tetto_rendimento(netto.loc[idx_fit])
    n2 = netto.copy()
    n2.loc[idx_val] = n2.loc[idx_val] * 50 + 1e5
    w2, tetto2 = S.pesi_validazione(n2, idx_fit, idx_val)
    assert tetto2 == tetto
    atteso = np.minimum(n2.loc[idx_val].abs(), tetto)
    np.testing.assert_allclose(w2, atteso / atteso.mean())          # min(|netto|, tetto del fit), media 1
    assert len(w) == len(idx_val)


def caso_vol_grandi(seme=0, n=5000):
    """
    «vol» predice solo la volatilita': con vol > 0 il mercato e' mosso, i vincenti sono molti di piu'
    ma i bp medi attesi sono gli stessi (-10). «grandi» predice il verso dei trade grandi (8%, 400 bp).
    """
    rng = np.random.default_rng([seme, 12])
    idx = pd.date_range("2023-01-01", periods=n * 3, freq="15min", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "Close": 1.0}, index=idx)
    ti = idx[np.sort(rng.choice(n * 3 - 10, n, replace=False))]
    vol, grandi = rng.normal(size=n), rng.normal(size=n)
    piccolo = rng.normal(0, np.where(vol > 0, 40.0, 8.0))
    salto = rng.random(n) < 0.08
    verso = np.where(rng.random(n) < 0.8, np.sign(grandi), -np.sign(grandi))
    netto = pd.Series(piccolo + salto * verso * 400.0 - 10.0, index=ti)
    X = pd.DataFrame({"E1_TRIG": 1, "vol": vol, "grandi": grandi}, index=ti)
    for i in range(3):
        X[f"rumore_{i}"] = rng.normal(size=n)
    return X, pd.Series((netto > 0).astype(int), index=ti), df, ti, netto


def test_peso_rendimento_premia_i_trade_grandi():
    """(3) Con PESO_RENDIMENTO = True conta chi predice i trade grandi, non chi predice la volatilita'."""
    X, y, df, ti, netto = caso_vol_grandi()
    alto = X["vol"] > 0
    assert y[alto].mean() - y[~alto].mean() > 0.2                     # vol: molti piu' vincenti...
    assert abs(netto[alto].mean() - netto[~alto].mean()) < 10         # ...ma stessi bp medi (a meno del rumore)
    med = {}
    for acceso in (False, True):
        sel = S.selezione(X, y, df, ti, df.index[-1], 4, sempre=["E1_TRIG"], peso_rendimento=acceso, netto_bp=netto)
        med[acceso] = sel.importanze.median(axis=1)
    assert med[True]["grandi"] > med[True]["vol"]
    assert med[False]["vol"] > med[False]["grandi"]                   # spento: vince la volatilita'
