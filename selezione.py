"""
Feature selection per lato (lavoro D, 30/10/2026, docs/02 §D con le sonde del §F).

  controlla_solo_train(indice, inizio_test)
  blocchi_interni(idx_train, df, horizon, inizio_test, n_blocchi=5) -> list[(idx_fit, idx_val)]
  aggiungi_sonde(X, n=10, seme=0) -> DataFrame
  gruppi_correlati(X_train, soglia=0.7) -> DataFrame
  importanza_gruppi(X, y, df, horizon, blocchi, gruppi, ...) -> DataFrame
  seleziona(importanze, gruppi, n_train) -> (tenute, tabella)
  selezione(X, y, df, idx_train, inizio_test, horizon, sempre=(), ...) -> Selezione

TUTTO DENTRO IL TRAIN: test e OOS non entrano mai. Ogni funzione che riceve righe
controlla che vengano tutte prima della prima barra del test, e si ferma se no.

LE REGOLE, FISSATE PRIMA DI GUARDARE I DATI
-------------------------------------------
1. Blocchi: i trigger di train, in ordine di tempo, divisi in 5 blocchi consecutivi con
   lo stesso numero di trigger. 4 giri ad allargamento: fit sui blocchi 1..k, validazione
   sul blocco k+1 (k = 1..4). Purge ed embargo come split.py: una riga di fit resta fuori
   se la sua uscita (barra + H + 1) arriva al blocco di validazione.
2. Sonde: 10 colonne di rumore normale (seme fisso), SONDA_1..SONDA_10, trattate come
   tutte le altre colonne (entrano nei gruppi e nella permutazione).
3. Gruppi: Spearman sul solo train, a coppie (i NaN si ignorano coppia per coppia).
   Raggruppamento gerarchico «average» su 1 - |rho|, tagliato a 1 - 0,7: i gruppi tengono
   insieme colonne con |rho| >= 0,7. Rappresentante = la colonna con |rho| medio piu' alto
   verso le altre del gruppo (il target non entra). L'identita' del trigger non entra nei
   gruppi: resta sempre.
4. Importanza: in ogni giro LightGBM (PARAMETRI_BASE, pesi di unicita') su tutte le
   colonne; sul blocco di validazione si mescolano INSIEME le colonne di un gruppo (stessa
   permutazione delle righe) e si misura quanto peggiora la log-loss. Media di 5
   mescolamenti con semi fissi. Con PESO_RENDIMENTO acceso (revisione della PR #8) anche la
   log-loss di validazione, base e mescolata, e' pesata: min(|netto_bp|, tetto) a media 1, con
   il tetto delle righe di FIT del giro (lo stesso dell'addestramento); niente unicita' sulla
   validazione. Cosi' conta quanto la colonna aiuta sui trade grandi, non sulla frequenza delle vincite.
5. Regola B (decisione di Mattia del 30/10/2026, docs/02 §F): si tiene un gruppo se in
   almeno 3 giri su 4 la sua importanza e' sopra zero E sopra la sonda migliore DI QUEL GIRO.
   Poi al massimo K = min(15, trigger di train / 100) gruppi, i primi per mediana. Del gruppo
   tenuto entra nel modello la rappresentante. (La regola di prima, mediana contro la sonda
   migliore, lasciava passare circa una colonna di rumore su 20: docs/02 §F.)
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

N_BLOCCHI = 5
GIRI_POSITIVI = 3
SOGLIA_RHO = 0.7
RIPETIZIONI = 5
N_SONDE = 10
K_MAX = 15
TRIGGER_PER_FEATURE = 100
PREFISSO_SONDA = "SONDA_"


def controlla_solo_train(indice, inizio_test) -> None:
    """Si ferma se una data dell'indice e' nel test o nell'OOS (dalla prima barra del test in poi)."""
    indice = pd.DatetimeIndex(indice)
    if len(indice) and indice.max() >= pd.Timestamp(inizio_test):
        raise ValueError(f"la selezione ha ricevuto righe dal {indice.max()}: test e OOS non entrano "
                         f"(il test comincia il {pd.Timestamp(inizio_test)})")


def blocchi_interni(idx_train, df, horizon, inizio_test, n_blocchi=N_BLOCCHI):
    """I giri ad allargamento: [(idx_fit, idx_val), ...], n_blocchi - 1 giri."""
    controlla_solo_train(idx_train, inizio_test)
    idx = pd.DatetimeIndex(idx_train).sort_values()
    pos = df.index.get_indexer(idx)
    if (pos < 0).any():
        raise ValueError("trigger di train che non sono nel df")
    parti = np.array_split(np.arange(len(idx)), n_blocchi)
    giri = []
    for k in range(1, n_blocchi):
        val = parti[k]
        inizio_val = pos[val[0]]
        fit = np.concatenate(parti[:k])
        fit = fit[pos[fit] + int(horizon) + 1 < inizio_val]          # purge ed embargo sull'H del lato
        giri.append((idx[fit], idx[val]))
    return giri


def aggiungi_sonde(X, n=N_SONDE, seme=0):
    """X con n colonne di rumore normale; il rumore di ogni sonda ha il suo seme fisso."""
    sonde = {f"{PREFISSO_SONDA}{i}": np.random.default_rng([seme, i]).standard_normal(len(X))
             for i in range(1, n + 1)}
    return pd.concat([X, pd.DataFrame(sonde, index=X.index)], axis=1)


def gruppi_correlati(X_train, soglia=SOGLIA_RHO):
    """Per ogni colonna: il gruppo, la rappresentante del gruppo e se la colonna e' la rappresentante."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    colonne = list(X_train.columns)
    if len(colonne) == 1:
        return pd.DataFrame({"gruppo": [1], "rappresentante": colonne, "e_rappresentante": [True]}, index=colonne)
    valori = X_train.astype(float).corr(method="spearman").abs().fillna(0.0).to_numpy(copy=True)  # a coppie
    np.fill_diagonal(valori, 1.0)
    rho = pd.DataFrame(valori, index=colonne, columns=colonne)
    distanza = np.clip(1.0 - valori, 0.0, None)
    np.fill_diagonal(distanza, 0.0)
    distanza = (distanza + distanza.T) / 2
    legami = linkage(squareform(distanza, checks=False), method="average")
    etichette = fcluster(legami, t=1.0 - soglia, criterion="distance")
    righe = []
    for g in pd.unique(etichette):
        membri = [c for c, e in zip(colonne, etichette) if e == g]
        if len(membri) == 1:
            rappr = membri[0]
        else:
            medie = rho.loc[membri, membri].sum(axis=1).sub(1.0) / (len(membri) - 1)
            rappr = medie.idxmax()                     # a parita', la prima nell'ordine delle colonne
        righe += [(c, rappr) for c in membri]
    tab = pd.DataFrame(righe, columns=["colonna", "rappresentante"]).set_index("colonna").loc[colonne]
    numeri = {r: i + 1 for i, r in enumerate(pd.unique(tab["rappresentante"]))}
    tab.insert(0, "gruppo", tab["rappresentante"].map(numeri))
    tab["e_rappresentante"] = tab.index == tab["rappresentante"]
    return tab


def _log_loss(y, p, pesi=None):
    from sklearn.metrics import log_loss
    return log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1], sample_weight=pesi)


def pesi_validazione(netto_bp, idx_fit, idx_val):
    """
    Pesi della log-loss sul blocco di validazione, con PESO_RENDIMENTO acceso: min(|netto_bp|,
    tetto) a media 1, con il tetto = 95° percentile di |netto_bp| delle SOLE righe di fit del
    giro. Niente pesi di unicita'. Ritorna (pesi sulle righe di idx_val, tetto).
    """
    from pesi import pesi_rendimento, tetto_rendimento
    tetto = tetto_rendimento(netto_bp.loc[idx_fit])
    return pesi_rendimento(netto_bp.loc[idx_val], tetto=tetto).to_numpy(), tetto


def importanza_gruppi(X, y, df, horizon, blocchi, gruppi, inizio_test, ripetizioni=RIPETIZIONI, seme=0,
                      con_unicita=True, peso_rendimento=True, netto_bp=None, parametri=None):
    """
    Importanza di ogni gruppo in ogni giro: aumento medio della log-loss sul blocco di
    validazione quando si mescolano insieme le colonne del gruppo. Una riga per gruppo
    (indice = rappresentante), una colonna per giro.
    """
    from modello import PARAMETRI_BASE, addestra
    from pesi import pesi_unicita
    membri = gruppi.groupby("rappresentante", sort=False).groups
    imp = {}
    for g, (idx_fit, idx_val) in enumerate(blocchi, 1):
        controlla_solo_train(idx_fit.append(idx_val), inizio_test)
        pesi = pesi_unicita(idx_fit, df, horizon) if con_unicita else None
        m = addestra(X, y, idx_fit, idx_val, parametri=parametri or PARAMETRI_BASE, pesi=pesi,
                     peso_rendimento=peso_rendimento, netto_bp=netto_bp)
        Xv = X.loc[idx_val]
        yv = y.loc[idx_val].to_numpy()
        wv = pesi_validazione(netto_bp, idx_fit, idx_val)[0] if peso_rendimento else None
        base = _log_loss(yv, m.predict_proba(Xv)[:, 1], wv)
        colonna = {}
        for r, cols in membri.items():
            aumenti = []
            for rip in range(ripetizioni):
                ordine = np.random.default_rng([seme, g, rip]).permutation(len(Xv))
                Xp = Xv.copy()
                Xp[list(cols)] = Xv[list(cols)].to_numpy()[ordine]
                aumenti.append(_log_loss(yv, m.predict_proba(Xp)[:, 1], wv) - base)
            colonna[r] = float(np.mean(aumenti))
        imp[f"giro_{g}"] = pd.Series(colonna)
    return pd.DataFrame(imp)


def seleziona(importanze, gruppi, n_train, sempre=()):
    """
    La regola del punto 5 (regola B, decisione di Mattia del 30/10/2026). Ritorna (tenute,
    tabella): le rappresentanti tenute, in ordine di mediana, e una riga per colonna con
    gruppo, importanza per giro, giri sopra zero, giri sopra la sonda, mediana ed esito,
    piu' la riga delle sonde (la sonda migliore di ogni giro).
    """
    imp = importanze.copy()
    giri = list(imp.columns)
    con_sonda = [r for r in imp.index
                 if any(str(c).startswith(PREFISSO_SONDA) for c in gruppi.index[gruppi["rappresentante"] == r])]
    # la soglia di ogni giro: zero e la sonda migliore di quel giro
    sonda_giro = imp.loc[con_sonda, giri].max() if con_sonda else pd.Series(-np.inf, index=giri)
    imp["positivi"] = (imp[giri] > 0).sum(axis=1)
    imp["sopra_sonda"] = ((imp[giri] > 0) & (imp[giri] > sonda_giro)).sum(axis=1)
    imp["mediana"] = imp[giri].median(axis=1)
    K = min(K_MAX, int(n_train) // TRIGGER_PER_FEATURE)
    candidati = [r for r in imp.index if r not in con_sonda]
    passano = [r for r in candidati if imp.loc[r, "sopra_sonda"] >= GIRI_POSITIVI]
    tenute = sorted(passano, key=lambda r: -imp.loc[r, "mediana"])[:K]
    motivo = {}
    for r in candidati:
        if r in tenute:
            motivo[r] = "tenuta"
        elif imp.loc[r, "positivi"] < GIRI_POSITIVI:
            motivo[r] = f"tolta: instabile ({int(imp.loc[r, 'positivi'])} giri su {len(giri)} sopra zero)"
        elif r not in passano:
            motivo[r] = (f"tolta: sotto la sonda migliore del giro ({int(imp.loc[r, 'sopra_sonda'])} giri su "
                         f"{len(giri)} sopra la sonda)")
        else:
            motivo[r] = f"tolta: oltre K = {K}"
    righe = []
    for col, g in gruppi.iterrows():
        r = g["rappresentante"]
        if str(col).startswith(PREFISSO_SONDA):
            continue
        esito = motivo[r] if g["e_rappresentante"] else f"tolta: doppione di {r}"
        righe.append({"colonna": col, "gruppo": int(g["gruppo"]), **imp.loc[r, giri].to_dict(),
                      "positivi": int(imp.loc[r, "positivi"]), "sopra_sonda": int(imp.loc[r, "sopra_sonda"]),
                      "mediana": imp.loc[r, "mediana"], "esito": esito})
    for c in sempre:
        righe.append({"colonna": c, "gruppo": 0, "esito": "sempre (identità del trigger)"})
    if con_sonda:
        migliore = imp.loc[con_sonda, giri].idxmax()
        righe.append({"colonna": f"SONDE ({len(con_sonda)}): la migliore di ogni giro", "gruppo": np.nan,
                      **sonda_giro.to_dict(), "positivi": np.nan, "sopra_sonda": np.nan,
                      "mediana": float(sonda_giro.median()),
                      "esito": "soglia del giro · " + ", ".join(f"{g}: {migliore[g]}" for g in giri)})
    tab = pd.DataFrame(righe).set_index("colonna")
    return tenute, tab


@dataclass
class Selezione:
    tenute: list
    tabella: pd.DataFrame
    gruppi: pd.DataFrame
    importanze: pd.DataFrame
    sonda_migliore: float          # mediana, fra i giri, della sonda migliore di ogni giro
    K: int
    secondi: float
    conteggio: dict = field(default_factory=dict)


def conta_esiti(tabella) -> dict:
    """Quante colonne tenute e tolte per motivo (senza la riga delle sonde e le colonne sempre tenute)."""
    e = tabella.loc[~tabella.index.str.startswith("SONDE"), "esito"]
    e = e[~e.str.startswith("sempre")]
    motivi = e.str.replace(r"\(.*\)|di .*$|= \d+$", "", regex=True).str.strip()
    return motivi.value_counts().to_dict()


def selezione(X, y, df, idx_train, inizio_test, horizon, sempre=(), n_blocchi=N_BLOCCHI, soglia=SOGLIA_RHO,
              ripetizioni=RIPETIZIONI, n_sonde=N_SONDE, seme=0, con_unicita=True, peso_rendimento=True,
              netto_bp=None, parametri=None) -> Selezione:
    """
    La selezione completa sul solo train. X: le candidate piu' le colonne `sempre` (identita'
    del trigger), una riga per trigger. Ritorna la Selezione con le colonne tenute.
    """
    t0 = time.perf_counter()
    controlla_solo_train(idx_train, inizio_test)
    idx_train = pd.DatetimeIndex(idx_train).sort_values()
    Xs = aggiungi_sonde(X.loc[idx_train], n=n_sonde, seme=seme)
    candidate = [c for c in Xs.columns if c not in sempre]
    gruppi = gruppi_correlati(Xs[candidate], soglia=soglia)
    blocchi = blocchi_interni(idx_train, df, horizon, inizio_test, n_blocchi=n_blocchi)
    imp = importanza_gruppi(Xs, y, df, horizon, blocchi, gruppi, inizio_test, ripetizioni=ripetizioni, seme=seme,
                            con_unicita=con_unicita, peso_rendimento=peso_rendimento, netto_bp=netto_bp,
                            parametri=parametri)
    tenute, tab = seleziona(imp, gruppi, len(idx_train), sempre=sempre)
    con_sonda = tab.index[tab.index.str.startswith("SONDE")]
    sonda = float(tab.loc[con_sonda[0], "mediana"]) if len(con_sonda) else -np.inf
    return Selezione(tenute=tenute, tabella=tab, gruppi=gruppi, importanze=imp, sonda_migliore=sonda,
                     K=min(K_MAX, len(idx_train) // TRIGGER_PER_FEATURE), secondi=time.perf_counter() - t0,
                     conteggio=conta_esiti(tab))
