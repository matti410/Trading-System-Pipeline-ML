"""
Primo modello di meta-labeling (LightGBM) e sua valutazione, per un lato.

Modulo ADDITIVO (29/9/2026): non modifica nessun file esistente.

Funzioni:

  rendimenti_trigger(target)
      -> Series: guadagno NETTO di ogni trigger (frazione), dal target

  addestra(X, y, idx_train, idx_test, parametri=None, pesi=None,
           peso_rendimento=True, netto_bp=None)
      -> modello LightGBM addestrato solo sulle righe di train

  valuta(modello, X, y, rend, idx_train, idx_test)
      -> stampa AUC e distribuzione delle probabilita',
         restituisce la tabella per fasce (quintili) del test

  importanza(modello, X)
      -> Series: peso di ogni feature nel modello, in %

Niente soglie di probabilita' fisse: le probabilita' possono essere
concentrate in un intervallo stretto (dipende da symbol e feature), quindi
si ragiona per FASCE di probabilita' (quintili), che funzionano uguale su
qualunque distribuzione. Pesi di unicita': non ancora, si aggiungono dopo
come interruttore.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from oos import piu_meno_bp
from pesi import pesi_rendimento

# impostazioni prudenti, NON ottimizzate (primo giro = modello di base)
PARAMETRI_BASE = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=7,            # coerente con max_depth=3 (2^3 - 1)
    min_child_samples=50,    # niente regole costruite su meno di 50 trigger
    random_state=42,
    n_jobs=1,                # risultato identico a ogni esecuzione
    verbose=-1,
)


def rendimenti_trigger(target):
    """
    Guadagno NETTO di ogni trigger, in frazione del prezzo d'ingresso, preso
    dal target (30/10/2026): la colonna netto_bp di target.costruisci_target,
    con i costi del Lab (commissione, spread di un passo, funding). Cosi'
    rend > 0 coincide con TARGET = 1 per costruzione.
    """
    return (target["netto_bp"] / 1e4).rename("rend")


def _controlla_split(idx_train, idx_test):
    if len(idx_train.intersection(idx_test)) > 0:
        raise ValueError("righe presenti sia in train sia in test")
    if idx_train.max() >= idx_test.min():
        raise ValueError("il train contiene righe non precedenti al test: "
                         "usa gli indici di split_train_test")


def addestra(X, y, idx_train, idx_test, parametri=None, pesi=None,
             peso_rendimento=True, netto_bp=None):
    """
    Addestra LightGBM SOLO sulle righe idx_train. idx_test serve solo al
    controllo: se il train non viene tutto prima del test, si ferma.

    pesi            : None (modello senza pesi) oppure Series da
                      pesi.pesi_unicita(idx_train, ...), indicizzata su idx_train.
    peso_rendimento : interruttore D1, ACCESO di default (decisione di Mattia del 31/10/2026). Ogni riga pesa
                      anche min(|netto_bp|, 95° percentile del train)
                      (pesi.pesi_rendimento): peso = unicita' x rendimento,
                      riportato a media 1. Serve netto_bp (la colonna del target).
    """
    _controlla_split(idx_train, idx_test)
    sw = None
    if pesi is not None:
        if not pesi.index.equals(idx_train):
            raise ValueError("i pesi devono essere calcolati su idx_train (stesso indice)")
        sw = pesi.to_numpy()
    if peso_rendimento:
        if netto_bp is None:
            raise ValueError("peso_rendimento=True vuole netto_bp (la colonna netto_bp del target)")
        r = pesi_rendimento(netto_bp.loc[idx_train]).to_numpy()
        sw = r if sw is None else sw * r
        sw = sw / sw.mean()
    modello = LGBMClassifier(**(parametri or PARAMETRI_BASE))
    modello.fit(X.loc[idx_train], y.loc[idx_train], sample_weight=sw)
    modello.peso_rendimento_ = bool(peso_rendimento)      # per il registro e per il controllo di coerenza
    return modello


def _distribuzione(p):
    q = np.quantile(p, [0, 0.10, 0.50, 0.90, 1.0])
    return "  ".join(f"{n} {v:.3f}" for n, v in zip(["min", "p10", "p50", "p90", "max"], q))


def valuta(modello, X, y, rend, idx_train, idx_test, n_fasce=5):
    """
    Stampa AUC e distribuzione delle probabilita' (train e test) e
    restituisce la tabella del TEST divisa in `n_fasce` fasce di uguale
    numerosita', dalla probabilita' piu' bassa (1) alla piu' alta, piu'
    la riga di riferimento "tutti".

    rend : Series da rendimenti_trigger (netto, almeno sulle righe di test).
    """
    _controlla_split(idx_train, idx_test)
    p_tr = modello.predict_proba(X.loc[idx_train])[:, 1]
    p_te = modello.predict_proba(X.loc[idx_test])[:, 1]
    y_tr, y_te = y.loc[idx_train], y.loc[idx_test]

    print(f"AUC   train {roc_auc_score(y_tr, p_tr):.3f}   test {roc_auc_score(y_te, p_te):.3f}"
          f"   (0.5 = ordine casuale)")
    print(f"prob. train: {_distribuzione(p_tr)}")
    print(f"prob. test:  {_distribuzione(p_te)}")

    t = pd.DataFrame({"p": p_te, "y": y_te.to_numpy(),
                      "bp": rend.loc[idx_test].to_numpy() * 1e4}, index=idx_test)
    righe = []
    distinti = len(np.unique(p_te))
    if distinti < n_fasce:
        # stessa protezione di oos.controlla_fasce: niente fasce con troppo pochi valori distinti
        print(f"fasce del test non calcolate: le probabilità hanno {distinti} valori distinti, "
              f"meno delle {n_fasce} fasce. Si mostra solo «tutti».")
        t["fascia"] = "tutti"
    else:
        t["fascia"] = _etichette_fasce(t["p"], n_fasce)
    for f, g in (t.groupby("fascia", observed=True) if distinti >= n_fasce else []):
        righe.append({"fascia": str(f), "prob_da": g.p.min(), "prob_a": g.p.max(),
                      "trade": len(g), "vincenti_%": 100 * g.y.mean(),
                      "bp_medi_netti": g.bp.mean(), "±bp": piu_meno_bp(g.bp)})
    righe.append({"fascia": "tutti", "prob_da": t.p.min(), "prob_a": t.p.max(),
                  "trade": len(t), "vincenti_%": 100 * t.y.mean(),
                  "bp_medi_netti": t.bp.mean(), "±bp": piu_meno_bp(t.bp)})
    return pd.DataFrame(righe).set_index("fascia").round(
        {"prob_da": 3, "prob_a": 3, "vincenti_%": 1, "bp_medi_netti": 2, "±bp": 2})


def _etichette_fasce(p: pd.Series, n_fasce: int):
    """Le fasce del test: rank 'first' -> fasce di uguale numerosita' anche con probabilita' identiche."""
    return pd.qcut(p.rank(method="first"), n_fasce, labels=range(1, n_fasce + 1))


def differenza_con_incertezza(bp_alta, bp_tutti) -> tuple:
    """
    bp medi della fascia alta meno bp medi di tutti, con il suo ± (circa 2 errori standard).
    La fascia e' DENTRO «tutti», quindi i due non sono indipendenti: la differenza e'
    (1 - n_a/N) x (media della fascia - media degli altri), e il ± viene da qui:
    2 x radice((1 - n_a/N)^2 x (s_a^2/n_a + s_altri^2/n_altri)). Ritorna (differenza, ±).
    """
    alta, tutti = np.asarray(bp_alta, dtype=float), np.asarray(bp_tutti, dtype=float)
    n_a, n = len(alta), len(tutti)
    somma_altri = tutti.sum() - alta.sum()
    q_altri = (tutti ** 2).sum() - (alta ** 2).sum()
    n_r = n - n_a
    diff = alta.mean() - tutti.mean()
    if n_a < 2 or n_r < 2:
        return float(diff), float("nan")
    media_r = somma_altri / n_r
    var_r = max((q_altri - n_r * media_r ** 2) / (n_r - 1), 0.0)
    f = 1 - n_a / n
    return float(diff), float(2 * np.sqrt(f ** 2 * (np.var(alta, ddof=1) / n_a + var_r / n_r)))


def fascia_alta_contro_tutti(modello, X, rend, idx_test, n_fasce=5):
    """
    Sul test: (differenza, ±) fra i bp netti medi della fascia piu' alta e quelli di tutti i
    trigger, con le stesse fasce di valuta. (NaN, NaN) se le fasce non si calcolano.
    """
    p = pd.Series(modello.predict_proba(X.loc[idx_test])[:, 1], index=idx_test)
    if len(np.unique(p)) < n_fasce:
        return float("nan"), float("nan")
    bp = rend.loc[idx_test].to_numpy() * 1e4
    alta = (_etichette_fasce(p, n_fasce) == n_fasce).to_numpy()
    return differenza_con_incertezza(bp[alta], bp)


def importanza(modello, X):
    """Peso di ogni feature (guadagno totale degli split), in % del totale. Nessuno split: tutto 0."""
    g = modello.booster_.feature_importance(importance_type="gain")
    s = pd.Series(100 * g / g.sum() if g.sum() > 0 else 0.0 * g, index=X.columns, name="peso_%")
    return s.sort_values(ascending=False).round(1)
