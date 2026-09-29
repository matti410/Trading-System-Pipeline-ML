"""
Primo modello di meta-labeling (LightGBM) e sua valutazione, per un lato.

Modulo ADDITIVO (29/9/2026): non modifica nessun file esistente.

Funzioni:

  rendimenti_trigger(df_lato, indice, direction, horizon=25)
      -> Series: guadagno di ogni trigger, STESSA formula del target

  addestra(X, y, idx_train, idx_test)
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


def rendimenti_trigger(df_lato, indice, direction, horizon=25):
    """
    Guadagno (frazione, nel verso del trade) di ogni trigger in `indice`:
    ingresso Open della barra successiva, uscita Close a `horizon` barre
    dal trigger. Stessa formula di target.costruisci_target, cosi'
    (rend > soglia_costi) riproduce esattamente il target.
    """
    pos = df_lato.index.get_indexer(indice)
    if (pos < 0).any() or (pos + horizon >= len(df_lato)).any():
        raise ValueError("indice con trigger assenti o senza horizon barre in avanti")
    entrata = df_lato["Open"].to_numpy()[pos + 1]
    uscita = df_lato["Close"].to_numpy()[pos + horizon]
    return pd.Series((uscita / entrata - 1.0) * direction, index=indice, name="rend")


def _controlla_split(idx_train, idx_test):
    if len(idx_train.intersection(idx_test)) > 0:
        raise ValueError("righe presenti sia in train sia in test")
    if idx_train.max() >= idx_test.min():
        raise ValueError("il train contiene righe non precedenti al test: "
                         "usa gli indici di split_train_test")


def addestra(X, y, idx_train, idx_test, parametri=None, pesi=None):
    """
    Addestra LightGBM SOLO sulle righe idx_train. idx_test serve solo al
    controllo: se il train non viene tutto prima del test, si ferma.

    pesi : None (default, modello di base) oppure Series da
           pesi.pesi_unicita(idx_train, ...), indicizzata su idx_train.
    """
    _controlla_split(idx_train, idx_test)
    sw = None
    if pesi is not None:
        if not pesi.index.equals(idx_train):
            raise ValueError("i pesi devono essere calcolati su idx_train (stesso indice)")
        sw = pesi.to_numpy()
    modello = LGBMClassifier(**(parametri or PARAMETRI_BASE))
    modello.fit(X.loc[idx_train], y.loc[idx_train], sample_weight=sw)
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

    rend : Series da rendimenti_trigger (almeno sulle righe di test).
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
    # rank 'first' -> fasce di uguale numerosita' anche con probabilita' identiche
    t["fascia"] = pd.qcut(t["p"].rank(method="first"), n_fasce, labels=range(1, n_fasce + 1))

    righe = []
    for f, g in t.groupby("fascia", observed=True):
        righe.append({"fascia": str(f), "prob_da": g.p.min(), "prob_a": g.p.max(),
                      "trade": len(g), "vincenti_%": 100 * g.y.mean(),
                      "bp_medi_lordi": g.bp.mean()})
    righe.append({"fascia": "tutti", "prob_da": t.p.min(), "prob_a": t.p.max(),
                  "trade": len(t), "vincenti_%": 100 * t.y.mean(),
                  "bp_medi_lordi": t.bp.mean()})
    return pd.DataFrame(righe).set_index("fascia").round(
        {"prob_da": 3, "prob_a": 3, "vincenti_%": 1, "bp_medi_lordi": 2})


def importanza(modello, X):
    """Peso di ogni feature (guadagno totale degli split), in % del totale."""
    g = modello.booster_.feature_importance(importance_type="gain")
    s = pd.Series(100 * g / g.sum(), index=X.columns, name="peso_%")
    return s.sort_values(ascending=False).round(1)
