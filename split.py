"""
Split train/test dentro df_is e matrice X per i modelli ML (LONG / SHORT).

Modulo ADDITIVO (29/9/2026): non modifica nessun file esistente.

Tre funzioni:

  barra_confine(df_is, quota=0.8) -> int
      la posizione del confine train/test (la usa anche l'event study del train)

  split_train_test(target, df_is, horizon, quota=0.8)
      -> (idx_train, idx_test)

  costruisci_X(df_lato, target, colonne_trigger, colonne_contesto)
      -> DataFrame di feature 0/1, allineato al target

PERCHE' SERVONO PURGE ED EMBARGO
--------------------------------
L'etichetta di un trigger sulla barra i si decide guardando i prezzi fino
all'Open della barra i + horizon + 1, dove il motore del Lab esce (target.py,
decisione del 30/10/2026). Quindi:

  - PURGE (confine IS/OOS): un trigger delle ultime `horizon` + 1 barre di
    df_is ha l'etichetta calcolata con prezzi di df_oos. Va escluso, se
    no l'OOS "entra" nello sviluppo del modello.

  - EMBARGO (confine train/test): un trigger di train vicino al confine ha
    l'etichetta calcolata con prezzi del periodo di test. Va escluso dal
    train, se no il test non e' piu' "mai visto".

Il test NON perde righe: i suoi trigger guardano avanti, dentro il test
stesso (o vengono tolti dal purge se sforano in OOS).

Pesi di unicita' (decisione A del 28/9): NON sono qui. Si aggiungono in un
secondo momento come interruttore separato, dopo un primo modello di base.
"""
import pandas as pd

_OHLCV = {"Open", "High", "Low", "Close", "Volume", "spread"}


def barra_confine(df_is: pd.DataFrame, quota: float = 0.8) -> int:
    """
    Posizione (in barre di df_is) della prima barra del test: le barre prima
    sono il train. Unica regola per lo split e per l'event study del train
    (pipeline.event_study_train), cosi' il confine e' lo stesso.
    """
    if not 0 < quota < 1:
        raise ValueError("quota deve stare fra 0 e 1")
    return round(len(df_is) * quota)


def split_train_test(target, df_is: pd.DataFrame, horizon: int, quota: float = 0.8):
    """
    Divide i trigger di `target` in train e test, per TEMPO, dentro df_is.

    Parametri
    ---------
    target  : il DataFrame di target.costruisci_target (o la sua colonna
              TARGET), indicizzato sulle barre trigger di tutto il df.
    df_is   : la parte in-sample del df (serve per posizioni e confine).
    horizon : H del lato, in barre, lo stesso del target (obbligatorio).
    quota   : frazione delle barre di df_is assegnata al train (default 0.8).

    Il confine e' una BARRA (posizione quota * len(df_is)), non un numero
    di trigger: cosi' LONG e SHORT hanno lo stesso confine temporale.

    Ritorna
    -------
    (idx_train, idx_test) : due DatetimeIndex, da usare come
    target.loc[idx_train], X.loc[idx_test], ecc.
    """
    y = target["TARGET"] if isinstance(target, pd.DataFrame) else target
    n_is = len(df_is)
    confine = barra_confine(df_is, quota)

    pos = df_is.index.get_indexer(target.index)   # -1 = trigger fuori da df_is
    in_is = pos >= 0
    pos_is = pos[in_is]
    idx_is = target.index[in_is]

    # PURGE: l'uscita (Open della barra trigger + horizon + 1) deve cadere dentro df_is
    ok_is = pos_is + horizon + 1 < n_is

    # train: finestra dell'etichetta [trigger+1, trigger+horizon+1] tutta prima del confine
    train = ok_is & (pos_is + horizon + 1 < confine)
    # test: trigger dal confine in poi
    test = ok_is & (pos_is >= confine)

    idx_train = idx_is[train]
    idx_test = idx_is[test]

    # ---- riepilogo ------------------------------------------------------
    n_oos = int((~in_is).sum())
    n_purge = int((~ok_is).sum())
    n_embargo = int((ok_is & ~train & ~test).sum())
    print(f"confine train/test: {df_is.index[confine]}  (barra {confine} di {n_is})")
    print(f"  trigger in OOS (non usati):       {n_oos}")
    print(f"  tolti dal purge (uscita in OOS):  {n_purge}")
    print(f"  tolti dall'embargo (train->test): {n_embargo}")
    print(f"  TRAIN: {len(idx_train):5d} righe, classe 1 {y.loc[idx_train].mean():.1%}"
          f"  ({idx_train[0].date()} -> {idx_train[-1].date()})")
    print(f"  TEST:  {len(idx_test):5d} righe, classe 1 {y.loc[idx_test].mean():.1%}"
          f"  ({idx_test[0].date()} -> {idx_test[-1].date()})")

    return idx_train, idx_test


def costruisci_X(df_lato: pd.DataFrame, target, colonne_trigger: list,
                 colonne_contesto: list) -> pd.DataFrame:
    """
    Matrice delle feature per un lato: le colonne trigger del lato (l'identita'
    del trigger) + le colonne di contesto, convertite in 0/1, una riga per ogni
    riga del target (stesso indice, stesso ordine).

    colonne_contesto : l'elenco dei filtri del Lab, dataset.apri(...).filtri
        (F* e VWAP_*). Non si riconoscono dal nome: tre filtri del Lab non
        cominciano con F (lavoro B, 30/10/2026).

    Mai colonne OHLCV: il modello vede solo "quale trigger" e "in che
    contesto", non il livello del prezzo.
    """
    colonne_f = [c for c in colonne_contesto if c not in colonne_trigger]
    colonne = list(colonne_trigger) + colonne_f
    mancanti = [c for c in colonne if c not in df_lato.columns]
    if mancanti:
        raise KeyError(f"colonne non presenti nel df: {mancanti}")

    vietate = _OHLCV.intersection(colonne)
    if vietate:
        raise ValueError(f"colonne di prezzo nella X: {sorted(vietate)}")

    X = df_lato.loc[target.index, colonne]
    if X.isna().any().any():
        raise ValueError("NaN nella X: controlla il taglio del riscaldamento")

    X = X.astype(int)
    print(f"X: {X.shape[0]} righe x {X.shape[1]} colonne "
          f"({len(colonne_trigger)} trigger + {len(colonne_f)} filtri di contesto)")
    return X
