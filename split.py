"""
Split train/test dentro df_is e matrice X per i modelli ML (LONG / SHORT).

Modulo ADDITIVO (29/9/2026): non modifica nessun file esistente.

Due funzioni:

  split_train_test(target, df_is, horizon=25, quota=0.8)
      -> (idx_train, idx_test)

  costruisci_X(df_lato, target, colonne_trigger)
      -> DataFrame di feature 0/1, allineato al target

PERCHE' SERVONO PURGE ED EMBARGO
--------------------------------
L'etichetta di un trigger sulla barra i si decide guardando i prezzi fino
alla barra i + horizon. Quindi:

  - PURGE (confine IS/OOS): un trigger delle ultime `horizon` barre di
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
import re

import pandas as pd

_PATTERN_F = re.compile(r"^F\d+_")
_OHLCV = {"Open", "High", "Low", "Close", "Volume", "spread"}


def split_train_test(target: pd.Series, df_is: pd.DataFrame,
                     horizon: int = 25, quota: float = 0.8):
    """
    Divide i trigger di `target` in train e test, per TEMPO, dentro df_is.

    Parametri
    ---------
    target  : Series 0/1 da costruisci_target (TARGET_LONG o TARGET_SHORT),
              indicizzata sulle barre trigger di tutto il df.
    df_is   : la parte in-sample del df (serve per posizioni e confine).
    horizon : barre usate dal target (default 25, come H).
    quota   : frazione delle barre di df_is assegnata al train (default 0.8).

    Il confine e' una BARRA (posizione quota * len(df_is)), non un numero
    di trigger: cosi' LONG e SHORT hanno lo stesso confine temporale.

    Ritorna
    -------
    (idx_train, idx_test) : due DatetimeIndex, da usare come
    target.loc[idx_train], X.loc[idx_test], ecc.
    """
    n_is = len(df_is)
    confine = round(n_is * quota)

    pos = df_is.index.get_indexer(target.index)   # -1 = trigger fuori da df_is
    in_is = pos >= 0
    pos_is = pos[in_is]
    idx_is = target.index[in_is]

    # PURGE: l'uscita (trigger + horizon) deve cadere dentro df_is
    ok_is = pos_is + horizon < n_is

    # train: finestra dell'etichetta [trigger+1, trigger+horizon] tutta prima del confine
    train = ok_is & (pos_is + horizon < confine)
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
    print(f"  TRAIN: {len(idx_train):5d} righe, classe 1 {target.loc[idx_train].mean():.1%}"
          f"  ({idx_train[0].date()} -> {idx_train[-1].date()})")
    print(f"  TEST:  {len(idx_test):5d} righe, classe 1 {target.loc[idx_test].mean():.1%}"
          f"  ({idx_test[0].date()} -> {idx_test[-1].date()})")

    return idx_train, idx_test


def costruisci_X(df_lato: pd.DataFrame, target: pd.Series,
                 colonne_trigger: list) -> pd.DataFrame:
    """
    Matrice delle feature per un lato: le colonne trigger E* del lato +
    tutte le colonne di contesto F*, convertite in 0/1, una riga per ogni
    riga del target (stesso indice, stesso ordine).

    Mai colonne OHLCV: il modello vede solo "quale trigger" e "in che
    contesto", non il livello del prezzo.
    """
    colonne_f = [c for c in df_lato.columns if _PATTERN_F.match(c)]
    colonne = list(colonne_trigger) + colonne_f

    vietate = _OHLCV.intersection(colonne)
    if vietate:
        raise ValueError(f"colonne di prezzo nella X: {sorted(vietate)}")

    X = df_lato.loc[target.index, colonne]
    if X.isna().any().any():
        raise ValueError("NaN nella X: controlla il taglio del riscaldamento")

    X = X.astype(int)
    print(f"X: {X.shape[0]} righe x {X.shape[1]} colonne "
          f"({len(colonne_trigger)} trigger E* + {len(colonne_f)} contesto F*)")
    return X
