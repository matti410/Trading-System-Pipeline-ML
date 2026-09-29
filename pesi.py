"""
Pesi di unicita' dei trigger (decisione A del 28/9, attivata il 29/9 come
interruttore dopo il modello di base).

Modulo ADDITIVO: non modifica nessun file esistente.

  pesi_unicita(indice, df_lato, horizon=25) -> Series di pesi, media 1

IDEA
----
L'esito di un trigger si decide sulle `horizon` barre successive. Se piu'
trigger scattano vicini, le loro finestre si sovrappongono e i loro esiti
sono quasi lo stesso esito ripetuto. Il peso dice al modello quanto contare
ogni riga:

  - per ogni barra si conta quante finestre di trigger la coprono (n)
  - il peso di un trigger e' la media di 1/n sulle sue barre
  - trigger isolato -> 1 ; due trigger con la stessa finestra -> 0.5 ciascuno

I pesi vanno calcolati SOLO sull'indice di train (idx_train): cosi'
dipendono solo dai trigger di train e non dal test. Alla fine si
normalizzano a media 1, cosi' il totale "vale" quanto le righe.
Il peso NON e' una feature: si passa al modello come importanza della riga.
"""
import numpy as np
import pandas as pd


def pesi_unicita(indice, df_lato, horizon=25):
    pos = df_lato.index.get_indexer(indice)
    n = len(df_lato)
    if (pos < 0).any() or (pos + horizon >= n).any():
        raise ValueError("indice con trigger assenti o senza horizon barre in avanti")

    # finestra di ogni trigger: barre pos+1 ... pos+horizon (come il target)
    conteggio = np.zeros(n + 1)
    np.add.at(conteggio, pos + 1, 1)
    np.add.at(conteggio, pos + horizon + 1, -1)
    conteggio = np.cumsum(conteggio)[:n]          # finestre aperte su ogni barra

    inverso = np.where(conteggio > 0, 1.0 / np.maximum(conteggio, 1), 0.0)
    somma = np.concatenate([[0.0], np.cumsum(inverso)])
    peso = (somma[pos + horizon + 1] - somma[pos + 1]) / horizon

    peso = peso / peso.mean()
    return pd.Series(peso, index=indice, name="peso")
