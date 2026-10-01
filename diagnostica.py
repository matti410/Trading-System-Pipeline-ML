"""
Diagnostica delle condizioni (1/10/2026).

  sovrapposizioni(df, colonne, finestra=0, soglia=0.0) -> DataFrame

Serve a vedere, ogni volta che si aggiunge un filtro F* o un trigger E*,
se duplica qualcosa che c'e' gia'. Un doppione non da' informazione nuova
al modello, ma gli da' un'occasione in piu' di trovare regole casuali nel
train e divide l'importanza fra due colonne.

Una riga per ogni coppia (A, B) di colonne booleane:

  n_A, n_B      quante barre sono True
  A_dentro_B_%  quota delle volte in cui A e' True che B e' True anche lei
  B_dentro_A_%  viceversa. Uno dei due vicino a 100 = una e' contenuta
                nell'altra
  max_dentro_%  il maggiore dei due, usato per ordinare e per la soglia
  corr          correlazione fra le due colonne 0/1 su tutte le barre.
                +1 = identiche, -1 = una e' il contrario esatto dell'altra

`finestra` serve per i trigger E*, che sono eventi rari: con finestra=0
"B e' True" vuol dire sulla stessa barra; con finestra=2 vuol dire entro
2 barre prima o dopo. Per i filtri F* (stati, non eventi) si lascia a 0.
La correlazione si calcola sempre sulla stessa barra.

Ordinamento: dal piu' sovrapposto, usando il maggiore fra max_dentro_% e
100*|corr| (cosi' in cima finiscono sia i sottoinsiemi sia i contrari
esatti, che hanno max_dentro_% = 0 ma corr = -1). `soglia` (in %) tiene
solo le coppie con quel valore >= soglia, per non stampare centinaia di
righe.

Colonne mai True vengono segnalate e saltate.
"""
import numpy as np
import pandas as pd


def sovrapposizioni(df, colonne, finestra=0, soglia=0.0):
    colonne = list(colonne)
    mancanti = [c for c in colonne if c not in df.columns]
    if mancanti:
        raise KeyError(f"colonne non presenti nel df: {mancanti}")
    if finestra < 0:
        raise ValueError("finestra deve essere >= 0")

    X = df[colonne].fillna(False).astype(bool)
    n = X.sum()
    vuote = [c for c in colonne if n[c] == 0]
    if vuote:
        print(f"[sovrapposizioni] mai True, saltate: {vuote}")
    colonne = [c for c in colonne if n[c] > 0]

    vicino = X if finestra == 0 else (
        X.astype(int).rolling(2 * finestra + 1, center=True, min_periods=1).max().astype(bool)
    )
    corr = X[colonne].astype(float).corr()

    righe = []
    for i, a in enumerate(colonne):
        for b in colonne[i + 1:]:
            a_in_b = 100 * (X[a] & vicino[b]).sum() / n[a]
            b_in_a = 100 * (X[b] & vicino[a]).sum() / n[b]
            righe.append({
                "A": a, "B": b, "n_A": int(n[a]), "n_B": int(n[b]),
                "A_dentro_B_%": a_in_b, "B_dentro_A_%": b_in_a,
                "max_dentro_%": max(a_in_b, b_in_a),
                "corr": corr.loc[a, b],
            })

    tab = pd.DataFrame(righe, columns=["A", "B", "n_A", "n_B", "A_dentro_B_%",
                                       "B_dentro_A_%", "max_dentro_%", "corr"])
    # punteggio unico per ordinare: il piu' forte fra contenimento e |corr|,
    # cosi' in cima finiscono sia i sottoinsiemi sia i contrari esatti
    punteggio = np.maximum(tab["max_dentro_%"], 100 * tab["corr"].abs())
    tab = tab[punteggio >= soglia].assign(_p=punteggio)
    tab = tab.sort_values("_p", ascending=False).drop(columns="_p")
    return tab.reset_index(drop=True).round(2)
