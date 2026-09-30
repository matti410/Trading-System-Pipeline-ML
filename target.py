import numpy as np
import pandas as pd


def costruisci_target(df_lato, colonne_trigger, direction, horizon=25, *, costo_pips, pip_size):
    """
    Costruisce il target binario (meta-labeling) per un lato (LONG o SHORT).

    Per ogni barra in cui scatta almeno uno dei trigger indicati, calcola il
    rendimento realizzato tra l'ingresso (Open della barra successiva al
    trigger) e l'uscita (Close a `horizon` barre di distanza dalla barra di
    trigger, stesso riferimento usato nell'event study), nel verso di
    `direction`. TARGET=1 se il rendimento supera il costo del trade, 0
    altrimenti.

    Il costo e' in pips (spread + commissione, da costi_symbol.costi) e viene
    convertito in rendimento trigger per trigger, sul prezzo di ingresso:
        soglia = costo_pips * pip_size / prezzo_ingresso
    Le barre MT5 sono in prezzo bid: un long compra all'ask ed esce al bid,
    uno short vende al bid e ricompra all'ask, quindi lo spread si paga una
    volta per trade in entrambi i versi.

    Nessun margine di embargo IS/OOS viene applicato qui: va fatto a parte,
    quando si costruisce lo split, scartando le righe vicino al confine.

    Parametri
    ---------
    df_lato : DataFrame con colonne 'Open', 'Close', le colonne trigger e le
        feature di contesto (es. df_LONG o df_SHORT). Indice temporale
        ordinato, senza gap, come prodotto dal notebook.
    colonne_trigger : lista di nomi colonna booleani (i 4 flag E* del lato).
    direction : 1 per LONG, -1 per SHORT.
    horizon : barre dopo il trigger su cui misurare l'uscita (default 25).
    costo_pips : costo andata e ritorno in pips (spread + commissione).
        Obbligatorio, da passare per nome: COSTI["costo_pips"].
    pip_size : valore di 1 pip in unita' di prezzo. Obbligatorio, da passare
        per nome: COSTI["pip_size"].

    Ritorna
    -------
    pd.Series di 0/1 (int), indicizzata solo sulle barre trigger che hanno
    `horizon` barre complete disponibili in avanti. name='TARGET'.
    """
    if direction not in (1, -1):
        raise ValueError("direction deve essere 1 (LONG) o -1 (SHORT)")

    is_trigger = df_lato[colonne_trigger].any(axis=1)
    trigger_idx = df_lato.index[is_trigger]

    n = len(df_lato)
    pos = df_lato.index.get_indexer(trigger_idx)  # posizione intera di ogni riga trigger

    entry_pos = pos + 1        # barra successiva al trigger
    exit_pos = pos + horizon   # barra a `horizon` dal trigger

    valido = exit_pos < n      # scarta le righe senza storia futura sufficiente

    open_arr = df_lato['Open'].to_numpy()
    close_arr = df_lato['Close'].to_numpy()

    entry_price = open_arr[entry_pos[valido]]
    exit_price = close_arr[exit_pos[valido]]

    # guadagno > costo, cioe' (exit/entry - 1)*direction > costo_pips*pip_size/entry.
    # Il confronto si fa in unita' di prezzo (moltiplicando per entry), che e'
    # la stessa disuguaglianza senza la divisione: i prezzi hanno pochi
    # decimali e un guadagno pari al costo (es. esattamente 8 punti = 0.8 pips)
    # capita spesso; con la divisione finiva a caso in 0 o 1 per
    # arrotondamento. Pareggio esatto -> 0 (il trade non supera il costo).
    # Tolleranza relativa al prezzo -> vale per qualunque symbol (anche BTCUSD).
    netto = (exit_price - entry_price) * direction - costo_pips * pip_size
    target = (netto > 1e-9 * entry_price).astype(int)

    return pd.Series(target, index=trigger_idx[valido], name='TARGET')
