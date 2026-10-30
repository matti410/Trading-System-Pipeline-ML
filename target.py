"""
Il target del meta-labeling: la regola d'uscita del backtest del Lab (30/10/2026, docs/02 §A e §F).

  trade_dei_trigger(df, colonne_trigger, direction, horizon, *, costi, quarantena=True)
      -> DataFrame nel formato dei trade del Lab (Size, EntryBar, ExitBar, ...)
  costruisci_target(df, colonne_trigger, direction, horizon, *, costi, quarantena=True)
      -> DataFrame: TARGET (0/1), lordo_bp, netto_bp, ingresso, uscita

LA REGOLA (decisione di Mattia del 30/10/2026, D5)
--------------------------------------------------
E' quella del motore veloce di V4 con uscita a tempo n_barre = H:
  - trigger sulla barra s;
  - ingresso all'Open della barra s+1 (con lo spread del motore dentro il
    prezzo, che sugli exchange e' 0: lo spread di un passo e' un costo a parte);
  - il motore chiede la chiusura alla barra s+H, che si riempie all'Open della
    barra s+H+1. Durata: ExitBar - EntryBar = H barre.
Cosi' target = backtest = Lab, senza tolleranze (test di identita' esatta).

Esclusi, come nel motore: il trigger sulla barra 0 (il motore non la guarda),
quelli senza la barra s+H+1 (trade ancora aperto alla fine) e quelli in
quarantena (quarantena_trade.maschera_segnali_puliti: barra del segnale o
d'ingresso in quarantena).

I COSTI: la stessa catena del Lab (engine/exit_search_bt.py), nello stesso ordine
  costo_extra_pips -> aggiungi_funding -> aggiungi_spread_passo ->
  metriche.netti_in_unita(..., unita="bp")
con commissione taker per lato, spread di un passo e, sui perpetual, il
funding di ogni liquidazione fra ingresso e uscita. TARGET = 1 se il netto in
bp e' sopra zero; il pareggio esatto vale 0.

A differenza del backtest, qui ogni trigger ha il suo trade anche se si
sovrappone a un altro: l'etichetta e' del trigger. Il motore, che tiene un
trade alla volta, prende un sottoinsieme di questi, con gli stessi numeri.
"""
import numpy as np
import pandas as pd


def _posizioni_trigger(df, colonne_trigger, horizon, quarantena):
    from engine.quarantena_trade import maschera_segnali_puliti
    if horizon < 1:
        raise ValueError("horizon deve essere >= 1")
    mancanti = [c for c in colonne_trigger if c not in df.columns]
    if mancanti:
        raise KeyError(f"colonne trigger non presenti nel df: {mancanti}")
    segnale = df[list(colonne_trigger)].astype(bool).any(axis=1).to_numpy() & maschera_segnali_puliti(df, quarantena)
    pos = np.flatnonzero(segnale)
    n = len(df)
    return pos[(pos >= 1) & (pos + horizon + 1 < n)]


def trade_dei_trigger(df, colonne_trigger, direction, horizon, *, costi, quarantena=True):
    """
    Un trade per trigger, con le colonne dei trade del motore (stessi tipi), e
    il costo del Lab gia' aggiunto (CostoExtraPips e, sui perpetual, FundingPips).
    L'indice e' la barra del trigger.
    """
    from engine.exchange import aggiungi_funding, aggiungi_spread_passo
    from engine.quarantena_trade import costo_extra_pips

    if direction not in (1, -1):
        raise ValueError("direction deve essere 1 (LONG) o -1 (SHORT)")
    pos = _posizioni_trigger(df, colonne_trigger, int(horizon), quarantena)
    entrata, uscita = pos + 1, pos + int(horizon) + 1
    apertura = df["Open"].to_numpy(np.float64)
    spread = float(costi.get("spread", 0.0))
    idx = df.index
    trades = pd.DataFrame({
        "Size": [float(direction)] * len(pos),
        "EntryBar": entrata.tolist(),
        "ExitBar": uscita.tolist(),
        # come il motore: o[i] * (1 + copysign(spread, entra)); spread 0 sugli exchange
        "EntryPrice": (apertura[entrata] * (1 + np.copysign(spread, direction))).tolist(),
        "ExitPrice": apertura[uscita].tolist(),
        "EntryTime": [idx[i] for i in entrata],
        "ExitTime": [idx[i] for i in uscita],
    }, index=idx[pos])
    pip = float(costi["pip_size"])
    trades = costo_extra_pips(trades, df, quarantena, spread, costi.get("spread_rollover_pips"), pip)
    trades = aggiungi_funding(trades, df, costi.get("funding"), pip)
    trades = aggiungi_spread_passo(trades, costi.get("spread_passo_pips"))
    return trades


def costruisci_target(df, colonne_trigger, direction, horizon, *, costi, quarantena=True):
    """
    Il target di un lato. Parametri:

    df              le barre del dataset (dataset.apri(...).df), con le colonne trigger
    colonne_trigger i trigger del lato scelti da Mattia (colonne bool)
    direction       1 = LONG, -1 = SHORT
    horizon         H del lato, in barre (obbligatorio)
    costi           il dizionario dei costi del Lab (dataset.apri(...).costi)
    quarantena      come nel Lab: True = trigger in quarantena scartati

    Ritorna un DataFrame indicizzato sulla barra del trigger, con TARGET (0/1),
    lordo_bp, netto_bp, ingresso e uscita (i prezzi del trade).
    """
    from engine.metriche import netti_in_unita
    trades = trade_dei_trigger(df, colonne_trigger, direction, horizon, costi=costi, quarantena=quarantena)
    lordi, netti = netti_in_unita(trades, float(costi["pip_size"]), float(costi["commission"]), "bp")
    out = pd.DataFrame({"TARGET": (netti > 0).astype(int), "lordo_bp": lordi, "netto_bp": netti,
                        "ingresso": trades["EntryPrice"], "uscita": trades["ExitPrice"]},
                       index=trades.index)
    out.index.name = df.index.name
    return out
