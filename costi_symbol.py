"""
Costi di trading per symbol - conto IC Markets EU STANDARD (29/9/2026).

Una riga per strumento. Nel notebook si scrive solo `symbol`: target e
backtest leggono da qui spread, commissione, dimensione del pip e del lotto.

  costi(symbol) -> dict con pip_size, spread_pips, commissione_pips, lotto,
                   costo_pips (= spread + commissione, andata e ritorno)

Campi di ogni riga:
  pip_size         : quanto vale 1 pip in unita' di prezzo
  spread_pips      : spread medio, in pips (si paga una volta per trade)
  commissione_pips : commissione andata e ritorno, in pips (0 sul conto Standard)
  lotto            : unita' dello strumento in 1 lotto (contract size)

Valgono solo per strumenti quotati in USD (conto in USD): su coppie come
GBPJPY il profitto e' in un'altra valuta e andrebbe convertito.

Fonti:
  - spread medi conto Standard: icmarkets.eu/en/trading-pricing/spreads-and-swaps
  - commissione Standard = 0: icmarkets.eu/en/trading-accounts/overview
    (confermato dal report MT5 del conto demo: commissioni a 0 su tutti i trade)
  - BTCUSD: scheda crypto IC Markets EU (Cryptocurrency-Specification-Sheet.pdf).
    Convenzione: 1 pip = 1 USD di prezzo. Spread 6.46 DA VERIFICARE in MT5
    (Finestra di Mercato, colonna Spread in punti: 646 punti = 6.46 USD).
  - lotto: confermato dal report MT5 (profitto / differenza di prezzo).

Per aggiungere uno strumento basta aggiungere una riga a COSTI_SYMBOL.
"""

COSTI_SYMBOL = {
    "EURUSD": {"pip_size": 0.0001, "spread_pips": 0.80, "commissione_pips": 0.0, "lotto": 100_000},
    "BTCUSD": {"pip_size": 1.0,    "spread_pips": 6.46, "commissione_pips": 0.0, "lotto": 1},
}


def costi(symbol):
    if symbol not in COSTI_SYMBOL:
        raise KeyError(
            f"'{symbol}' non e' in costi_symbol.COSTI_SYMBOL. "
            f"Strumenti presenti: {', '.join(COSTI_SYMBOL)}. "
            "Aggiungi una riga con pip_size, spread_pips, commissione_pips, lotto."
        )
    riga = dict(COSTI_SYMBOL[symbol])
    riga["costo_pips"] = riga["spread_pips"] + riga["commissione_pips"]
    return riga
