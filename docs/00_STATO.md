# 00 · Stato (aggiornato al 30/10/2026)

**Incarico in corso:** `docs/01_CONVERSIONE_CRYPTO.md`, in attesa della proposta
`docs/02_PROPOSTA_CONVERSIONE.md`.

## Com'è il repo oggi
Una prima bozza fatta su EURUSD M15 con una copia del vecchio motore (V3).

I moduli ML funzionano e sono collaudati su EURUSD:

| file | cosa fa |
|---|---|
| `target.py` | il target del trigger |
| `split.py` | divisione train/test, con purge ed embargo |
| `modello.py` | LightGBM e valutazione per fasce di probabilità |
| `pesi.py` | pesi di unicità |
| `oos.py` | soglia ricavata dal test e applicata all'OOS |
| `backtest_oos.py` | backtest con VectorBT |

Il resto è materiale V3 o MT5 da togliere, con l'elenco preciso nell'incarico:
- copie di engine, entry e filtri;
- costi IC Markets in pips;
- CSV EURUSD;
- celle per MT5 e per Colab.

## Decisioni di Mattia (30/10/2026)
1. **Target (a)**: si guarda la chiusura a H barre.
2. **Un modello per lato.** Ogni lato ha il suo H, scelto da Mattia dall'event study.
3. **Solo crypto**, come V4. Dati, costi e trigger vengono da V4.
4. **Feature:** un catalogo ampio di feature di contesto, compresi i rendimenti passati,
   e poi una feature selection.
   - Il catalogo è uguale per tutti.
   - La selezione si fa per dataset (symbol × timeframe × lato).
5. Il modello base resta **LightGBM**. Gli altri modelli si confrontano più avanti.

## Ordine dei lavori
Ogni lavoro ha la sua PR.
- A: pulizia e collegamento a V4.
- B: pipeline base crypto.
- C: catalogo delle feature.
- D: feature selection.

## Valori di collaudo, non decisioni
Questi valori non sono motivati e servono solo a far girare il codice:
- H = 25 del vecchio notebook;
- fascia operativa 5;
- `PARAMETRI_BASE` di LightGBM.

La storia del ramo EURUSD è nel Project di Mattia, non nel repo.
