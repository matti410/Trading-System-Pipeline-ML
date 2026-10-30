# 00 · Stato (aggiornato al 30/10/2026)

**Incarico in corso:** **lavoro B** (pipeline base crypto) di `docs/02_PROPOSTA_CONVERSIONE.md`, §B e §F:
codice sul ramo `lavoro-b-pipeline`, in revisione (chiusura in docs/02 §G).
Il lavoro A (pulizia e collegamento a V4) è fatto: PR #1, unita. C e D vengono dopo, ognuno con il suo via.

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
| `backtest_oos.py` | backtest sul motore veloce di V4 (lavoro B) |
| `pipeline.py` | le funzioni delle celle del notebook (lavoro B) |
| `confronti.py` | i due confronti fissi sul test, D3 (lavoro B) |
| `registro.py` | il registro delle prove in `registro_ml/`, D4 (lavoro B) |

Con il lavoro A:
- tolte le copie V3 (`engine/`, entry, filtri, `helpers.py`), i costi IC Markets e il CSV EURUSD;
- nuovi `collegamento_v4.py` (V4 in sola lettura, con ramo e commit stampati) e `dataset.py` (dati,
  confine, costi, trigger e filtri presi da V4);
- `target.py` usa la regola del motore veloce di V4 e i costi del Lab;
- nuovi `requirements.txt` e `tests/`;
- il notebook ha solo le celle 0–2. Con il lavoro B ha la sequenza in 9 passi di docs/02 §B.

## Decisioni di Mattia (30/10/2026)
1. **Target (a)**: si guarda l'esito a H barre. Dal 30/10 (docs/02 §F, D5) l'uscita è all'Open della
   barra i+H+1, come il motore veloce di V4.
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
- nel collaudo del lavoro B: il primo trigger long e il primo short dell'elenco del Lab, `H_LONG = 16`,
  `H_SHORT = 32`;
- fascia operativa 5;
- `PARAMETRI_BASE` di LightGBM.

La storia del ramo EURUSD è nel Project di Mattia, non nel repo.
