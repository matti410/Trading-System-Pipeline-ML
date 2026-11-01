# 00 · Stato (aggiornato al 01/11/2026)

**Incarico in corso:** `docs/04_GRUPPO_AUTOMATICO.md`, scelta automatica del gruppo di trigger e di
H per lato. Si parte dalla proposta, senza codice.

## Cosa c'è (main, PR #1–#10)

| file | cosa fa |
|---|---|
| `collegamento_v4.py`, `dataset.py` | V4 in sola lettura: dati, confine dello studio, costi, trigger, filtri |
| `target.py` | target = motore veloce di V4: ingresso Open i+1, uscita Open i+H+1, costi del Lab |
| `split.py`, `pesi.py` | train/test con purge ed embargo; pesi di unicità e per rendimento |
| `feature/` | catalogo di 73 feature di contesto (finestre in ore) e collaudi |
| `selezione.py` | feature selection: 5 blocchi, sonde, regola B, K |
| `modello.py`, `oos.py`, `confronti.py` | LightGBM, fasce con ±bp, soglia dal test, confronti D3 |
| `backtest_oos.py` | backtest sul motore veloce di V4, con il buy & hold |
| `pipeline.py` | le funzioni delle celle del notebook |
| `registro.py` | registro delle prove in `registro_ml/` (accanto allo studio, mai dentro) |

**Il notebook `TS-Pipeline-ML.ipynb`:**
1. dataset;
2. confine e feature;
3. event study sul train, con i primi 10 con picco > 0 e la linea del costo;
4. scelta dei trigger e di H; un lato vuoto si salta;
5. target, divisione e riga «risoluzione»;
6. modello base (passo 7);
7. selezione e confronto (passo 7b);
8. soglia e confronti (passo 8);
9. OOS e backtest (passo 9), solo con `GUARDA_OOS = True`.

## Decisioni di Mattia

**30/10/2026**
- Target (a): uscita a tempo, senza SL e TP.
- Un modello per lato, con un H per lato.
- Solo crypto.
- LightGBM come modello.
- Catalogo di feature unico per tutti; selezione fatta per dataset e lato.

**31/10/2026**
- `PESO_RENDIMENTO = True` di default (per principio).
- `GUARDA_OOS = False` di default: l'OOS si apre una volta, a scelta fatta.
- Colonna ±bp in tutte le tabelle.
- Le leve di miglioramento sono in `docs/03`, da leggere solo se il lavoro lo chiede.

**01/11/2026**
- Riga «risoluzione» al passo 5: il ± atteso viene dal solo train.
- Scelta automatica del gruppo di trigger e di H (`docs/04`).

## Valori di collaudo, non decisioni
- I trigger e gli H usati nei collaudi.
- Fascia operativa 5.
- `PARAMETRI_BASE` di LightGBM.

La storia è nel Project di Mattia, non nel repo.
