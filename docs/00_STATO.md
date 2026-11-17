# 00 · Stato (aggiornato al 17/11/2026)

**Incarico in corso:** i trigger generati di V4 (docs/83 di V4, `dati_catalogo/generati_v1.json`)
come candidati del gruppo automatico, interruttore `TRIGGER_GENERATI`. Ramo `trigger-generati`, PR in
bozza.

## Cosa c'è (main, PR #1–#11)

| file | cosa fa |
|---|---|
| `collegamento_v4.py`, `dataset.py` | V4 in sola lettura: dati, confine dello studio, costi, trigger (anche i generati), filtri |
| `gruppo.py` | proposta automatica del gruppo di trigger e di H per lato (docs/04) |
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

## Trigger generati (`TRIGGER_GENERATI`, 17/11/2026)

- Spento di base: il notebook è quello di prima.
- Acceso: `dataset.apri` carica e registra il catalogo generato con le funzioni di V4
  (`engine.generatore.catalogo`) e aggiunge i suoi trigger a quelli del Lab: 1.492 long e 1.492 short
  (112 ricette), gli short solo se il symbol non è solo long. Le colonne si calcolano come le altre.
- Il gruppo automatico (`gruppo.py`) non cambia: riceve più candidati, con le stesse regole.
- L'event study del passo 3 resta sui soli trigger del Lab. Nella composizione del passo 4, dei
  generati si vedono solo quelli del gruppo, più la riga «GENERATI (riepilogo)» con i provati, gli
  ammessi e quelli nel gruppo. Il registro scrive i trigger provati e la versione del catalogo.
- M1 e D1 non sono supportati né dal Lab né dal ML (`vetrina.TIMEFRAMES`: M5, M15, M30, H1, H4).

**Tempo e RAM** di `dataset.apri` + gruppo automatico dei due lati (griglia del notebook), BINANCE_PERP_BTCUSDT,
PC di Mattia (16 GB), 17/11/2026. «Picco» = memoria più alta usata dal processo.

| TF | barre | generati | `dataset.apri` | gruppo (2 lati) | totale | picco RAM |
|---|---:|---|---:|---:|---:|---:|
| H4 | 14.759 | spenti | 2 s | 3 s | 4 s | 0,29 GB |
| H4 | 14.759 | accesi | 15 s | 6 s | 21 s | 0,42 GB |
| M15 | 240.622 | spenti | 25 s | 16 s | 41 s | 0,67 GB |
| M15 | 240.622 | accesi | 3 min 32 s | 47 s | 4 min 19 s | 2,59 GB |
| M5 | 719.280 | spenti | 1 min 17 s | 43 s | 2 min 0 s | 1,59 GB |
| M5 | 719.280 | accesi | 10 min 35 s | 2 min 19 s | 12 min 54 s | 6,95 GB |

Tutto sta in memoria: i generati restano colonne del dataset, come gli altri trigger (il calcolo uno alla
volta, previsto se un timeframe non ci stava, non è servito).

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
- Scelta automatica del gruppo di trigger e di H (`docs/04`), con le decisioni in testa a `docs/05`:
  - Totale(H) = somma di netto_bp × peso di unicità non normalizzato;
  - `MIN_NUOVE_PER_META = 10`.

**17/11/2026**
- Trigger generati di V4 come candidati del gruppo automatico, con `TRIGGER_GENERATI = False` di base.
- Misure di tempo e RAM su M15, M5 e H4; M1 e D1 non sono supportati da Lab e ML.

## Valori di collaudo, non decisioni
- I trigger e gli H usati nei collaudi.
- Fascia operativa 5.
- `PARAMETRI_BASE` di LightGBM.

La storia è nel Project di Mattia, non nel repo.
