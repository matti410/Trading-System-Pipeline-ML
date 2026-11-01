# 05 · Proposta: scelta automatica del gruppo di trigger e di H

**01/11/2026**, risposta a `docs/04_GRUPPO_AUTOMATICO.md` (§7). Solo testo: il codice arriva dopo il via
di Mattia, su questo ramo (`gruppo-automatico`).

**Approvata da Mattia il 01/11/2026, con queste decisioni** (valgono sopra il testo che segue):
1. **Totale(H)** = somma, sulle barre del gruppo, di netto_bp × peso di unicità **non** normalizzato.
   La finestra è quella di `pesi.pesi_unicita`, calcolata sulle sole barre del gruppo nel train. È
   uguale a media netta × occorrenze effettive.
   - Perché: con H lunghi i trade di barre vicine si sovrappongono e la somma semplice conta più volte lo
     stesso movimento, mentre il motore tiene un trade alla volta.
   - La tabella per H mostra occorrenze, effettive, media netta e questo Totale. La regola non cambia:
     vince il Totale più alto, a parità l'H più corto.
2. **`MIN_NUOVE_PER_META = 10`**, in occorrenze grezze, come nel §3.
3. **`scelta_lato`**: se è scritto solo uno fra `TRIGGER_<LATO>` e `H_<LATO>` (l'altro è None), si ferma
   con «scrivi trigger e H insieme, oppure lasciali tutti e due a None».
4. **Test in più**: con prezzi a crescita costante, costi a zero e un trigger su 3 barre consecutive, il
   Totale conta ogni barra coperta una volta sola e non cresce per la sovrapposizione.
5. Il test «nessun numero esistente cambia» con la scelta a mano esclude solo la riga «risoluzione», che
   ora usa le effettive (docs/04 §5).

## 1 · File e funzioni

**Nuovo `gruppo.py`**: le regole del §2 e l'algoritmo del §3 di docs/04, solo sul train.
- `netti_train(ds, direzione, H, quota=0.8) -> DataFrame`
  - Chiama una volta `target.costruisci_target` sulle sole barre del train (`pipeline.barre_train`), con
    **tutti** i trigger del lato insieme.
  - Le barre sono le stesse della funzione del target: ingresso Open i+1, uscita Open i+H+1, costi del
    Lab, quarantena.
  - I prezzi del test non vengono passati, quindi un'uscita oltre il confine non esiste e il test non
    entra per costruzione.
  - Restituisce, per ogni barra con almeno un trigger: `netto_bp`, `meta` (1 o 2, a metà delle barre del
    train) e una colonna vero/falso per trigger.
  - Una barra con più trigger è una riga sola, quindi un solo trade.
- `ammessi(netti) -> DataFrame`: per trigger, le occorrenze, la media netta in tutto il train e in
  ciascuna metà, e la colonna `ammesso` (le tre medie > 0).
- `gruppo_per_H(netti, ammessi, min_per_meta) -> (gruppo, composizione)`
  - Ordine: media decrescente; a parità, più occorrenze.
  - Per ogni trigger: le occorrenze nuove (in tutto il train e per metà), la media netta di quelle
    nuove e delle due metà, poi «entra» oppure il motivo dell'esclusione.
- `scegli_H(totali) -> H | None`: il Totale più alto; a parità, l'H più corto. Se tutti i gruppi sono
  vuoti restituisce `None`: il lato è proposto vuoto e, come oggi, si salta.
- `proponi(ds, direzione, griglia_h, quota=0.8, min_per_meta=...) -> Proposta`
  - È un dataclass con: lato, griglia, la tabella per H (gruppo, occorrenze, effettive, media netta,
    Totale), l'H scelto, il gruppo e la composizione.
  - L'H va da 1 in su; un H oltre l'orizzonte dell'event study si accetta, ma con un avviso, perché la
    linea verticale cadrebbe fuori dal grafico.

**`pesi.py`** (§5)
- Nuova `occorrenze_effettive(indice, df_lato, horizon) -> float`: è la somma dei pesi di unicità
  **prima** della normalizzazione.
- `pesi_unicita` usa la stessa parte di calcolo, normalizza come oggi e quindi non cambia.

**`pipeline.py`**
- `tabella_proposta(proposta, timeframe)`: la tabella per H, con le ore accanto e l'H scelto segnato.
- `grafico_gruppo(ev, proposta, costo_bp)`: il grafico dell'event study con i soli trigger del gruppo e
  una linea verticale a H. Riusa `grafico_event_study`.
- `scelta_lato(proposta, trigger, H) -> (trigger, H, origine)`, nella cella 4. Stampa l'origine.
  - `TRIGGER_<LATO> = None`: vale la proposta → «proposta automatica».
  - Una lista scritta da Mattia → «scelta a mano»; `H_<LATO>` va scritto, come oggi.
  - `[]`: il lato si salta, come oggi.
- `riga_risoluzione(n_train, n_test, eff_train, eff_test, piu_meno, costo_bp)`
  - Stampa le grezze e le effettive.
  - Il ± atteso usa le effettive del test; le frasi «!!!» usano le effettive.
- `verifica_gruppo(lato) -> DataFrame` (passo 5): la media netta del gruppo sul train e sul test, con il
  ± di ciascuno (`oos.piu_meno_bp`).
  - Avviso «!!!» se il test è sotto il train di più del ± della differenza, cioè della radice di
    (±train² + ±test²).
- `tabella_trigger_train`: aggiunge la colonna `effettivi` (somma dei pesi non normalizzati delle righe
  del trigger).

**`registro.py`**: un nuovo tipo di voce, «gruppo». Contiene lato, griglia, minimo per metà, ammessi per
H, gruppo per H, Totali, H scelto, proposta, scelta finale, origine (proposta automatica / scelta a
mano) e il segno di collaudo.

**Notebook**
- Passo 4: si scrivono `GRIGLIA_H = [4, 8, 12, 16, 24, 32]` e `MIN_NUOVE_PER_META = 10` (vedi §3), con
  le ore stampate accanto ma senza effetto sulla scelta. Poi, per lato, la tabella per H, la
  composizione e il grafico.
- Cella 4: `TRIGGER_LONG = None` e `H_LONG = None` di default.
- Passo 5: la riga «risoluzione» con le effettive e la verifica onesta.
- Sullo spot si calcola solo il LONG.

**Test (`tests/test_gruppo.py`, più due casi in `test_split_pesi.py` e uno in `test_registro.py`)**
- I sette test del §6 di docs/04.
- Un test in più, proposto: i netti di `netti_train` coincidono con quelli di `costruisci_target` su
  tutto il df per gli stessi trade.

**Da sapere**
- Il gruppo si sceglie su tutti i trigger del train, come l'event study.
- I trigger nel riscaldamento delle feature si tolgono dopo, in `prepara_lato`. Per questo i numeri del
  passo 5 possono differire di poco da quelli della proposta.

## 2 · Tempi di calcolo

Ho misurato un prototipo usa e getta, fuori dal repo, su BTC perp di Binance, con 37 trigger LONG, 36
SHORT e 6 H nella griglia. Il dataset era già aperto: l'apertura serve comunque al notebook.

| | barre IS | per lato | LONG + SHORT |
|---|---|---|---|
| M15 | 190.906 | circa 7 s | circa 15 s |
| M5 | 573.318 | circa 20 s | circa 40 s |

- Quasi tutto il tempo va nel calcolo dei costi del Lab, una volta per H. Il gruppo e le effettive
  costano meno di un secondo.
- Ogni H in più nella griglia aggiunge circa 1 s su M15 e circa 3 s su M5 per lato.
- Su Colab il tempo può essere circa il doppio (stima, non misurato).

## 3 · La domanda aperta: un minimo di occorrenze nuove per metà?

**Pro**
- Senza minimo, un trigger con 2 occorrenze nuove in una metà supera la regola delle due metà se uno
  solo dei due trade è buono. La regola nata per la stabilità si decide allora sul caso.
- Un minimo mette un pavimento fisso, scritto prima di guardare, e non costa tempo.

**Contro**
- È un numero in più, ed è arbitrario: 10 non è migliore di 8 o 15.
- Esclude i trigger rari, anche quelli buoni. Il peso di quei trigger sul Totale è però piccolo per
  forza: 10 trade da +20 bp valgono 200 bp.
- Su timeframe lenti (H4) le occorrenze sono poche e il minimo pesa di più.
- Non risolve il rischio 5 di docs/04 §1, il migliore fra tanti: lo riduce soltanto.

**La mia risposta: sì, `MIN_NUOVE_PER_META = 10` in occorrenze grezze** (approvata il 01/11/2026)
- Si applica solo al passo 3, alle occorrenze nuove: un trigger entra se ha almeno 10 occorrenze nuove
  in **ciascuna** metà, oltre alle medie > 0.
- Il primo trigger del gruppo ha tutte le sue occorrenze come nuove, quindi il minimo vale anche per
  lui.
- È scritto nel notebook accanto a `GRIGLIA_H`; con 0 si torna alla regola senza minimo.
- Uso le occorrenze grezze e non le effettive perché sono più semplici da leggere e da controllare a
  mano.
- Decide Mattia.

## 4 · Comandi e cosa si vede (dopo il codice)

- `.venv\Scripts\python -m pytest`: tutti i test verdi, compresi quelli nuovi. Il test lento
  (`-m lento`) non serve, perché `selezione.py` e `modello.py` non cambiano.
- Nel notebook, Avvio → passo 4 con la cella 4 lasciata così com'è (`None`):
  - per lato, la tabella per H con l'H scelto segnato e la composizione del gruppo;
  - il grafico con le sole curve del gruppo e la linea a H;
  - poi «LONG: proposta automatica, N trigger, H = … barre (… ore)».
  - Un lato con tutti i gruppi vuoti stampa «nessun trigger ammesso: lato saltato».
- Scrivendo `TRIGGER_LONG = ["…"]` e `H_LONG = 16` si vede «scelta a mano». I numeri dei passi dopo
  sono identici a quelli di oggi con la stessa scelta; cambia solo la riga «risoluzione», che ora ha le
  effettive (§5 di docs/04).
- Passo 5: la riga «risoluzione» con grezze ed effettive, la verifica del gruppo train/test con i ± e
  l'eventuale «!!!».
- `registro.stampa_conto` conta anche le voci «gruppo».
