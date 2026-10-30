# 01 · Conversione a crypto, pulizia e feature di contesto

**Incarico del 30/10/2026**, preparato da Mattia e dal revisore per Claude Code.

**Flusso:**
1. proposta in testo in `docs/02_PROPOSTA_CONVERSIONE.md`, senza codice;
2. revisione;
3. via di Mattia;
4. codice, con una PR per ogni lavoro.

**Indice:** §1 Lavoro A · §2 Lavoro B · §3 Lavoro C · §4 Lavoro D · §5 Domande aperte ·
§6 Cosa contiene la proposta

## §1 · Lavoro A · Pulizia e collegamento a V4

**Da togliere** (verifica ogni voce e conferma l'elenco nella proposta):
- `EURUSD_M15.csv`;
- `costi_symbol.py`;
- la cartella `engine/` (copie V3);
- `entry_long.py`, `entry_short.py`, `filter_conditions.py`;
- `helpers.py`, se non serve;
- nel notebook: le celle MT5, IC Markets e token di Colab, e ogni riferimento ai pips
  IC Markets.

**Da prendere da V4.** Trova le funzioni esatte e scrivile nella proposta:
- dati: `engine/exchange.py` e `engine/aggiorna_dati.py`, come la schermata 1 del Lab;
- symbol: `engine/simboli.py`;
- costi: `exchange.costi_exchange` e il modo in cui il Lab li applica a ogni trade,
  perpetual compresi;
- catalogo dei trigger e dei filtri F*, con il registry;
- event study;
- confine dell'in-sample nello `studio/`, in sola lettura;
- quarantena, se il Lab la applica agli ingressi.

**Collegamento.** V4 sta in una cartella sorella. Proponi il modo più semplice e robusto per
chi non è informatico. Il notebook deve stampare la versione di V4 usata (il commit).

**Restano e si adattano:** `target`, `split`, `modello`, `pesi`, `oos`, `backtest_oos`,
`diagnostica`.

**Prove:**
- il target su dati inventati con costi noti dà le etichette calcolate a mano;
- i costi sono identici a quelli del Lab sullo stesso symbol.

## §2 · Lavoro B · Pipeline base crypto (si rifà `TS-Pipeline-ML.ipynb`)

**Sequenza del notebook:**
1. scelta del dataset e aggiornamento dei dati;
2. confine IS/OOS preso da V4;
3. event study (periodo da decidere: vedi D2);
4. Mattia scrive la lista dei trigger per lato e sceglie `H_LONG` e `H_SHORT`. La selezione
   automatica di oggi (`l1`/`s1` prime due righe) si toglie;
5. target;
6. divisione train/test, con purge ed embargo calcolati sull'H del lato;
7. modello base con le sole F* di V4, come oggi: è la base di riferimento;
8. soglia per fasce, ricavata dal test;
9. OOS e backtest.

**Sullo spot:** vedi D6.

## §3 · Lavoro C · Catalogo delle feature di contesto

**Requisiti:**
- causali;
- adimensionali;
- finestre fissate prima di guardare i dati, scritte in ore e convertite in barre;
- nessun parametro ottimizzato sui dati;
- un dizionario come i cataloghi di V4, con nome, famiglia e finestra per ogni feature;
- test di lookahead e di degenerazione di V4.

Il codice vive in questo repo. Gli indicatori che V4 ha già si importano, non si riscrivono.

**Famiglie di partenza.** La proposta dice quali feature tenere, con la formula in una riga
per ciascuna:

1. **Rendimenti passati**
   - singoli ritardi r(t−k), per k = 1…N;
   - rendimenti cumulati su più finestre, divisi per la volatilità della stessa finestra.
2. **Volatilità e regime**
   - volatilità realizzata su più finestre;
   - stimatori su OHLC (Parkinson, Garman-Klass);
   - rapporto fra volatilità breve e lunga;
   - percentile della volatilità.
3. **Efficienza e persistenza**
   - efficiency ratio di Kaufman;
   - variance ratio;
   - autocorrelazione dei rendimenti;
   - pendenza e R² della regressione del prezzo.
4. **Posizione nel range**
   - dove sta il Close nel range delle ultime N barre;
   - distanza dal massimo e dal minimo, in unità di volatilità;
   - barre trascorse dall'ultimo massimo e dall'ultimo minimo.
5. **Distanze**: dal VWAP rolling e dalle medie, in unità di volatilità.
6. **Forma della distribuzione**: asimmetria e curtosi dei rendimenti.
7. **Volume**
   - volume relativo alla stessa ora dei giorni precedenti;
   - z-score del volume;
   - rapporto fra volume e movimento;
   - taker buy ratio, quando V4 lo avrà (docs/77 di V4, lavoro 4).
8. **Candela**: corpo/range e ombre/range, sulla barra del trigger e sulle precedenti.
9. **Tempo**: ora UTC (seno e coseno) e giorno della settimana.
10. **Cross-asset** (solo altcoin): rendimenti e volatilità di BTC, correlazione rolling
    con BTC.
11. **Densità dei trigger**: quanti trigger del lato nelle ultime N barre, e quante barre
    dal trigger precedente.

**In più:** le colonne F* di V4 e l'identità del trigger (quale trigger è scattato).

## §4 · Lavoro D · Feature selection

**Vincoli:**
- si fa solo dentro il train, con una validazione interna a blocchi di tempo e con purge ed
  embargo;
- il test non entra, l'OOS mai.

**Metodo:** lo proponi tu, spiegato in parole semplici. Deve comprendere:
1. togliere i doppioni: fra feature molto correlate se ne tiene una per gruppo;
2. misurare l'importanza fuori dal campione di addestramento, per esempio con la permutazione
   su gruppi di feature correlate;
3. controllare che l'importanza sia stabile fra i blocchi.

Il numero di feature finali segue una regola fissata prima di guardare i dati.

**Stampa:** per ogni selezione, le feature tenute, quelle tolte e il motivo.

**Da spiegare a Mattia.** Più feature e più scelte vogliono dire più occasioni di adattarsi
al caso. Il controllo è un test che la selezione non ha mai visto, più l'OOS.

## §5 · Domande aperte

La proposta risponde a ciascuna con pro e contro. Decide Mattia.

- **D1 · Pesi.** Oltre all'unicità, pesare ogni trigger anche per |rendimento|, così il
  modello bada di più ai trade che muovono il PnL?
- **D2 · Periodo dell'event study.** Farlo solo sul train? Oggi gira su tutto l'IS, compreso
  il test da cui si ricava la soglia.
- **D3 · Confronti fissi sul test.** Confrontare il modello con «trigger + miglior singolo
  filtro F*» e con «20% dei trigger preso a caso»?
- **D4 · Registro delle prove del ramo ML.** Dove e come registrare le selezioni di feature e
  i modelli provati, senza toccare lo studio di V4?
- **D5 · Motore del backtest.** VectorBT o il motore veloce di V4?
- **D6 · Spot.** Solo il modello LONG, visto che V4 non fa short sullo spot?

## §6 · Cosa contiene la proposta

Per ogni lavoro (A–D):
- file aggiunti, toccati e tolti;
- funzioni, con firma e cosa restituiscono;
- test;
- comandi per eseguire e cosa si vede.

In più, le risposte a D1–D6.

**Limiti:** al massimo 4 pagine, nessun codice.
