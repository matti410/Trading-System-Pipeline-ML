# Istruzioni per Claude Code — Trading-System-Pipeline-ML

Mattia NON è un informatico. Rispondi in italiano, con parole semplici. A ogni passo di':
cosa hai fatto, come si esegue ogni comando, cosa si deve vedere. Ragiona come un trader
algoritmico esperto.

## Cos'è
Meta-labeling sopra i trigger del Lab. Il Lab è il repo fratello
`../Trading-System-Creator_V4` (qui lo chiamiamo «V4»). Quando scatta un trigger, il modello
stima la probabilità che il trade chiuda in guadagno netto. Sono due modelli, uno LONG e uno
SHORT. Solo crypto da exchange, come V4.

## Cosa leggere
- Sempre e solo: questo file e `docs/00_STATO.md`.
- Il resto solo se il lavoro lo chiede (l'incarico in corso è indicato in docs/00).
- Di V4 leggi solo i moduli che usi. Non leggere i suoi docs né il suo CLAUDE.md.

## Regole fisse
- **I risultati sui dati collaudano il tool, non sono verdetti.** Niente commenti del tipo
  «tiene/non tiene». Niente proposte di cambiare asset o segnali.
- **Niente codice senza il via di Mattia.**
  - Lavori grandi: proposta in docs, senza codice → revisione → via di Mattia → codice su un
    ramo → PR.
  - Lavori piccoli: descrizione breve in chat → via di Mattia → codice.
  - Niente aggiunte non richieste: si propongono.
- **V4 è in sola lettura.** Mai modificare file di V4 né la sua cartella `studio/`. Se in V4
  manca qualcosa, scrivilo come proposta per V4.
- **Una sola fonte, niente copie.** Vengono da V4: dati, symbol, costi, catalogo dei trigger
  e dei filtri, event study, confine dell'in-sample.
- **Target = regola d'uscita del backtest.**
  - Ingresso all'Open della barra dopo il trigger.
  - Uscita al Close a H barre dal trigger. Niente SL/TP.
  - TARGET = 1 se il guadagno netto supera i costi del Lab (stessa funzione di V4).
  - H è uno per lato: `H_LONG` e `H_SHORT`.
- **Nessun lookahead.** Ogni feature usa solo barre chiuse fino al trigger. Ogni feature
  passa `verifica_lookahead` e `verifica_degenerazione` di V4.
- **Separazione dei dati.**
  - L'OOS non sceglie mai niente.
  - La feature selection e i parametri si decidono solo nel train.
  - Il test serve solo a fissare la soglia di probabilità.
- **Feature adimensionali**: rapporti, z-score, percentili su una finestra passata. Mai il
  livello del prezzo. Le finestre si scrivono in ore e si convertono in barre in base al
  timeframe.
- **Commit e PR.**
  - Prima di ogni commit lancia tutti i test.
  - Prima di ogni push controlla che la PR sia ancora aperta.
  - Il merge lo fa Mattia.
  - Le date si prendono dal sistema (`date`), non si inventano.
- **A fine lavoro** scrivi un resoconto breve: cosa cambia, test superati, comandi da
  lanciare, cosa si vede.
