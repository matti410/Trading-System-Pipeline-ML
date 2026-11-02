# 04 · Scelta automatica del gruppo di trigger e di H

**Incarico del 01/11/2026**, preparato da Mattia e dal revisore per Claude Code.

**Flusso:**
1. proposta in testo in `docs/05_PROPOSTA_GRUPPO.md`, senza codice;
2. revisione;
3. via di Mattia;
4. codice su un ramo, poi PR.

**Indice:** §1 Perché · §2 Le regole (decise) · §3 L'algoritmo · §4 Cosa si vede · §5 Occorrenze
effettive · §6 Test · §7 Cosa contiene la proposta

## §1 · Perché

Oggi Mattia sceglie i trigger e H a mano. La riga «risoluzione» mostra però che con pochi trigger
il test distingue solo differenze di circa ±20–27 bp, più del costo di un trade (circa 10 bp).
Unire trigger promettenti aumenta le occorrenze, ma va fatto con regole fisse. I controlli
riguardano cinque rischi:
1. annacquare la media del gruppo;
2. le sovrapposizioni;
3. H unico per lato;
4. meccanismi diversi nello stesso gruppo;
5. il migliore fra tanti sembra migliore di quello che è.

Questo incarico li automatizza.

## §2 · Le regole (decisioni di Mattia del 01/11/2026)

1. **Soglia d'ingresso:** un trigger entra solo con **media netta > 0 bp**, cioè dopo i costi del
   Lab (target.py).
2. **Griglia di H in barre**, non in ore: `GRIGLIA_H` è una lista scritta da Mattia nel notebook,
   per esempio `[4, 8, 12, 16, 24, 32]`. Accanto si stampa l'equivalente in ore, ma non lo guida.
   È una lista corta fissata prima di guardare: ogni H provato è una prova in più.
3. **Famiglie (controllo 4): nessun blocco**, perché è la strada più semplice. L'identità del
   trigger è già una feature, e il controllo che conta per il gruppo è la regola netta e stabile.
4. **«Oltre il rumore» non è richiesto.** Con un catalogo grande diventa sempre più severo. Al suo
   posto c'è la stabilità nelle due metà del train (§3).
5. **Tutto si calcola solo sul train**, come l'event study. Il test serve solo alla verifica del
   §4. L'OOS mai.
6. **Mattia può sempre scegliere a mano.** Il gruppo automatico è una proposta e Mattia può
   sovrascriverla nella cella 4.

## §3 · L'algoritmo, per un lato

Per ogni H della griglia:
1. **Bp netti di ogni occorrenza** di ogni trigger del lato nel train: stessa funzione del target
   (ingresso Open i+1, uscita Open i+H+1, costi del Lab, quarantena). Si tengono solo le uscite
   che cadono dentro il train.
2. **Trigger ammessi:** media netta > 0 su tutte le loro occorrenze **e** in ciascuna delle due
   metà del train. Le metà si dividono per tempo, a metà delle barre del train.
3. **Gruppo un trigger alla volta.** Si ordinano gli ammessi per media netta decrescente; a
   parità, vince quello con più occorrenze. Si parte con il gruppo vuoto. Un trigger si aggiunge
   se le sue **occorrenze nuove**, cioè le barre non già coperte dal gruppo, hanno media netta > 0
   sia in tutto il train sia in ciascuna metà. Una barra in cui scattano più trigger è un solo
   trade, perché con lo stesso H il netto è lo stesso.
4. **Totale(H)** = somma dei bp netti delle barre del gruppo, ciascuna contata una volta.

**H scelto:** quello con il Totale più alto; a parità, l'H più corto. Il gruppo è quello di
quell'H.

Domanda aperta, da trattare nella proposta: serve un minimo di occorrenze nuove per metà (per
esempio 10), perché la regola delle due metà non si decida su 1–2 trade? Pro e contro. Decide
Mattia.

## §4 · Cosa si vede nel notebook

- **Passo 4, la proposta:**
  - una tabella per H: trigger nel gruppo, occorrenze, effettive (§5), media netta, Totale, con
    l'H scelto evidenziato;
  - la composizione del gruppo: per ogni trigger, le occorrenze nuove, la media netta e le due
    metà;
  - il grafico dell'event study con le curve dei soli trigger del gruppo e una linea verticale a
    H.
- **Cella 4:** `TRIGGER_<LATO>` e `H_<LATO>` si compilano con la proposta. Mattia può
  sovrascriverli; il notebook stampa «proposta automatica» oppure «scelta a mano».
- **Passo 5, verifica onesta prima del modello:** media netta del gruppo sul train e sul test, con
  il ± per ciascuna. Se il test è molto sotto il train, il notebook lo dice.
- **Registro:** la scelta del gruppo è una voce («gruppo»): griglia, ammessi, gruppo, H, Totali,
  proposta o scelta a mano.
- Sullo spot solo il LONG, come oggi.

## §5 · Occorrenze effettive

**Nella riga «risoluzione» del passo 5**, per il train e per il test:
- occorrenze effettive = somma dei pesi di unicità **non** normalizzati, con la stessa finestra di
  `pesi.pesi_unicita`, calcolati separatamente sul train e sul test;
- il ± atteso si calcola con le effettive del test;
- le frasi «!!!» usano le effettive.

**Nella tabella per trigger:** la colonna «effettivi».

## §6 · Test minimi

- **Dati inventati con 3 trigger.** A: positivo e stabile. B: positivo solo nella prima metà. C:
  negativo. Il gruppo deve contenere solo A.
- **Due trigger sempre sulle stesse barre:** il secondo non aggiunge occorrenze nuove e non entra.
- **H:** l'H scelto è quello con il Totale più alto; a parità, il più corto.
- **Solo train:** cambiando i prezzi del test, gruppo e H non cambiano.
- **Scelta a mano:** sovrascrive la proposta e finisce nel registro come «scelta a mano».
- **Effettive:** con trigger isolati, effettive = grezze; due trigger sempre sulla stessa barra →
  la metà.
- **Nessun numero esistente cambia** quando la scelta è fatta a mano con gli stessi trigger e H di
  prima.

## §7 · Cosa contiene la proposta

- File, funzioni con firma e cosa restituiscono.
- Tempi di calcolo stimati su BTC perp M5 e M15.
- La risposta alla domanda aperta del §3.
- Comandi da lanciare e cosa si vede.

**Limiti:** al massimo 2 pagine, nessun codice.
