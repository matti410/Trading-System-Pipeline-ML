# 02 · Proposta: conversione a crypto, pulizia e feature di contesto

**30/10/2026**, risposta a `docs/01`. Solo testo, niente codice. V4 letto al ramo `mappa-semplice`,
commit `8d5a8f3`.

**Indice:** §A Pulizia e collegamento · §B Pipeline base · §C Catalogo · §D Selezione · §E Risposte D1–D6

---

## §A · Pulizia e collegamento a V4

**Da togliere** (verificato: nessun modulo ML che resta li usa):
- `EURUSD_M15.csv` e `costi_symbol.py` (IC Markets, in pips);
- `engine/`, tutti i 9 file: sono copie V3 e coprirebbero l'`engine` di V4, che ha lo stesso nome;
- `entry_long.py`, `entry_short.py`, `filter_conditions.py` e `helpers.py` (lo usano solo i due `entry_*`);
- nel notebook: token Colab e `git clone`, cella MT5, lettura del CSV EURUSD con `to_utc_index`, celle dei
  costi IC Markets, `freq="15min"` fisso, ogni riferimento ai pips.

**Da prendere da V4**: le stesse funzioni della schermata 1 del Lab (`vetrina.prepara`).

| cosa | funzione di V4 |
|---|---|
| cartella dei dati | `vetrina.cartella_dati_avvio`: Drive per desktop se c'è, altrimenti `dati/` |
| aggiornamento | `aggiorna_dati.aggiorna_per_lab`, chiamata da `vetrina.apri_dati(..., aggiorna=True)` |
| lettura e funding | `vetrina.apri_dati(symbol, timeframe, cartella, confine=<data>)`: CSV in UTC, taglio al funding, `carica_funding` |
| indicatori, IS/OOS | `indicatori.aggiungi_indicatori(df)`; `vetrina.dividi_dati(d, df)` → `splitting.split_al_confine` |
| symbol | `simboli.info_symbol`, `pip_symbol`; `exchange.parti_symbol`, `solo_long` |
| costi | `vetrina.costi_storico(symbol, df, funding, df_prezzo=df_is)` → `exchange.costi_exchange` |
| costo di ogni trade | `exchange.aggiungi_spread_passo` (un passo), `exchange.aggiungi_funding` (perpetual), `metriche.bp_per_trade` (taker per lato, tutto in bp) |
| trigger e filtri | `registra_trigger_long/short`, `registra_filtri`, `registra_filtri_vwap` dei file di V4; `registry.get_entry/get_filter`; elenco da `vetrina._trigger_candidati(dir)` e `_tutti_i_filtri()` |
| event study | `event_study.run_event_study(df, entry_names, horizon, min_trades, pip)` |
| confine | `studio.leggi_confine(cartella)`, **in sola lettura** |
| quarantena | sì, il Lab la applica agli ingressi: `quarantena_trade.maschera_segnali_puliti(df)` scarta il trigger se è in quarantena la sua barra o quella d'ingresso |

**Lo studio non si tocca.** Il Lab entra nello studio con `studio.entra`, che può **scrivere** (apre lo studio
se manca, scrive `impronta.json` per un dataset nuovo). Qui non lo usiamo: si legge `confine.json` e lo si
passa ad `apri_dati` come data. Se lo studio manca, ci si ferma («aprilo prima nel Lab»). L'impronta si
controlla in sola lettura (`studio.impronta(df_is)` contro il file, se c'è).

**Funzioni private.** `_trigger_candidati` e `_tutti_i_filtri` sono private: le usiamo con un test che si
ferma se spariscono. **Proposta per V4:** renderle pubbliche.

**Collegamento.** Sul PC il notebook trova da solo `../Trading-System-Creator_V4` e la mette in `sys.path`:
niente da installare, niente copie. Su Colab clona V4 (repo pubblico, senza token) nel ramo del campo
`V4_RAMO`, di default `main`. All'avvio stampa cartella, ramo, commit, data del commit e «modifiche non
committate sì/no». Se V4 non è su `main` o ha file modificati non committati, stampa un avviso ben
visibile (`!!! ATTENZIONE`, §F). Si ferma se l'`engine` importato non viene solo da V4.

**File.** Nuovi: `collegamento_v4.py`, `dataset.py`, `requirements.txt` (versioni bloccate di V4: TA-Lib
0.8.1, numba 0.68.0, backtesting 0.6.6; più lightgbm, scikit-learn, pytest), `tests/`. Toccati: `target.py`,
`modello.py` (`rendimenti_trigger` dà il netto in bp), `split.py`, `pesi.py`, `oos.py` (solo docstring e
`horizon` obbligatorio), notebook celle 0–2. Le altre celle del vecchio notebook usavano i file tolti:
si tolgono, e il lavoro B rifà la sequenza. `diagnostica.py` resta com'è.

**Funzioni.**
- `collegamento_v4.collega(percorso=None, ramo="main") -> dict` con cartella, ramo, commit, data e
  modifiche locali.
- `dataset.apri(symbol, timeframe, cartella, aggiorna=False) -> Dataset` con `df` (indicatori, E*, F*),
  `df_is`, `df_oos`, `confine`, `funding`, `costi`, `pip`, `solo_long`.
- `target.costruisci_target(df, colonne_trigger, direction, horizon, *, costi, quarantena=True) ->
  DataFrame` con `TARGET` (0/1), `lordo_bp`, `netto_bp`, `ingresso` e `uscita` per ogni trigger valido
  (corretto dopo §F, D5). Fa la tabella dei trade come il motore veloce di V4 con `n_barre = H`: ingresso
  all'Open di i+1, uscita all'**Open di i+H+1**. Poi la passa alla stessa catena di costi del Lab
  (`costo_extra_pips` → `aggiungi_funding` → `aggiungi_spread_passo` → `netti_in_unita(..., "bp")`).
  `TARGET = 1` se il netto è sopra zero; il pareggio vale 0. I trigger in quarantena si scartano. Il
  funding arriva con `costi["funding"]` e si conta come nel Lab (`exchange.funding_per_trade`).

**Test** (`python -m pytest -q`):
1. target su 12 barre inventate con commissione, passo e funding noti: etichette e bp uguali a quelli fatti
   a mano, compreso un pareggio esatto (→ 0);
2. costi identici al Lab: `dataset.apri(...).costi` = `vetrina.costi_storico` sullo stesso symbol (si salta
   con un messaggio se mancano i dati);
3. identità esatta con il motore veloce di V4 (§F, D5): per ogni trade del motore, stesso prezzo
   d'ingresso, stesso prezzo d'uscita e stessi bp netti del target, uguali al bit. Su barre inventate e
   sui dati veri, con funding e quarantena;
4. un trigger in quarantena non entra nel target;
5. il collegamento trova V4 e legge il commit; le due funzioni private esistono;
6. nessun file di `studio/` cambia (ora di modifica prima e dopo `dataset.apri`).

**Comandi e cosa si vede.** Una volta, nella cartella del repo: `python -m venv .venv` e
`.venv\Scripts\python -m pip install -r requirements.txt`. Poi `.venv\Scripts\python -m pytest -q` →
`N passed` (qualche `skipped` se mancano i dati). Nel notebook, celle 0–2: la riga «V4: ramo …, commit …» e
la tabella del dataset (symbol, barre, confine, costo in bp, funding sì/no).

---

## §B · Pipeline base crypto (si rifà `TS-Pipeline-ML.ipynb`)

**Sequenza del notebook.**
1. **Dataset**: campi `exchange`, `mercato`, `moneta`, `timeframe`; interruttore `aggiorna_dati` spento di
   default, attivo solo sul PC, come nel Lab.
2. **Confine IS/OOS** da V4, con barre IS e OOS stampate.
3. **Event study** sulle sole barre del train (D2), con tutti i trigger candidati e il `min_trades` del Lab.
   Si stampa la sintesi ordinata, senza scelte automatiche.
4. **Scelta di Mattia**: si scrivono `TRIGGER_LONG`, `H_LONG`, `TRIGGER_SHORT`, `H_SHORT`. H in barre come
   l'event study, con le ore stampate accanto. Il notebook controlla che i nomi esistano. Sullo spot lo
   SHORT si salta (D6). Le celle `l1/l2/s1/s2` si tolgono.
5. **Target** per lato, con il suo H: righe, % di classe 1, bp netti medi.
6. **Train/test** con `split_train_test(target, df_is, horizon=H_lato)`: purge ed embargo sull'H del lato.
7. **Modello base**: sole F* di V4 più identità del trigger (`costruisci_X`, come oggi). AUC e fasce in bp
   netti.
8. **Soglia** per fasce dal test (`oos.valuta_oos`, invariato).
9. **OOS e backtest**: il motore dipende da D5.

**File.** Toccati: il notebook, `backtest_oos.py` (secondo D5), `modello.valuta` e `oos.valuta_oos` (bp
netti al posto dei lordi), `split.costruisci_X`. Oggi `costruisci_X` riconosce i filtri dal nome `F<numero>_`,
ma 3 filtri del Lab si chiamano `VWAP_*` (visto nel lavoro A): userà l'elenco `dataset.filtri`. Nuovo:
`pipeline.py`.

**Funzioni.**
- `pipeline.event_study_train(ds, direzione, H, quota=0.8) -> EventStudy`: l'event study di V4 sulle barre
  prima del confine train/test.
- `pipeline.controlla_scelta(trigger, H, direzione) -> None`: si ferma se un nome non esiste, se H < 1 o se
  si chiede lo short sullo spot.

**Test:** lo split con H diversi per lato toglie le righe giuste (dati inventati); l'event study del train
non vede barre del test; sullo spot `controlla_scelta` rifiuta lo short.

**Cosa si vede:** il notebook si ferma alla cella 4 finché non scrivi trigger e H. Alla fine c'è la tabella
OOS «fascia scelta» contro «tutti», in bp netti.

---

## §C · Catalogo delle feature di contesto

**Regole.** Finestre fisse, decise ora: **W = 4 h, 24 h, 168 h** e una lunga **L = 720 h** per percentili e
normalizzazioni. Le ore diventano barre con `indicatori.barre_per_durata` di V4. Se una finestra dà meno di 6
barre (es. 4 h su H4), la feature non si calcola e si stampa. r = rendimento logaritmico di una barra;
σ_W = deviazione standard di r su W. Solo barre chiuse fino al trigger compreso.

| # | feature | formula |
|---|---|---|
| 1 | `ret_lag_k`, k=1…6 | r(t−k+1) / σ_24h (sono ritardi in barre, non finestre) |
| 1 | `ret_cum_W` | ln(C_t / C_{t−W}) / (σ_W · √n_W) |
| 2 | `vol_W` · `vol_breve_lunga` · `vol_pct` | σ_W / σ_L · σ_4h / σ_168h · percentile di σ_24h su 720 h |
| 2 | `park_24h`, `gk_24h` | Parkinson e Garman-Klass su 24 h, divisi per σ_24h |
| 3 | `er_W` · `vr_24h` | efficiency ratio (`indicatori.efficienza_kaufman`) · variance ratio (`indicatori.variance_ratio`) |
| 3 | `ac1_168h` | autocorrelazione di r a ritardo 1, su 168 h |
| 3 | `pend_W`, `r2_W` | pendenza della regressione di ln C sul tempo, in σ_W per barra, e il suo R² |
| 4 | `pos_range_W` | (C − min Low) / (max High − min Low) su W |
| 4 | `dist_max_W`, `dist_min_W` | ln(max High / C) / σ_24h; ln(C / min Low) / σ_24h |
| 4 | `da_max_W`, `da_min_W` | barre dall'ultimo massimo (minimo) / n_W |
| 5 | `dist_vwap_W`, `dist_ema_W` | ln(C / VWAP rolling su W) / σ_24h (`engine/vwap_ops` di V4); ln(C / EMA di n_W) / σ_24h |
| 6 | `asim_168h`, `curt_168h` | asimmetria e curtosi di r su 168 h |
| 7 | `vol_ora` | Volume / mediana del volume alla stessa ora UTC degli ultimi 20 giorni |
| 7 | `vol_z_168h` · `vol_mov_24h` | z-score di ln Volume · (Σ Volume / Σ\|r\|) su 24 h diviso la sua mediana su 720 h |
| 8 | `corpo_k`, `ombra_su_k`, `ombra_giu_k`, k=0,1,2 | (C−O)/(H−L), (H−max(O,C))/(H−L), (min(O,C)−L)/(H−L); range nullo → 0 |
| 9 | `ora_sin/cos`, `giorno_sin/cos` | sen e cos di 2π·ora/24 e 2π·giorno/7, in UTC |
| 10 | `btc_ret_cum_W`, `btc_vol_24h`, `corr_btc_168h` | come 1–2, su BTC dello **stesso exchange e mercato**; correlazione rolling di r. Solo altcoin |
| 11 | `trig_24h`, `da_trig_prec` | trigger del lato nelle ultime 24 h (escluso il corrente); barre dal precedente / n_24h, con tetto a 1 |

In tutto sono circa 80 colonne, più le F* e l'identità del trigger (una colonna 0/1 per trigger). Il taker
buy ratio entra quando V4 lo avrà (docs/77 di V4, lavoro 4). Riscaldamento: 720 h (su M15 sono 2.880 barre)
più quello degli indicatori di V4.

**File.** Nuovi: `feature/catalogo.py` (`CATALOGO`: nome → famiglia, finestra_ore, funzione),
`feature/calcolo.py`, `tests/test_feature.py`.

**Funzioni.**
- `calcola_feature(df, timeframe, trigger_lato, df_btc=None) -> DataFrame`: una colonna per feature, stesso
  indice di `df`.
- `elenco_catalogo() -> DataFrame`: nome, famiglia, finestra in ore e in barre, calcolata sì/no.

**Test.**
1. **Lookahead:** `collaudo_catalogo.verifica_lookahead(df_grezzo, prepara=indicatori + feature)` dà OK su
   ogni colonna: la funzione di V4 controlla già tutte le colonne aggiunte da `prepara`.
2. **Degenerazione:** `verifica_degenerazione` di V4 guarda solo le condizioni 0/1 del registro. Le F*
   passano da lì; per le colonne continue serve lo stesso controllo, con tre casi: mai calcolata, quasi
   costante (un valore in ≥ 95% delle barre), troppo rara (< 30 valori non nulli). **Proposta per V4:**
   estendere la funzione alle colonne continue; intanto il controllo sta in `feature/calcolo.py`.
3. **Adimensionali:** con prezzi ×10 e volume ×10 ogni feature resta uguale (tolleranza 1e-9).
4. **Ore → barre:** 24 h danno 288, 96, 24 e 6 barre su M5, M15, H1 e H4.

**Cosa si vede:** la tabella `elenco_catalogo()` e l'esito dei collaudi, una riga per feature (OK /
LOOKAHEAD / DEGENERE).

---

## §D · Feature selection

**Il metodo, in parole semplici.** Tutto succede dentro il train; test e OOS non si toccano.
1. **Blocchi di tempo.** Il train si divide in 5 blocchi consecutivi. Si fanno 4 giri: si allena sui blocchi
   1…k e si valuta sul k+1. Fra un pezzo e l'altro si toglie H del lato (purge ed embargo, come `split.py`).
2. **Sonde** (§F). Si aggiungono 10 colonne di puro rumore (seme fisso), che entrano nei gruppi e nella
   permutazione come tutte le altre. Dicono quanta «importanza» si ottiene per caso.
3. **Doppioni.** Sul solo train, correlazione di Spearman e gruppi con raggruppamento gerarchico (|ρ| ≥ 0,7).
   Di ogni gruppo si tiene la feature più correlata con le altre del gruppo: il target non entra in questa
   scelta. L'identità del trigger resta sempre.
4. **Importanza fuori campione.** In ogni giro si allena LightGBM (`PARAMETRI_BASE`, pesi di unicità). Poi,
   sul blocco di validazione, si mescolano **insieme** le colonne di un gruppo: l'importanza è quanto
   peggiora la log-loss (media di 5 mescolamenti, semi fissi).
5. **Stabilità e sonde.** Un gruppo si tiene solo se l'importanza è sopra zero in almeno 3 giri su 4 **e**
   la sua importanza mediana supera quella della **sonda migliore** (§F).
6. **Numero finale**, regola fissata ora, dopo il punto 5: al massimo K = min(15, trigger di train / 100) gruppi, i più
   importanti per mediana fra quelli stabili. Poi si riallena su tutto il train con le feature scelte.

**Stampa.** Una tabella per dataset e lato: feature, gruppo, importanza per giro e mediana, esito. L'esito è
«tenuta» oppure «tolta» con il motivo (doppione di X / instabile / sotto la sonda migliore / oltre K).
In tabella c'è anche la riga delle sonde (la migliore e la mediana delle 10). Il confronto sul test fra
modello base e modello con le feature scelte si registra in `registro_ml` come prova (§F, D4).

**Da spiegare a Mattia.** Ogni feature in più e ogni regola provata sono un'occasione in più di trovare per
caso qualcosa che funziona solo nel passato. La selezione riduce il rischio ma non lo toglie: sceglie
comunque fra circa 80 candidate. Il controllo onesto è doppio: il test, che la selezione non ha mai visto, e
l'OOS, che si guarda una volta. Per questo le soglie (0,7, 3 su 4, sonde, K) si fissano adesso e non si ritoccano
dopo aver visto il test.

**File.** Nuovi: `selezione.py`, `tests/test_selezione.py`.

**Funzioni.**
- `blocchi_interni(idx_train, df, horizon, n_blocchi=5) -> list[(idx_fit, idx_val)]`
- `gruppi_correlati(X_train, soglia=0.7) -> DataFrame`: gruppo di ogni feature e se è la rappresentante.
- `importanza_gruppi(X, y, pesi, blocchi, gruppi, ripetizioni=5, seme=0) -> DataFrame`: una riga per gruppo,
  una colonna per giro.
- `aggiungi_sonde(X, n=10, seme=0) -> DataFrame`: X con 10 colonne di rumore `SONDA_1…SONDA_10`.
- `seleziona(importanze, n_train) -> (tenute, tabella_esiti)`: con la regola delle sonde, poi K.

**Test:** i blocchi non si sovrappongono e rispettano purge ed embargo; con 1 feature vera e 20 di rumore,
la vera si tiene e **nessuna** di rumore passa (§F); due copie della stessa feature
finiscono nello stesso gruppo; se entra una data del test, la funzione si ferma.

**Cosa si vede:** la tabella della selezione per lato, poi AUC e fasce sul test del modello con le feature
scelte, accanto al modello base.

---

## §E · Risposte alle domande aperte (decide Mattia)

**D1 · Pesi anche per |rendimento|.**
- Pro: il modello bada di più ai trade che spostano il PnL. Oggi un +3 bp e un +300 bp contano uguale.
- Contro: sulle crypto le code sono grosse. Pochi trade estremi (crolli, notizie) dominerebbero il train, e
  non si ripetono.
- **Proposta:** interruttore spento di default. Acceso: peso = unicità × min(|netto_bp|, 95° percentile del
  train), con il tetto preso dal solo train. È una variante in più e si registra (D4).

**D2 · Event study solo sul train.**
- Pro: il test resta pulito. Oggi H e trigger si scelgono guardando anche il test, e dallo stesso test si
  ricava la soglia, che così risulta più ottimista del vero.
- Contro: l'event study ha il 20% di dati in meno, e i numeri non coincidono con il Lab, che usa tutto l'IS.
  Se i trigger si scelgono guardando il Lab, quella scelta ha già visto il test: si può solo dichiarare.
- **Proposta:** solo sul train.

**D3 · Confronti fissi sul test.**
- Pro: dicono se il modello batte una regola banale; senza, un buon risultato può venire solo dal periodo.
- Contro: più tabelle. Il «miglior filtro» è scelto fra tanti, quindi è un avversario un po' gonfiato.
- **Proposta:** sì, tutti e due, solo stampati e mai usati per scegliere. Il miglior F* si sceglie sul
  **train** (bp netti medi più alti, almeno 100 trade) e si misura sul test. Il «20% a caso» si fa con 1.000
  estrazioni a seme fisso, e si stampa in che percentile cade il modello.

**D4 · Registro delle prove.**
- **Proposta:** cartella `DATI_DIR/registro_ml/`, accanto a `studio/` ma separata. Un file `.jsonl` per
  dataset e per macchina, solo in aggiunta, come i giornali di V4, scritto con `studio.scrivi_sicuro`.
- Una riga per giro: data, commit di V4 e di questo repo, dataset, lato, H, trigger, feature tenute e tolte,
  parametri, semi, metriche su blocchi e test, «OOS guardato sì/no». Il notebook stampa quante prove ci sono
  già e quante volte si è guardato l'OOS. Anche il confronto sul test «modello base contro modello con le
  feature scelte» è una prova e si registra (§F).
- Pro: si vede da PC e da Colab, e lo studio di V4 non si tocca. Contro: non sta su git; serve una copia ogni
  tanto.

**D5 · Motore del backtest.**
- Motore veloce di V4 (`MotoreVeloce`, uscita a tempo `n_barre`). Pro: è lo stesso motore del Lab, con gli
  stessi costi (passo, taker, funding); i numeri si confrontano con il Lab; vectorbt non serve più. Contro:
  esce all'**Open** della barra dopo, non al Close di i+H (sulle crypto 24/7 di solito zero o un passo di
  differenza, ma non identico al target); investe tutto il capitale a ogni trade, invece del lotto fisso.
- VectorBT. Pro: segue esattamente la regola del target e gestisce già LONG + SHORT combinati. Contro: è una
  seconda fonte di costi da allineare a mano, e una libreria pesante da tenere d'accordo con numba 0.68 di V4.
- ~~Proposta: uscita entro un passo~~. **Decisione (§F):** motore veloce di V4, e il target si allinea al
  motore: uscita all'Open di i+H+1. Il test è di identità esatta su prezzi d'uscita e bp netti.

**D6 · Spot solo LONG.**
- Pro: coerente con V4 (`exchange.solo_long`); un modello SHORT sullo spot non avrebbe trade possibili.
- Contro: sullo spot non c'è la tabella «combinato».
- **Proposta:** sì. Sullo spot la parte SHORT si salta, con un messaggio.

**Valori di collaudo** (H = 25, fascia 5, `PARAMETRI_BASE`): restano tali. I nuovi numeri fissati qui (W, L,
0,7, 3 su 4, K, 5 blocchi) sono regole decise prima dei dati: si cambiano solo con una nuova proposta.

---

## §F · Decisioni di Mattia (30/10/2026)

1. **D1 sì**: pesi anche per |rendimento|, interruttore spento di default, tetto al 95° percentile del
   train.
2. **D2 sì**: event study solo sul train.
3. **D3 sì**: i due confronti fissi sul test, solo stampati.
4. **D4 sì**: registro in `DATI_DIR/registro_ml/`. In più, il confronto sul test «modello base contro
   modello con le feature scelte» si registra come prova.
5. **D5: motore veloce di V4**, e il **target si allinea al motore**: uscita all'**Open della barra
   i+H+1**, non al Close di i+H, con la stessa durata (H barre fra ingresso e uscita). Così target =
   backtest = Lab, senza tolleranze. Il test «entro un passo» diventa un test di identità esatta sui prezzi
   d'uscita e sui bp netti.
   - **Verifica fatta** su `engine/motore_veloce.py` (`_ciclo`). Con trigger sulla barra s, il motore entra
     all'Open di s+1. Con `n_barre = H` chiede la chiusura alla barra s+H (`i − t_eb ≥ n_barre − 1`) e la
     riempie all'Open di s+H+1, quindi `ExitBar − EntryBar = H`. **Coincide** con la regola decisa.
   - Due dettagli del motore, ripresi nel target: il trigger sulla barra 0 non entra (il ciclo parte dalla
     barra 1) e un trade senza la barra s+H+1 resta aperto ed è escluso.
6. **D6 sì**: sullo spot solo LONG.
7. **§D, le sonde**: 10 colonne di rumore (seme fisso) entrano nei gruppi e nella permutazione come le
   altre. Un gruppo si tiene solo se è stabile (3 giri su 4) **e** la sua importanza mediana supera la
   sonda migliore. Poi vale K. Nella tabella si stampa la riga delle sonde. Test: con 20 feature di rumore
   e 1 vera, nessuna di rumore passa.
8. **§A, collegamento**: avviso ben visibile se V4 non è sul ramo `main` o ha modifiche non committate.

Il codice del lavoro A è stato collaudato su V4 al ramo `main`, commit `ab62755`.

---

## §G · Chiusura del lavoro B (30/10/2026)

**Fatto come §B e §F.** Note su quello che il lavoro ha fissato:
- **Event study sul train**: le barre di `df_is` prima di `split.barra_confine`, la stessa regola dello
  split. Orizzonte: il più lungo della griglia del Lab (8 ore); la curva mostra quelli intermedi.
- **X**: identità del trigger + l'elenco `dataset.filtri` (38 filtri, di cui 3 `VWAP_*`).
- **Registro** (`registro.py`): una riga JSON aggiunta in fondo a
  `DATI_DIR/registro_ml/<DATASET>/giornale_<macchina>.jsonl`, aperta in aggiunta come i giornali dello
  studio di V4 (`studio.aggiungi_voce`), non con `scrivi_sicuro`, che riscrive il file intero. Si registrano
  ogni addestramento e ogni sguardo all'OOS (tabella e backtest), con `collaudo` sì/no.
- **Backtest** (`backtest_oos.py`): `MotoreVeloce` di V4 con `n_barre` = H del lato, `exclusive_orders`, un
  trade alla volta per lato, costi con la catena del Lab. Il motore gira sullo storico intero con i
  segnali accesi solo sulle barre del periodo. Unità: bp netti per trade, a capitale fisso.

**Come si calcola il COMBINATO (LONG + SHORT).**
- I due lati girano separati, ciascuno con il suo capitale: possono essere aperti insieme (esposizione
  doppia, nessuna compensazione fra i due).
- Il COMBINATO è l'unione dei trade dei due lati, in ordine di ora d'uscita.
- Trade, vincenti, trade medio e profit factor si calcolano sui trade uniti; il totale è la somma dei bp.
- La curva è la somma cumulata dei bp di tutti i trade; il max drawdown si misura su quella curva.
- Esposizione: quota delle barre del periodo con almeno un lato aperto. Saltati: somma dei due lati.
