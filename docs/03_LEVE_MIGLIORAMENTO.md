# 03 · Leve di miglioramento del ramo ML

**31/10/2026.** Mattia e il revisore, dopo il primo giro reale (BTC perp M5). È un elenco di leve,
non un incarico: ogni leva diventa lavoro solo con il via di Mattia.
**Indice:** §1 Diagnosi · §2 Leve in ordine · §3 Cosa non conviene ora · §4 Come si prova ·
§5 Ingressi limit (maker), messa da parte · §6 Flusso degli ordini (leve 4a e 4b)

## §1 · Diagnosi del primo giro

**1. Il modello filtra, non crea guadagno.**
- Può alzare la media dei trigger scelti solo di quanto riesce a separare i trade buoni da quelli
  cattivi.
- Giro M5 LONG: tutti i trigger −11 bp netti a trade nell'OOS, fascia 5 −7,5 bp. Il modello ha
  recuperato circa 3,5 bp, ma il costo di un trade è circa 10 bp.
- Per passare sopra lo zero, il modello deve partire da trigger con una media vicina allo zero o
  sopra.

**2. Il target binario premia la volatilità.**
- TARGET = 1 vuol dire «netto > 0», cioè «il movimento supera il costo». Con il mercato più mosso
  questo succede più spesso, in tutte e due le direzioni.
- Il modello impara «volatilità alta → più vincenti». L'AUC sale, ma i bp quasi non cambiano, perché
  crescono anche le perdite.
- Si vede dalle feature tenute: `vol_168h`, `vol_4h`, `ret_cum_4h`.

## §2 · Leve, in ordine di impatto atteso

| # | leva | perché | stato |
|---|---|---|---|
| 1 | **Pesi per rendimento** (D1), un solo interruttore `PESO_RENDIMENTO` per selezione e modelli | ogni trade pesa per quanto guadagna o perde: il modello impara il guadagno atteso, non la frequenza delle vincite (§1.2) | in lavoro, ramo `peso-rendimento` |
| 2 | **Trigger con picco netto vicino o sopra il costo** | la §1.1: il filtro aggiunge pochi bp | fatto (PR #7): linea del costo e `picco_netto_bp` al passo 3; tabella per trigger al passo 5 |
| 3 | **Un trigger frequente non deve annacquare gli altri** | più trigger in OR: quello molto frequente domina il dataset | fatto (PR #7): la tabella per trigger del passo 5 lo mostra, decide Mattia |
| 4a | **Flusso degli scambi** (taker buy volume), famiglia 12 | chi spinge il prezzo: un'informazione che il prezzo non contiene, mentre oggi tutte le feature vengono dal prezzo | da fare: download in V4 (docs/77, lavoro 4), poi famiglia 12 (§6) |
| 4b | **Sbilanciamento del book** (bookDepth di Binance) | liquidità in attesa sopra e sotto il prezzo | prima un download di prova (§6.3), poi si decide |
| 5 | **Un modello su più monete insieme** (BTC, ETH, SOL, XRP) | più esempi per un segnale debole; le feature sono già adimensionali | più avanti, da progettare |
| 6 | **Costi: ingressi limit (maker)** | circa −3 bp a trade | messa da parte (§5) |

## §3 · Cosa non conviene ora

- **Cambiare modello** (XGBoost, CatBoost, reti neurali).
  - Su dati tabellari con un segnale debole LightGBM è fra i migliori: gli altri danno differenze
    dell'ordine di 0,00x di AUC.
  - Una rete neurale qui fa peggio.
  - Unica aggiunta utile, più avanti: una regressione logistica come controllo. Se LightGBM non la
    batte, sta imparando rumore.
- **Ottimizzare i parametri di LightGBM.** Ogni combinazione provata è una prova in più, per un
  guadagno piccolo.
- **Filtri aggiuntivi messi a mano.** Il modello è già il filtro e vede tutte le F*. Un filtro scelto
  a mano rifà a occhio lo stesso lavoro, e conta come un'altra prova.

## §4 · Come si prova una leva

- **`GUARDA_OOS = False`** finché si sceglie: si confronta sul test, al passo 7b (bp della fascia 5
  meno bp di tutti i trigger).
- **L'OOS si apre una volta**, a scelta fatta.
- **`COLLAUDO = True`** solo per le prove del tool. Le prove vere restano nel registro e si contano.
- **Una leva alla volta**: per esempio `PESO_RENDIMENTO` False, poi True, con gli stessi trigger e
  gli stessi H.

## §5 · Ingressi limit (maker), messa da parte

Questa è la proposta del revisore. Non c'è nessuna decisione.

### Perché esiste questa leva

Costi da `engine/exchange.COMMISSIONI` di V4, Binance perp: maker 0,02%, taker 0,05% per lato.
- **Oggi:** taker + taker = circa 10 bp a trade.
- **Ingresso limit e uscita a mercato:** maker + taker = circa 7 bp.
- Lo spread di un passo pesa una frazione di bp: il guadagno viene tutto dalla commissione.

### Come si applica a «ingresso alla candela dopo»

1. Il trigger chiude la barra s. Invece di comprare a mercato all'Open di s+1, alla chiusura di s si
   mette un ordine **limit post-only** (che resta sempre maker) al prezzo P = Close(s) per il long e
   allo stesso prezzo per lo short. Variante: P più favorevole di k passi.
2. L'ordine vale **N barre**, con N fissato prima, per esempio 1.
3. **Regola di riempimento prudente nel backtest:** il long è eseguito nella prima barra j di
   s+1…s+N con Low(j) **strettamente** < P; lo short con High(j) > P. «Strettamente» tiene conto
   della coda: toccare il prezzo non basta per essere eseguiti.
4. **Se non viene eseguito, niente trade.** Nessun inseguimento a mercato: costerebbe taker e un
   prezzo peggiore.
5. **Uscita a tempo come oggi**, a mercato (taker), all'Open di s+H+1, a contare dall'ingresso
   effettivo.

### Il costo nascosto: la selezione avversa

- L'ordine limit viene eseguito proprio quando il prezzo va contro. I trade persi sono spesso i
  migliori, quelli in cui il prezzo parte subito a favore.
- Il risparmio di 3 bp può essere mangiato da questo effetto. Va misurato, non dato per scontato.

### Come si misura, dal più economico

1. **Senza toccare nessun motore:** per ogni trigger, sul train, si misura:
   - la % di barre s+1 con Low < Close(s), cioè quanti ingressi verrebbero eseguiti;
   - i bp netti medi dei trade eseguiti contro quelli persi.

   Se i persi sono molto migliori degli eseguiti, la leva non vale.
2. **Se il punto 1 promette:** gli ordini limit nel motore veloce di V4. Oggi il motore non li
   gestisce, quindi è un lavoro su V4.

### Effetto sul ML (da decidere allora)

Il target cambia:
- prezzo d'ingresso = P;
- un trigger non eseguito vale «nessun trade»: o si toglie dal target, o si tiene con netto 0
  (classe 0).

Le due scelte danno modelli diversi.

## §6 · Flusso degli ordini (leve 4a e 4b)

Decisione di Mattia del 31/10/2026: si tengono 4a (flusso degli scambi) e 4b (sbilanciamento del
book). Altre fonti di dati dei derivati sono fuori.

### §6.1 · Dove entra nel flusso di lavoro

| passo | strumento | domanda |
|---|---|---|
| 1 | trigger del Lab | **quando**: rotture, eccessi, momentum (eventi sul prezzo) |
| 2 | event study | il movimento medio supera il costo? Su quale H? |
| 3 | ML | **quale** trigger prendere, guardando il contesto al momento in cui scatta |

Il flusso entra al **passo 3**. A parità di evento sul prezzo, dice se il movimento è
**sostenuto** (qualcuno spinge) o **vuoto** (nessuno spinge, oppure qualcuno assorbe dall'altra
parte). Il grafico OHLCV non lo mostra: una candela verde con tanto volume può venire da
compratori aggressivi o da venditori che si ritirano.

**Il vero edge, per le tre famiglie di docs/77 di V4:**
- **Rotture:** separare quelle vere (aggressione forte e prezzo che avanza) da quelle false (poca
  aggressione, oppure tanta aggressione senza avanzamento).
- **Inversioni dopo eccessi:** riconoscere l'esaurimento, cioè l'aggressione che resta estrema
  mentre il prezzo non avanza più (assorbimento), comprese le cascate di vendite forzate.
- **Trend e momentum:** la persistenza. Uno sbilanciamento che dura ore indica un flusso costante.

### §6.2 · Leva 4a · Flusso degli scambi (famiglia 12)

**Dato.** Ogni candela Binance contiene il *taker buy volume*, cioè il volume comprato a mercato.
- delta = 2 × taker_buy − volume.
- Ha lo stesso storico delle candele: nessun accorciamento del train.
- ccxt (`fetch_ohlcv`) scarta questo campo: va letto dalla risposta grezza dell'API di Binance.
- Su Bybit va verificato se il campo c'è.
- I singoli scambi (`fetch_trades`) non servono: sugli orizzonti di 1–8 ore la somma per candela
  basta.

**Da sola o combinata.** Lo sbilanciamento da solo dice poco, perché spesso racconta lo stesso
movimento che mostra già il prezzo. L'informazione sta nel **rapporto fra l'aggressione e la
risposta del prezzo**:

| aggressione | prezzo | lettura |
|---|---|---|
| forte | avanza | spinta sostenuta → continuazione |
| forte | non avanza | assorbimento → spinta che si esaurisce |
| debole | avanza | movimento vuoto → rottura fragile |
| di segno opposto | avanza | divergenza |

Il modello è poco profondo e il segnale è debole, quindi le combinazioni gli si danno già
calcolate.

**Feature della famiglia 12** (circa 10, finestre in ore fissate prima, come il catalogo).
Formule definitive nella proposta di Claude Code:
1. `sbil_W`, W = 4, 24, 168 h: Σdelta / Σvolume (da −1 a +1).
2. `sbil_barra_k`, k = 0, 1, 2: sbilanciamento della candela del trigger e delle due precedenti.
3. `accordo_W`, W = 4, 24 h: sbil_W × segno del rendimento cumulato su W (positivo = sostenuto,
   negativo = divergenza).
4. `aggr_mov_24h`: |Σdelta| / Σ|r| su 24 h, diviso la sua mediana su 720 h (alto = assorbimento).
5. `sbil_pct_24h`: percentile di sbil_24h su 720 h.

Stesse regole delle altre famiglie: causali, adimensionali, collaudo di lookahead e di
degenerazione, selezione con sonde e regola B. Se non portano informazione, la selezione le
toglie.

**Ordine:**
1. V4 scarica taker_buy insieme alle candele (docs/77, lavoro 4).
2. Famiglia 12 nel catalogo del ML.
3. Prova: stessi trigger e stesso H, `GUARDA_OOS = False`, passo 7b prima e dopo la famiglia 12.

### §6.3 · Leva 4b · Sbilanciamento del book

**Dato.** ccxt dà solo il book di adesso (`fetch_order_book`, `watch_order_book`): senza storico
non si addestra né si fa backtest. Lo storico c'è nei file pubblici di Binance (futures USD-M,
**bookDepth**). Sono istantanee della profondità cumulata a fasce di prezzo, ±1% … ±5%.

**Feature possibile:** (liquidità sotto − liquidità sopra) / totale, a una o più fasce, letta
all'ultima istantanea prima della chiusura della candela.

**Limiti:**
- lo storico parte più tardi delle candele (probabilmente dal 2023, da verificare);
- le fasce sono larghe: non è il book fine;
- possibili interruzioni della raccolta (per esempio bookTicker da aprile/maggio 2024), da
  controllare file per file.

**Effetto sul ML.** Con la regola dei NaN, la feature accorcia il train a partire dall'inizio
del suo storico. Per questo è un **esperimento separato**, non parte della famiglia 12.

**Primo passo, prima di qualunque decisione:** un download di prova di bookDepth BTCUSDT perp.
Deve dire da quando parte lo storico, ogni quanto c'è un'istantanea, le fasce disponibili, i
buchi e il peso dei file. Solo dopo si decide se vale l'esperimento.
