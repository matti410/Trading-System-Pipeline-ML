"""
Condizioni di FILTER.

Un filtro viene sempre messo in AND con la condizione di Entry: la
strategia entra solo quando l'Entry E TUTTI i filtri assegnati sono veri
contemporaneamente. Stessa firma delle altre condizioni: funzione(df) ->
Series booleana.

Adattato allo stile V4 (17-18/9): niente decoratori applicati alla
definizione — a differenza della versione precedente, la registrazione nel
motore avviene tramite `registra_filtri()`, chiamata a mano nel notebook.
Stesso pattern di `registra_trigger_long()` in entry_long.py e
`registra_exit_long()` in exit_long.py: un dizionario nome -> (funzione,
direction), e un bridge idempotente verso `engine.registry.register_filter`.


LA DIREZIONE DI UN FILTRO
-------------------------
`register_filter(nome, direction=...)` dichiara a quale lato del mercato
il filtro appartiene:

    +1  descrive uno stato di mercato RIALZISTA
    -1  descrive uno stato di mercato RIBASSISTA
     0  neutro rispetto alla direzione (volatilita', volume, regime)

Serve alla grid search dei filtri: su un sistema solo long si provano i
neutri piu' quelli +1, senza sprecare campione su filtri che per
costruzione lavorano contro l'ingresso.

ATTENZIONE, ED E' IL PUNTO PIU' DELICATO DEL FILE. `direction` descrive
lo STATO DI MERCATO che il filtro individua, non il verso dell'entry con
cui va accoppiato. Per un'entry di CONTINUAZIONE i due coincidono: un
Marubozu rialzista vuole contesto rialzista. Per un'entry di INVERSIONE
si rovesciano: un Engulfing rialzista di esaurimento vuole contesto
RIBASSISTA, cioe' un filtro -1 su un ingresso long.

Quindi `direction` e' un DEFAULT ragionevole, non una regola. Su entry di
inversione va usato l'insieme completo dei filtri e lasciato decidere ai
dati — che era la scelta originale documentata in F8/F9 e che resta
valida.

Nota per questo percorso (V4, 18/9): la pipeline Trend Following usa
`direction` cosi' com'e' (filtro e entry stessa direzione = conferma). La
pipeline Mean Reverting, quando riparte, la usa rovesciata (filtro
opposto all'entry) — vedi `ROADMAP_RICERCA.md`, Passo 5.


COPPIE DI FILTRI (`pair`) — AGGIUNTA 18/9/2026
------------------------------------------------
Un filtro come F12_EXTENDED_UP/F13_EXTENDED_DOWN e' una singola idea
("il prezzo e' esteso rispetto alla media") che ha due facce, una per
direzione. Testarli come due righe indipendenti nella grid search (F12 sul
solo long, F13 sul solo short, ciascuna a se') e' due esperimenti diversi
("l'estensione aiuta il long da sola?", "aiuta lo short da sola?"), non lo
stesso esperimento di un sistema che richiede l'estensione coerente su
entrambe le gambe. Per questo le coppie dichiarate qui sotto vengono
raccolte in un'UNICA riga dalla grid search (vedi
`engine.filter_search_bt`): il membro +1 va sul lato long, il -1 sul lato
short, nella stessa riga — stessa logica gia' in uso per `exit_rule_pairs`.

COME REGISTRARE UN FILTRO NUOVO (leggi questo prima di aggiungerne uno):

1. NEUTRO/SIMMETRICO PER NATURA (misura una magnitudine, non ha verso —
   volatilita', volume, coerenza) -> `direction=0`, NESSUN `pair`. Si
   applica sempre a entrambi i lati attivi, come oggi.

2. DIREZIONALE CON UNO SPECULARE NATURALE -> scrivi le due funzioni
   insieme, stesso `pair`, direction opposte: la versione "su" con
   `direction=1, pair="ETICHETTA"`, la versione "giu'" con
   `direction=-1, pair="ETICHETTA"`. La grid search le trova da sola e le
   testa in un'unica riga.

3. DIREZIONALE SENZA SPECULARE (idea a senso unico, o lo speculare non e'
   ancora stato scritto) -> `direction=±1`, SENZA `pair`. Resta testato da
   solo sul proprio unico lato compatibile — comportamento invariato,
   nessun obbligo di scrivere sempre una coppia.

4. Se in futuro scrivi lo speculare mancante di un filtro del punto 3
   (diventa punto 2): aggiungi `pair="ETICHETTA"` a ENTRAMBE le
   registrazioni — quella nuova e quella vecchia, va editata anche lei.
   Da quel momento vengono raccolte insieme automaticamente, nessun'altra
   modifica al motore.

GUARDIA EREDITATA da `list_exit_pairs()`: se un'etichetta `pair` finisce
condivisa da piu' di un filtro +1 o piu' di un -1 (ambiguo), quella coppia
viene scartata con un avviso stampato, non un errore — i suoi membri, se
richiesti singolarmente, tornano al comportamento standalone del punto 3.

SE IL TRIGGER DI INGRESSO E' A UN SOLO LATO (solo long o solo short): la
grid search applica solo il membro della coppia compatibile con quel lato
e etichetta la riga col nome del singolo filtro, non con l'etichetta della
coppia — la coppia come unita' unica ha senso solo quando entrambi i lati
del trigger sono davvero attivi. Nessun errore, nessuna riga inventata per
il lato assente.


PROVENIENZA E SCARTI
--------------------
I filtri da F10 in poi derivano dai "101 Formulaic Alphas" (Kakushadze,
WorldQuant, 2015 - arXiv:1601.00991). Vivevano in un file separato per
provenienza; sono stati uniti qui perche' dividere per provenienza e' un
criterio debole: si cercano "i filtri di trend", non "i filtri Alpha101".

Del set originale sono stati ELIMINATI, e non vanno reintrodotti senza
rifare l'analisi:

A8_OPEN_RETURNS_SHOCK
    `sum(open,5) * sum(returns,5)`: su BTC il primo fattore vale
    centinaia di migliaia e trenda con il prezzo, il secondo oscilla
    intorno a zero. Il prodotto e' dominato dal LIVELLO del prezzo, non
    dallo shock di rendimento che dovrebbe misurare. Serie non
    stazionaria: il filtro misura il 2021 contro il 2026.

A20_GAP_ANOMALY
    Doppio problema. Su cripto H24 il gap d'apertura non esiste: `Open`
    coincide praticamente sempre con il `Close` precedente, quindi i tre
    termini sono rumore intorno a zero. E c'e' un bug di soglia: il
    prodotto di tre percentili in [0,1] supera 0.5 in una frazione
    minima dei casi, il filtro non scatterebbe quasi mai.

A30 - parte volume
    `sum(volume,5)/sum(volume,20)` poggia sul tick volume di MetaTrader,
    che su BTCUSD CFD e' il numero di variazioni di prezzo registrate dal
    feed del singolo broker, non il volume scambiato. La parte sulla
    coerenza delle ultime variazioni e' invece pulita e sopravvive da
    sola, qui sotto, come F15_SHORT_TERM_CONSISTENCY.

Sono state eliminate anche tutte le entry e le bidirezionali Alpha101:
rank cross-sezionale non trasferibile su asset singolo, tick volume del
broker inaffidabile, segno fisso che e' una magnitudine e non una
direzione.


NOTA SUL VWAP (riguarda F16/F17)
--------------------------------
Su un asset H24 l'ancoraggio giornaliero del VWAP e' una convenzione, non
un fatto. Mezzanotte UTC e' lo standard piu' diffuso su cripto, ma se il
feed IC Markets e' su fuso broker (tipicamente UTC+2/+3) il reset cade a
meta' sessione asiatica e il filtro si sposta in modo non banale.
Verificare su quale ora ancora `engine.vwap_ops.vwap_anchored_daily`
prima di dare peso ai risultati di questi due filtri.
"""
import numpy as np
import pandas as pd

from engine.alpha_ops import delta, ts_rank, ts_sum
#from engine.livelli import al_ultima_apertura, quarantena_cached, range_finestra
#from engine.sessioni import in_sessione


# ======================================================================
# Neutri - volatilita', volume, regime
# ======================================================================

def filter_adx_above(df: pd.DataFrame, threshold: float = 20.0) -> pd.Series:
    """ADX sopra soglia. Misura la FORZA del trend, non il verso: neutro."""
    return df["adx"] > threshold


def filter_low_volatility(df: pd.DataFrame, window: int = 500,
                          percentile: float = 0.35) -> pd.Series:
    """
    Volatilita' BASSA

    Una soglia fissa in unita' di prezzo non e' trasferibile: il valore
    giusto per EURUSD non lo e' per XAUUSD, e non lo e' nemmeno per EURUSD
    fra due regimi di volatilita' diversi. Si usa il percentile rolling
    dell'ATR sulla propria storia recente, cosi' il filtro si auto-adatta
    a strumento, timeframe e regime - stessa scelta fatta in
    F14_LOW_DRIFT_REGIME per lo stesso motivo.

    window=500 su M15 sono circa 5 giorni di contrattazione.
    percentile=0.5 seleziona la meta' piu' volatile; alza a 0.7 per essere
    piu' severo.
    """
    soglia = df["atr"].rolling(int(window)).quantile(percentile)
    return df["atr"] <= soglia


def filter_mid_volatility(df: pd.DataFrame, window: int = 500,
                          percentile: float = 0.75) -> pd.Series:
    """
    Volatilita' MEDIA

    Una soglia fissa in unita' di prezzo non e' trasferibile: il valore
    giusto per EURUSD non lo e' per XAUUSD, e non lo e' nemmeno per EURUSD
    fra due regimi di volatilita' diversi. Si usa il percentile rolling
    dell'ATR sulla propria storia recente, cosi' il filtro si auto-adatta
    a strumento, timeframe e regime - stessa scelta fatta in
    F14_LOW_DRIFT_REGIME per lo stesso motivo.

    window=500 su M15 sono circa 5 giorni di contrattazione.
    percentile=0.5 seleziona la meta' piu' volatile; alza a 0.7 per essere
    piu' severo.
    """
    soglia = df["atr"].rolling(int(window)).quantile(percentile)
    return df["atr"] <= soglia


def filter_high_volatility(df: pd.DataFrame, window: int = 500,
                          percentile: float = 0.75) -> pd.Series:
    """
    Volatilita' ALTA

    Una soglia fissa in unita' di prezzo non e' trasferibile: il valore
    giusto per EURUSD non lo e' per XAUUSD, e non lo e' nemmeno per EURUSD
    fra due regimi di volatilita' diversi. Si usa il percentile rolling
    dell'ATR sulla propria storia recente, cosi' il filtro si auto-adatta
    a strumento, timeframe e regime - stessa scelta fatta in
    F14_LOW_DRIFT_REGIME per lo stesso motivo.

    window=500 su M15 sono circa 5 giorni di contrattazione.
    percentile=0.5 seleziona la meta' piu' volatile; alza a 0.7 per essere
    piu' severo.
    """
    soglia = df["atr"].rolling(int(window)).quantile(percentile)
    return df["atr"] > soglia


def filter_volume_above_avg(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume sopra la propria media mobile."""
    return df["Volume"] > df["Volume"].rolling(window).mean()


def filter_tall_candle(df: pd.DataFrame, window: int = 20,
                       multiplier: float = 1.5) -> pd.Series:
    """
    Range (High-Low) della barra corrente sopra `multiplier` volte la
    propria media mobile recente -> candela "alta"/di convinzione.
    E' il fattore con l'evidenza piu' forte trovata nella ricerca (piu'
    robusto del volume secondo i test di Bulkowski): non un filtro
    generico qualsiasi, e' quello da provare per primo.
    """
    candle_range = df["High"] - df["Low"]
    avg_range = candle_range.rolling(window).mean()
    return candle_range > avg_range * multiplier


def filter_volume_above_avg_50(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Stessa logica di F3, finestra 50 invece di 20 -> l'ipotesi originale
    di volume "sopra media" con orizzonte piu' lungo, tenuta come
    variante a parita' di logica.
    """
    return df["Volume"] > df["Volume"].rolling(window).mean()


def filter_volume_climax(df: pd.DataFrame, window: int = 50,
                         quantile: float = 0.9) -> pd.Series:
    """
    Volume sopra il `quantile`-esimo percentile della propria finestra
    recente (default 90), non semplicemente sopra la media come F3/F6.
    Copre l'ipotesi diversa di "climax/esaurimento" (tipica dei pattern
    di reversal), distinta da "partecipazione sostenuta sopra media"
    (piu' adatta ai pattern di continuation).
    """
    threshold = df["Volume"].rolling(window).quantile(quantile)
    return df["Volume"] > threshold


def filter_low_drift_regime(
    df: pd.DataFrame,
    window: float = 100,
    rank_window: float = 500,
    percentile: float = 0.3,
) -> pd.Series:
    """
    (ex A24) Alpha#24 del paper: usa
    `delta(sum(close,100)/100,100)/delay(close,100) < 0.05` come innesco
    per scegliere fra due rami di comportamento. Qui isolato come filtro
    di regime a se'.

    DUE CORREZIONI rispetto alla versione originale.

    1. Il confronto era unilaterale (`drift < 0.05`) e quindi non
       misurava affatto la bassa direzionalita': un mercato in forte
       discesa ha drift molto negativo e passava il filtro senza
       problemi. Diceva "non fortemente rialzista", non "poco
       direzionale". Ora si usa il valore assoluto.

    2. La soglia 0.05 e' un default da dati daily. Su M15 il drift a 100
       barre (25 ore) non la raggiunge praticamente mai: verificato
       empiricamente su 20.000 barre, il filtro risultava True nel 99%
       dei casi - cioe' non filtrava niente. Al posto della magnitudine
       fissa si usa il PERCENTILE ROLLING del drift sulla propria storia
       recente: il filtro si auto-adatta al timeframe e al regime di
       volatilita', e non ha piu' nessun numero calibrato a mano sulla
       scala del prezzo.

    True quando il drift assoluto della media a `window` barre e' nel
    `percentile` piu' basso delle ultime `rank_window` barre - regime di
    consolidamento, che tipicamente precede un breakout e non e' di per
    se' un trigger di ingresso.

    Neutro per costruzione: il valore assoluto rende la domanda "quanto
    poco si muove", che non ha verso.
    """
    sma = df["Close"].rolling(int(window)).mean()
    drift = (delta(sma, window) / df["Close"].shift(int(window))).abs()
    return ts_rank(drift, rank_window) < percentile


def filter_short_term_consistency(
    df: pd.DataFrame,
    consistency_threshold: int = 2,
) -> pd.Series:
    """
    (ex A30) Alpha#30 del paper, ridotta alla sola componente di coerenza
    di segno: `sign(close-delay(close,1)) + sign(delay(close,1)-delay(close,2))
    + sign(delay(close,2)-delay(close,3))`.

    True quando almeno `consistency_threshold` delle ultime 3 variazioni
    sono concordi in segno - micro-trend coerente a brevissimo termine,
    indipendente dal verso (si usa il valore assoluto della somma).

    Neutro per costruzione: il valore assoluto unisce le micro-serie al
    rialzo e quelle al ribasso, ed e' corretto perche' la domanda e' "c'e'
    coerenza?", non "in che verso". Si potrebbe spaccare in due per
    guadagnare informazione, ma si perderebbe la domanda originale.

    Il fattore di ponderazione sul volume del paper e' stato rimosso: vedi
    la nota sugli scarti in cima al file.
    """
    segno = np.sign(df["Close"].diff().fillna(0.0))
    coerenza = (segno + segno.shift(1) + segno.shift(2)).abs()
    return coerenza >= consistency_threshold


# ======================================================================
# Direzionali - a coppie speculari
# ======================================================================

def filter_uptrend_context(df: pd.DataFrame) -> pd.Series:
    """
    Prezzo sopra l'EMA50 -> contesto di trend rialzista in corso.

    NOTA STORICA, ancora valida. La versione precedente non aveva
    direction, e il docstring motivava la scelta: sta alla ricerca
    combinatoria scoprire empiricamente se un dato pattern funziona
    meglio ALLINEATO al trend (continuation, es. Marubozu/Belt-hold) o
    CONTRO il trend precedente (reversal da esaurimento, es. Engulfing/
    Harami). Quel ragionamento non e' stato buttato: direction=1 dice
    "questo filtro individua un mercato che sale", che e' un fatto, e
    resta un DEFAULT per gli ingressi di continuazione. Su un'entry di
    inversione vanno provati tutti i filtri, non solo quelli concordi.
    """
    return df["Close"] > df["ema50"]


def filter_downtrend_context(df: pd.DataFrame) -> pd.Series:
    """Speculare di F8: prezzo sotto l'EMA50 -> trend ribassista in corso."""
    return df["Close"] < df["ema50"]


def filter_trend_conviction_up(
    df: pd.DataFrame,
    lookback: float = 350,
    rank_window: float = 500,
    threshold: float = 0.7,
) -> pd.Series:
    """
    (ex A19_TREND_CONVICTION) Alpha#19 del paper:
    `(-1*sign(...)) * (1+rank((1+sum(returns,250))))`. Si usa solo la
    seconda componente; gli `1+` esterni si elidono nel confronto con una
    soglia, quindi la condizione e' "il rendimento cumulato sulle ultime
    `lookback` barre e' nel percentile alto della propria storia recente".

    CORREZIONE IMPORTANTE. Il docstring precedente dichiarava che si
    usava solo la MAGNITUDO del trend di fondo, "mai il fattore di
    segno", e concludeva che il filtro fosse indipendente dal verso. Non
    lo era: `ts_sum(returns, lookback)` e' il rendimento CUMULATO, una
    grandezza con segno, non una magnitudine. Stare nel percentile alto
    significa "il cumulato e' vicino al suo massimo recente", cioe'
    deriva rialzista. Il filtro e' quindi direzionale long, ed e' ora
    dichiarato tale. F11 e' lo speculare che mancava.

    DUE FINESTRE DISTINTE, e vanno tenute distinte:
      lookback     su quante barre si somma il rendimento (350)
      rank_window  su quante barre si calcola il PERCENTILE (500)

    `rank_window` era 20, cioe' un percentile stimato su venti
    osservazioni. Allineato a 500 come in F14, che fa la stessa
    operazione. Alzare `lookback` tenendo `rank_window` corta peggiora le
    cose: un cumulato a 350 barre si muove lentamente, su 20 barre cambia
    pochissimo, e il percentile finirebbe per misurare il contributo
    delle ultime manciate di barre invece della posizione del trend.

    Riscaldamento: lookback + rank_window barre (circa 9 giorni su M15).
    """
    returns = df["Close"].pct_change()
    return ts_rank(ts_sum(returns, lookback), rank_window) > threshold


def filter_trend_conviction_down(
    df: pd.DataFrame,
    lookback: float = 350,
    rank_window: float = 500,
    threshold: float = 0.7,
) -> pd.Series:
    """
    Speculare di F10: il rendimento cumulato e' nel percentile BASSO della
    propria storia recente - deriva ribassista di fondo.

    La soglia si specchia (`< 1 - threshold`) invece di essere un secondo
    numero indipendente, cosi' i due filtri restano simmetrici per
    costruzione: con threshold=0.7 F10 prende il 30% piu' alto e F11 il
    30% piu' basso. Cambiando un solo parametro si muovono entrambi,
    ed e' impossibile tararli di fino uno contro l'altro per sbaglio.
    """
    returns = df["Close"].pct_change()
    return ts_rank(ts_sum(returns, lookback), rank_window) < (1.0 - threshold)


def filter_extended_up(
    df: pd.DataFrame,
    long_window: float = 8,
    short_window: float = 2,
) -> pd.Series:
    """
    (ex A21_PRICE_EXTENDED_FROM_MEAN, ramo rialzista) Alpha#21 del paper:
    struttura a tre rami sul confronto fra media mobile lunga (piu' o meno
    la deviazione standard) e media mobile corta. Qui ridotta al nucleo di
    "estensione di prezzo".

    PERCHE' E' STATO SPACCATO IN DUE. La versione originale restituiva
    `estesa_giu | estesa_su`, cioe' un OR fra due rami OPPOSTI: si
    accendeva sia quando il prezzo era esteso al rialzo sia quando lo era
    al ribasso. Per un'entry direzionale quelle due popolazioni non sono
    equivalenti - una e' continuazione, l'altra e' ritorno alla media - e
    mescolarle in un unico booleano produce un filtro che seleziona due
    gruppi che si contraddicono, col risultato medio vicino a niente
    qualunque sia il merito delle due meta'.

    Il resto della base di codice questo problema lo aveva gia' risolto:
    F18/F19 sono la stessa idea gia' divisa in due.

    True quando la media corta e' uscita SOPRA la banda
    SMA(long) + stddev(long). Normalizzato sulla deviazione standard,
    quindi scale-free e auto-adattivo alla volatilita' del momento.

    Il ramo sul volume del paper e' omesso: e' coperto separatamente da
    F3_VOLUME_ABOVE_AVG.
    """
    sma_long = df["Close"].rolling(int(long_window)).mean()
    std_long = df["Close"].rolling(int(long_window)).std()
    sma_short = df["Close"].rolling(int(short_window)).mean()
    return sma_short > (sma_long + std_long)


def filter_extended_down(
    df: pd.DataFrame,
    long_window: float = 8,
    short_window: float = 2,
) -> pd.Series:
    """
    Speculare di F12: la media corta e' scesa SOTTO la banda
    SMA(long) - stddev(long).
    """
    sma_long = df["Close"].rolling(int(long_window)).mean()
    std_long = df["Close"].rolling(int(long_window)).std()
    sma_short = df["Close"].rolling(int(short_window)).mean()
    return sma_short < (sma_long - std_long)


def filter_price_above_vwap(df: pd.DataFrame) -> pd.Series:
    """
    (ex A41_PRICE_ABOVE_VWAP) Alpha#41 del paper: sqrt(high*low) - vwap,
    qui come filtro booleano di posizionamento. True quando la media
    geometrica di High/Low e' sopra il VWAP - regime "prezzo forte
    rispetto al fair value della sessione".

    Concettualmente il piu' adatto all'intraday di tutto il gruppo, ed e'
    l'unico che usa il volume senza esserne danneggiato: il VWAP e' una
    media di prezzi PESATA per volume, quindi molto piu' robusto alla
    qualita' del tick volume di quanto lo sia un rapporto di volumi.

    Richiede df["vwap"] - vedi engine.vwap_ops.vwap_anchored_daily e la
    nota sull'ancoraggio in cima al file.
    """
    mid_geometrico = (df["High"] * df["Low"]) ** 0.5
    return mid_geometrico > df["vwap"]


def filter_price_below_vwap(df: pd.DataFrame) -> pd.Series:
    """Speculare di F16: prezzo sotto il VWAP di sessione."""
    mid_geometrico = (df["High"] * df["Low"]) ** 0.5
    return mid_geometrico < df["vwap"]


def filter_high_near_range_top(
    df: pd.DataFrame,
    window: float = 9,
    threshold: float = 0.25,
) -> pd.Series:
    """
    (ex A4_HIGH_NEAR_RANGE_TOP) True quando il massimo della barra e'
    nella parte ALTA del proprio range recente.

    Si usa ts_rank(-High) invece di ts_rank(High) > 1-threshold perche'
    ts_rank su `window` barre assume solo `window` valori discreti: con
    window=9 e threshold=0.25 il filtro basso cattura le posizioni 1 e 2,
    mentre "> 0.75" ne catturerebbe TRE (7, 8, 9). Negando la serie il
    rank riparte dall'alto e la soglia 0.25 cattura esattamente le due
    posizioni piu' alte - stessa selettivita' del gemello, che e' il punto.

    Su direction: e' un filtro POSIZIONALE, non di trend. direction=1
    perche' descrive prezzo forte, ma su un'entry di inversione
    (esaurimento) l'accoppiamento utile e' l'opposto. Vedi la nota in
    cima al file.
    """
    return ts_rank(-df["High"], window) < threshold


def filter_low_near_range_bottom(
    df: pd.DataFrame,
    window: float = 9,
    threshold: float = 0.25,
) -> pd.Series:
    """
    (ex A4_LOW_NEAR_RANGE_BOTTOM) Alpha#4 del paper:
    -1*Ts_Rank(rank(low),9). Il rank cross-sezionale interno e' rimosso
    (privo di senso su asset singolo, l'outer ts_rank basta): qui si usa
    direttamente ts_rank(Low, window).

    True quando il minimo della barra e' nella parte bassa del proprio
    range recente - regime di possibile supporto/ipervenduto.

    ATTENZIONE alla granularita': ts_rank su `window` barre assume solo
    `window` valori discreti (1/window, 2/window, ..., 1), quindi il
    minimo raggiungibile e' 1/window. Con window=9 il minimo e' ~0.111 e
    una soglia sotto quel valore non farebbe mai scattare il filtro. Il
    default 0.25 include le due posizioni piu' basse.

    Stessa avvertenza di F18: filtro posizionale, direction e' un default.
    """
    return ts_rank(df["Low"], window) < threshold


FILTRI = {
    "F1_ADX_ABOVE":              (filter_adx_above, 0, None),
    "F2_LOW_VOLATILITY":         (filter_low_volatility, 0, None), #
    "F2_MID_VOLATILITY":         (filter_mid_volatility, 0, None),
    "F2_HIGH_VOLATILITY":        (filter_high_volatility, 0, None),
    "F3_VOLUME_ABOVE_AVG":       (filter_volume_above_avg, 0, None),
    "F5_TALL_CANDLE":            (filter_tall_candle, 0, None),
    "F6_VOLUME_ABOVE_AVG_50":    (filter_volume_above_avg_50, 0, None),
    "F7_VOLUME_CLIMAX":          (filter_volume_climax, 0, None),
    "F14_LOW_DRIFT_REGIME":      (filter_low_drift_regime, 0, None),
    "F15_SHORT_TERM_CONSISTENCY": (filter_short_term_consistency, 0, None),
    "F8_UPTREND_CONTEXT":        (filter_uptrend_context, 1, "TREND_CONTEXT"),
    "F9_DOWNTREND_CONTEXT":      (filter_downtrend_context, -1, "TREND_CONTEXT"),
    "F10_TREND_CONVICTION_UP":   (filter_trend_conviction_up, 1, "TREND_CONVICTION"),
    "F11_TREND_CONVICTION_DOWN": (filter_trend_conviction_down, -1, "TREND_CONVICTION"),
    "F12_EXTENDED_UP":           (filter_extended_up, 1, "EXTENDED_FROM_MEAN"),
    "F13_EXTENDED_DOWN":         (filter_extended_down, -1, "EXTENDED_FROM_MEAN"),
    "F16_PRICE_ABOVE_VWAP":      (filter_price_above_vwap, 1, "VWAP_POSITION"),
    "F17_PRICE_BELOW_VWAP":      (filter_price_below_vwap, -1, "VWAP_POSITION"),
    "F18_HIGH_NEAR_RANGE_TOP":   (filter_high_near_range_top, 1, "RANGE_POSITION"),
    "F19_LOW_NEAR_RANGE_BOTTOM": (filter_low_near_range_bottom, -1, "RANGE_POSITION"),
}


def registra_filtri():
    """
    Registra tutti i filtri di questo file nel motore (engine.registry),
    cosi' filter_search_bt li trova da sola tramite get_filter(nome), e le
    8 coppie tramite list_filter_pairs().

    Va chiamata una volta prima di usare i filtri. E' sicura da richiamare
    piu' volte nella stessa sessione: i nomi gia' registrati vengono
    saltati con un avviso, non sollevano errore.
    """
    from engine.registry import register_filter, list_filters

    gia_presenti = set(list_filters())
    nuovi = 0
    for nome, (funzione, direction, pair) in FILTRI.items():
        if nome in gia_presenti:
            print(f"[registra_filtri] '{nome}' gia' registrato, salto.")
            continue
        register_filter(nome, direction=direction, pair=pair)(funzione)
        nuovi += 1
    print(f"[registra_filtri] {nuovi} filtri registrati "
          f"({len(FILTRI) - nuovi} gia' presenti).")
