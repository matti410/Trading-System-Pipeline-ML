"""
Registro delle condizioni Entry / Exit / Filter / Bidirectional Entry.

Le condizioni sono definite dall'utente come funzioni Python in file esterni
(cartella `conditions/`) e registrate tramite i decorator qui sotto.

Il motore NON conosce il contenuto delle condizioni: sa solo che ogni
condizione registrata è una funzione che riceve il DataFrame di mercato e
restituisce una Series della stessa lunghezza.

Quattro tipi di condizione:

- Entry / Exit: restituiscono un booleano (True dove la condizione è
  verificata) e portano una `direction` FISSA obbligatoria: 1 = long,
  -1 = short. Usate dalla ricerca automatica (Pass 1, 2, 3...): Entry ed
  Exit si combinano solo se condividono la stessa direction.
- Filter: restituisce un booleano, nessuna direction (si applica in AND
  sull'entry indipendentemente da long/short).
- Bidirectional Entry: restituisce direttamente un segnale CON SEGNO
  (+1 = segnale long, -1 = segnale short, 0 = nessun segnale) nella STESSA
  colonna. Non ha una direction fissa: è la funzione stessa a deciderla,
  bar per bar. Usata SOLO in Strategy Inspection (analisi di dettaglio di
  una singola strategia scelta a mano), mai nella ricerca automatica.

Aggiungere una nuova condizione = aggiungere una funzione decorata in un
file .py della cartella conditions/. Il core (registry, combination,
backtest, ranking, inspection) non va mai toccato.
"""
from __future__ import annotations
from typing import Callable, Dict
import pandas as pd

ConditionFunc = Callable[[pd.DataFrame], pd.Series]

_ENTRY_REGISTRY: Dict[str, ConditionFunc] = {}
_EXIT_REGISTRY: Dict[str, ConditionFunc] = {}
_FILTER_REGISTRY: Dict[str, ConditionFunc] = {}
_BIDIR_ENTRY_REGISTRY: Dict[str, ConditionFunc] = {}

_ENTRY_DIRECTION: Dict[str, int] = {}
# Direzione dei filtri: +1 long, -1 short, 0 neutro. Chi non la dichiara
# resta 0, quindi il comportamento preesistente non cambia.
_FILTER_DIRECTION: Dict[str, int] = {}
_FILTER_PAIR: Dict[str, str] = {}
_EXIT_DIRECTION: Dict[str, int] = {}
_EXIT_PAIR: Dict[str, str] = {}


def _validate_filter_direction(direction: int) -> None:
    if direction not in (1, -1, 0):
        raise ValueError(
            "direction di un filtro deve essere 1 (long), -1 (short) "
            "o 0 (neutro)."
        )


def _validate_direction(direction: int) -> None:
    if direction not in (1, -1):
        raise ValueError("direction deve essere 1 (long) o -1 (short).")


def register_entry(name: str, direction: int):
    """Decorator: registra una condizione di Entry. `direction` è obbligatoria: 1=long, -1=short."""
    _validate_direction(direction)

    def decorator(func: ConditionFunc) -> ConditionFunc:
        if name in _ENTRY_REGISTRY:
            raise ValueError(f"Entry '{name}' è già registrata. Usa un nome univoco.")
        _ENTRY_REGISTRY[name] = func
        _ENTRY_DIRECTION[name] = direction
        func.condition_name = name
        func.direction = direction
        return func
    return decorator


def register_exit(name: str, direction: int, pair: str | None = None):
    """
    Decorator: registra una condizione di Exit. `direction` è obbligatoria:
    1=long, -1=short.

    `pair` (opzionale) è un'etichetta comune usata per dire "questa exit long
    e questa exit short sono la stessa idea nelle due direzioni": due exit,
    una direction=1 e una direction=-1, con lo STESSO `pair`, diventano
    automaticamente una coppia utilizzabile nelle strategie bidirezionali
    (Pass 1 in modalità long_and_short). Senza `pair`, l'exit resta
    utilizzabile solo nella ricerca a direction fissa (only_long/only_short).
    """
    _validate_direction(direction)

    def decorator(func: ConditionFunc) -> ConditionFunc:
        if name in _EXIT_REGISTRY:
            raise ValueError(f"Exit '{name}' è già registrata. Usa un nome univoco.")
        _EXIT_REGISTRY[name] = func
        _EXIT_DIRECTION[name] = direction
        if pair is not None:
            _EXIT_PAIR[name] = pair
        func.condition_name = name
        func.direction = direction
        func.pair = pair
        return func
    return decorator


def register_filter(name: str, direction: int = 0, pair: str | None = None):
    """
    Decorator: registra una condizione di Filter. Si applica sempre in AND
    sull'entry.

    `direction` dichiara a quale lato del mercato il filtro appartiene:

        +1  filtro di trend rialzista, sensato solo su un'entry long
        -1  filtro di trend ribassista, sensato solo su un'entry short
         0  neutro rispetto alla direzione (volatilità, volume, regime)

    Il default è 0, quindi i filtri scritti prima di questo campo
    continuano a funzionare esattamente come prima.

    Serve alla grid search dei filtri: su un sistema solo long si provano
    i neutri più quelli +1, e non si spreca il campione su filtri che per
    costruzione lavorano contro l'ingresso.

    `pair` (opzionale) è un'etichetta comune usata per dire "questo filtro
    +1 e questo -1 sono la stessa idea nelle due direzioni" — stessa
    funzione di `pair` in `register_exit`. Due filtri, uno direction=1 e
    uno direction=-1, con lo STESSO `pair`, vengono raccolti
    automaticamente in un'unica riga da `list_filter_pairs()` (e dalla
    grid search dei filtri, che applica il membro +1 al lato long e il -1
    al lato short nella stessa riga, invece di due righe separate). Un
    filtro neutro (direction=0) con `pair` non ha senso — non c'è un verso
    da accoppiare — e solleva `ValueError`. Senza `pair`, il filtro resta
    testato da solo sul proprio lato: comportamento invariato.
    """
    _validate_filter_direction(direction)
    if pair is not None and direction == 0:
        raise ValueError(
            "Un filtro neutro (direction=0) non può avere un pair: "
            "pair serve solo ad accoppiare due direzioni opposte."
        )

    def decorator(func: ConditionFunc) -> ConditionFunc:
        if name in _FILTER_REGISTRY:
            raise ValueError(f"Filter '{name}' è già registrata. Usa un nome univoco.")
        _FILTER_REGISTRY[name] = func
        _FILTER_DIRECTION[name] = direction
        if pair is not None:
            _FILTER_PAIR[name] = pair
        func.condition_name = name
        return func
    return decorator


def register_bidirectional_entry(name: str):
    """
    Decorator: registra una condizione di Entry bidirezionale. La funzione
    NON restituisce un booleano: restituisce una Series con valori
    {-1, 0, 1} (short / nessun segnale / long) — è la funzione stessa a
    portare la direction, bar per bar. Usata solo in Strategy Inspection.
    """
    def decorator(func: ConditionFunc) -> ConditionFunc:
        if name in _BIDIR_ENTRY_REGISTRY:
            raise ValueError(f"Bidirectional entry '{name}' è già registrata. Usa un nome univoco.")
        _BIDIR_ENTRY_REGISTRY[name] = func
        func.condition_name = name
        return func
    return decorator


# Getter singoli (usati dal Backtesting Engine / Inspection)
def get_entry(name: str) -> ConditionFunc:
    return _ENTRY_REGISTRY[name]


def get_exit(name: str) -> ConditionFunc:
    return _EXIT_REGISTRY[name]


def get_filter(name: str) -> ConditionFunc:
    return _FILTER_REGISTRY[name]


def get_filter_direction(name: str) -> int:
    """+1 long, -1 short, 0 neutro. 0 anche per i filtri che non l'hanno
    dichiarata, che e' il comportamento storico."""
    return _FILTER_DIRECTION.get(name, 0)


def get_filter_pair(name: str) -> str | None:
    return _FILTER_PAIR.get(name)


def list_filter_pairs() -> list[tuple[str, str, str]]:
    """
    Scopre automaticamente le coppie di filtri valide: due filtri (uno
    direction=1, uno direction=-1) registrati con lo stesso `pair`.
    Ritorna una lista di tuple (pair_label, nome_filtro_up, nome_filtro_down).
    Un'etichetta `pair` usata su più di un filtro +1 o più di un -1
    (ambigua) viene scartata con un avviso stampato, non solleva errore —
    stesso comportamento di `list_exit_pairs()`.
    """
    by_pair: Dict[str, Dict[int, list[str]]] = {}
    for name, pair in _FILTER_PAIR.items():
        direction = _FILTER_DIRECTION[name]
        by_pair.setdefault(pair, {1: [], -1: []})[direction].append(name)

    pairs = []
    for pair, sides in sorted(by_pair.items()):
        ups, downs = sides[1], sides[-1]
        if len(ups) == 1 and len(downs) == 1:
            pairs.append((pair, ups[0], downs[0]))
        else:
            print(
                f"[list_filter_pairs] etichetta '{pair}' ignorata: serve esattamente "
                f"1 filtro +1 + 1 filtro -1 con questo pair (trovati {len(ups)} up, {len(downs)} down)."
            )
    return pairs


def get_bidirectional_entry(name: str) -> ConditionFunc:
    return _BIDIR_ENTRY_REGISTRY[name]


def get_entry_direction(name: str) -> int:
    return _ENTRY_DIRECTION[name]


def get_exit_direction(name: str) -> int:
    return _EXIT_DIRECTION[name]


def get_exit_pair(name: str) -> str | None:
    return _EXIT_PAIR.get(name)


def list_exit_pairs() -> list[tuple[str, str, str]]:
    """
    Scopre automaticamente le coppie di exit valide: due exit (una
    direction=1, una direction=-1) registrate con lo stesso `pair`.
    Ritorna una lista di tuple (pair_label, nome_exit_long, nome_exit_short).
    Un'etichetta `pair` usata su più di una exit long o più di una short
    (ambigua) viene scartata con un avviso stampato, non solleva errore.
    """
    by_pair: Dict[str, Dict[int, list[str]]] = {}
    for name, pair in _EXIT_PAIR.items():
        direction = _EXIT_DIRECTION[name]
        by_pair.setdefault(pair, {1: [], -1: []})[direction].append(name)

    pairs = []
    for pair, sides in sorted(by_pair.items()):
        longs, shorts = sides[1], sides[-1]
        if len(longs) == 1 and len(shorts) == 1:
            pairs.append((pair, longs[0], shorts[0]))
        else:
            print(
                f"[list_exit_pairs] etichetta '{pair}' ignorata: serve esattamente "
                f"1 exit long + 1 exit short con questo pair (trovate {len(longs)} long, {len(shorts)} short)."
            )
    return pairs


# Elenchi (usati dalla Strategy Generation / dal notebook)
def list_entries(direction: int | None = None) -> list[str]:
    """Senza `direction`: tutte le entry. Con direction=1 o -1: solo quelle di quella direzione."""
    if direction is None:
        return sorted(_ENTRY_REGISTRY)
    _validate_direction(direction)
    return sorted(n for n in _ENTRY_REGISTRY if _ENTRY_DIRECTION[n] == direction)


def list_exits(direction: int | None = None) -> list[str]:
    """Senza `direction`: tutte le exit. Con direction=1 o -1: solo quelle di quella direzione."""
    if direction is None:
        return sorted(_EXIT_REGISTRY)
    _validate_direction(direction)
    return sorted(n for n in _EXIT_REGISTRY if _EXIT_DIRECTION[n] == direction)


def list_filters(direction=None) -> list[str]:
    """
    Nomi dei filtri registrati.

    direction=None  tutti (comportamento storico)
    direction=1     solo i filtri long
    direction=-1    solo i filtri short
    direction=0     solo i neutri

    Per un sistema solo long si usa list_filters(0) + list_filters(1).
    """
    if direction is None:
        return sorted(_FILTER_REGISTRY)
    _validate_filter_direction(direction)
    return sorted(n for n in _FILTER_REGISTRY
                  if _FILTER_DIRECTION.get(n, 0) == direction)


def list_bidirectional_entries() -> list[str]:
    return sorted(_BIDIR_ENTRY_REGISTRY)


def clear_registry() -> None:
    """
    Utile in notebook per ricaricare le condizioni da zero dopo una modifica.

    FIX: mancava _EXIT_PAIR.clear(). Senza questa riga, rinominare o
    rimuovere un'exit registrata con `pair=...` e poi richiamare
    load_conditions() lasciava un'etichetta "fantasma" in _EXIT_PAIR (il
    nome vecchio, non più presente in _EXIT_DIRECTION dopo il reset) che
    faceva esplodere list_exit_pairs() con un KeyError al primo utilizzo
    successivo — quindi Pass 1 in modalità long_and_short.

    Stesso motivo per _FILTER_PAIR.clear() (aggiunta 18/9/2026 insieme al
    parametro `pair` di register_filter): senza, lo stesso bug si
    ripresenterebbe identico su list_filter_pairs().
    """
    _ENTRY_REGISTRY.clear()
    _FILTER_DIRECTION.clear()
    _FILTER_PAIR.clear()
    _EXIT_REGISTRY.clear()
    _FILTER_REGISTRY.clear()
    _BIDIR_ENTRY_REGISTRY.clear()
    _ENTRY_DIRECTION.clear()
    _EXIT_DIRECTION.clear()
    _EXIT_PAIR.clear()
