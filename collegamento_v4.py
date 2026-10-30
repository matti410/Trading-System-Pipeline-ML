"""
Collegamento al Lab (V4), in sola lettura (30/10/2026, docs/02 §A).

  collega(percorso=None, ramo="main") -> dict
      trova V4, lo mette in sys.path, controlla che `engine` venga da li',
      stampa ramo e commit (con un avviso se non e' main o ha modifiche)
  versione() -> dict
      ramo, commit e modifiche locali di V4 (per il registro delle prove)
  cartella_dati(su_colab=None) -> str
      la stessa cartella dei dati del Lab (Drive per desktop o dati/ di V4)

DOVE SI CERCA V4
----------------
1. `percorso`, se lo si passa;
2. la variabile d'ambiente V4_DIR, se c'e';
3. la cartella sorella di questo repo: ../Trading-System-Creator_V4.
Su Colab, se non c'e', si clona il repo pubblico di V4 (senza token) nel
ramo `ramo`. Sul PC `ramo` non cambia niente: V4 resta sul ramo in cui e'
(niente pull, niente checkout), e se non e' main si avvisa.

Nessuna scrittura in V4: solo `git rev-parse`, `git log`, `git status` (senza
riscrivere l'indice), e le cache di Python e numba dirottate in .cache/ di
questo repo.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_V4 = "https://github.com/matti410/Trading-System-Creator_V4.git"
NOME_V4 = "Trading-System-Creator_V4"
RAMO_ATTESO = "main"
DATI_DRIVE_COLAB = "/content/drive/MyDrive/Trading-System-Creator_V4/dati"

_INFO = {}      # l'ultimo collegamento fatto, per cartella_dati


def su_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _git(cartella, *argomenti) -> str:
    # --no-optional-locks: git status non riscrive l'indice di V4 (sola lettura vera)
    esito = subprocess.run(["git", "--no-optional-locks", "-C", str(cartella), *argomenti],
                           capture_output=True, text=True, encoding="utf-8")
    if esito.returncode != 0:
        raise RuntimeError(f"git {' '.join(argomenti)} in {cartella}: {esito.stderr.strip()}")
    return esito.stdout.strip()


def _trova(percorso):
    if percorso is not None:
        return Path(percorso)
    if os.environ.get("V4_DIR"):
        return Path(os.environ["V4_DIR"])
    return Path(__file__).resolve().parent.parent / NOME_V4


def _clona(cartella, ramo):
    esito = subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", ramo, REPO_V4, str(cartella)],
                           capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if esito.returncode != 0:
        raise RuntimeError(f"Non riesco a clonare V4 ({REPO_V4}, ramo {ramo}): {esito.stderr.strip()}")


def leggi_versione(cartella) -> dict:
    """Ramo, commit, data del commit e modifiche locali di V4 (solo lettura)."""
    modificati = [r for r in _git(cartella, "status", "--porcelain", "--untracked-files=no").splitlines() if r]
    non_tracciati = [r for r in _git(cartella, "status", "--porcelain").splitlines() if r.startswith("??")]
    return {
        "cartella": str(Path(cartella).resolve()),
        "ramo": _git(cartella, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git(cartella, "rev-parse", "--short", "HEAD"),
        "data": _git(cartella, "log", "-1", "--format=%ci"),
        "messaggio": _git(cartella, "log", "-1", "--format=%s"),
        "modifiche_locali": len(modificati),
        "non_tracciati": len(non_tracciati),
    }


def avviso(info: dict) -> str | None:
    """Il testo dell'avviso se V4 non e' su main o ha file modificati, altrimenti None."""
    motivi = []
    if info["ramo"] != RAMO_ATTESO:
        motivi.append(f"V4 è sul ramo «{info['ramo']}», non su «{RAMO_ATTESO}».")
    if info["modifiche_locali"]:
        motivi.append(f"V4 ha {info['modifiche_locali']} file modificati e non committati.")
    if not motivi:
        return None
    return " ".join(motivi) + " I risultati dipendono da questa versione di V4, non da quella di main."


def _stampa(info: dict) -> None:
    print(f"V4: {info['cartella']}")
    print(f"    ramo {info['ramo']} · commit {info['commit']} del {info['data'][:16]} · «{info['messaggio']}»")
    print(f"    file modificati non committati: {info['modifiche_locali']}"
          f" · file non tracciati: {info['non_tracciati']}")
    testo = avviso(info)
    if testo:
        riga = "!" * 78
        print(f"\n{riga}\n!!! ATTENZIONE\n!!! {testo}\n{riga}")


def collega(percorso=None, ramo: str = RAMO_ATTESO, verbose: bool = True) -> dict:
    """
    Trova V4, lo mette in testa a sys.path e ne restituisce la versione:
    {cartella, ramo, commit, data, messaggio, modifiche_locali, non_tracciati}.
    Si ferma se la cartella non c'e' (sul PC) o se `engine` non viene da V4.
    """
    cartella = _trova(percorso)
    if not cartella.is_dir():
        if not su_colab():
            raise FileNotFoundError(
                f"Non trovo V4 in {cartella}. Deve stare accanto a questo repo, con il nome {NOME_V4} "
                "(oppure indica la cartella con la variabile V4_DIR).")
        cartella = Path("/content") / NOME_V4
        if not cartella.is_dir():
            _clona(cartella, ramo)
    cartella = cartella.resolve()
    # le cache di Python (.pyc) e di numba vanno in .cache/ di questo repo, mai in V4
    cache = Path(__file__).resolve().parent / ".cache"
    sys.pycache_prefix = str(cache / "pycache")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    if str(cartella) not in sys.path:
        sys.path.insert(0, str(cartella))
    # un engine gia' importato (per esempio la vecchia copia V3) va dimenticato
    for nome in [m for m in sys.modules if m == "engine" or m.startswith("engine.")]:
        del sys.modules[nome]
    import engine
    # engine di V4 non ha __init__.py: e' un namespace package, che unisce tutte le
    # cartelle engine/ in sys.path. Deve essercene una sola, quella di V4.
    origini = [Path(p).resolve() for p in engine.__path__]
    estranee = [str(p) for p in origini if p != cartella / "engine"]
    if estranee or not origini:
        raise ImportError(f"Il modulo engine non viene solo da V4 ({cartella / 'engine'}): "
                          f"c'è un'altra cartella engine/ in {', '.join(estranee) or '?'}.")
    info = leggi_versione(cartella)
    _INFO.clear()
    _INFO.update(info)
    if verbose:
        _stampa(info)
    return info


def versione() -> dict:
    """Ramo, commit e modifiche locali dell'ultimo collegamento (per il registro)."""
    return {k: _INFO.get(k) for k in ("ramo", "commit", "modifiche_locali")}


def cartella_dati(colab: bool | None = None) -> str:
    """
    La cartella dei dati del Lab, con la stessa funzione del Lab
    (vetrina.cartella_dati_avvio): su Colab Drive, sul PC Google Drive per
    desktop se c'e', altrimenti dati/ di V4. Va chiamata dopo collega().
    """
    if not _INFO:
        raise RuntimeError("Chiama prima collega().")
    from engine.vetrina import cartella_dati_avvio
    colab = su_colab() if colab is None else colab
    return cartella_dati_avvio(_INFO["cartella"], colab, DATI_DRIVE_COLAB, False)
