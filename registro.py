"""
Registro delle prove del ramo ML (D4, decisione di Mattia del 30/10/2026).

  aggiungi(cartella_dati, dataset, voce, macchina=None) -> dict
  leggi(cartella_dati, dataset) -> list[dict]
  conta(cartella_dati, dataset) -> dict
  versione_ml() -> dict

DOVE
----
    <DATI_DIR>/registro_ml/<SYMBOL>_<TF>/giornale_<macchina>.jsonl

Accanto a studio/ di V4 ma separato: lo studio non si tocca. Un file per
dataset e per macchina (pc, colab), come i giornali di V4: con Google Drive
per desktop PC e Colab non scrivono mai lo stesso file.

SOLO IN AGGIUNTA
----------------
Ogni voce e' una riga JSON aggiunta in fondo al file (apertura in "a", come
studio.aggiungi_voce di V4). Una riga scritta non si cambia e non si cancella.

COSA SI REGISTRA (lavoro B)
---------------------------
  tipo "addestramento"  ogni modello addestrato: lato, H, trigger, colonne della X,
                        pesi, parametri, semi, metriche train/test
  tipo "oos"            ogni volta che si guarda l'OOS (tabella OOS o backtest)
Ogni voce porta data (UTC, dal sistema), macchina, commit di V4 e di questo
repo, e collaudo = True/False (valori di collaudo, non scelte).
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

CARTELLA = "registro_ml"
TIPI = ("addestramento", "oos")


def cartella_registro(cartella_dati) -> Path:
    return Path(cartella_dati) / CARTELLA


def file_giornale(cartella_dati, dataset: str, macchina: str) -> Path:
    if not macchina or not all(c.isalnum() or c in "-_" for c in macchina):
        raise ValueError(f"nome della macchina non valido: {macchina!r} (es. pc, colab)")
    return cartella_registro(cartella_dati) / str(dataset).upper() / f"giornale_{macchina}.jsonl"


def _normale(x):
    """Rende serializzabile in JSON quello che arriva da numpy e pandas."""
    if isinstance(x, dict):
        return {str(k): _normale(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, pd.Index)):
        return [_normale(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            return x.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(x, float) and x != x:
        return None
    return x


def versione_ml() -> dict:
    """Commit di questo repo e se ci sono modifiche non committate."""
    radice = Path(__file__).resolve().parent

    def git(*a):
        esito = subprocess.run(["git", "--no-optional-locks", "-C", str(radice), *a],
                               capture_output=True, text=True, encoding="utf-8")
        return esito.stdout.strip() if esito.returncode == 0 else ""
    return {"commit": git("rev-parse", "--short", "HEAD") or None,
            "modifiche_locali": bool(git("status", "--porcelain", "--untracked-files=no"))}


def macchina_predefinita() -> str:
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "pc"


def aggiungi(cartella_dati, dataset: str, voce: dict, macchina: str | None = None, quando=None) -> dict:
    """Aggiunge una riga al giornale di questa macchina. Ritorna la riga scritta."""
    if voce.get("tipo") not in TIPI:
        raise ValueError(f"tipo della voce: uno di {TIPI}")
    macchina = macchina or macchina_predefinita()
    f = file_giornale(cartella_dati, dataset, macchina)
    if "studio" in [p.lower() for p in f.relative_to(Path(cartella_dati)).parts]:
        raise ValueError("il registro non deve stare dentro studio/")
    quando = pd.Timestamp.now(tz="UTC") if quando is None else pd.Timestamp(quando)
    riga = {"quando": quando.isoformat(), "macchina": macchina, "dataset": str(dataset).upper(),
            **_normale(voce)}
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(riga, ensure_ascii=False) + "\n")
    return riga


def leggi(cartella_dati, dataset: str) -> list[dict]:
    """Tutte le voci del dataset, da tutte le macchine, in ordine di tempo."""
    cartella = cartella_registro(cartella_dati) / str(dataset).upper()
    voci = []
    for f in sorted(cartella.glob("giornale_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            voci += [json.loads(r) for r in fh if r.strip()]
    return sorted(voci, key=lambda v: v.get("quando", ""))


def conta(cartella_dati, dataset: str) -> dict:
    """Quanti addestramenti e quante occhiate all'OOS sono gia' registrati (di cui di collaudo)."""
    voci = leggi(cartella_dati, dataset)
    out = {}
    for tipo in TIPI:
        di_tipo = [v for v in voci if v.get("tipo") == tipo]
        out[tipo] = len(di_tipo)
        out[f"{tipo}_collaudo"] = sum(bool(v.get("collaudo")) for v in di_tipo)
    return out


def stampa_conto(cartella_dati, dataset: str) -> None:
    c = conta(cartella_dati, dataset)
    print(f"Registro {str(dataset).upper()} ({cartella_registro(cartella_dati)}): "
          f"{c['addestramento']} modelli addestrati (di collaudo {c['addestramento_collaudo']}) · "
          f"OOS guardato {c['oos']} volte (di collaudo {c['oos_collaudo']})")
