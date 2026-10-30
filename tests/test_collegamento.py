"""Il collegamento a V4: versione letta, engine solo da V4, avviso se non e' main o ha modifiche."""
import re
from pathlib import Path

import collegamento_v4
from conftest import RADICE


def test_versione_letta(v4):
    assert re.fullmatch(r"[0-9a-f]{7,}", v4["commit"])
    assert v4["ramo"] and v4["data"]
    assert isinstance(v4["modifiche_locali"], int)


def test_engine_viene_solo_da_v4(v4):
    import engine
    assert [Path(p).resolve() for p in engine.__path__] == [Path(v4["cartella"]) / "engine"]
    assert not (RADICE / "engine").exists(), "la vecchia copia engine/ non deve tornare"


def test_avviso():
    base = {"ramo": "main", "modifiche_locali": 0}
    assert collegamento_v4.avviso(base) is None
    testo = collegamento_v4.avviso({**base, "ramo": "mappa-semplice"})
    assert "mappa-semplice" in testo
    assert "3 file modificati" in collegamento_v4.avviso({**base, "modifiche_locali": 3})


def test_stampa_avviso_ben_visibile(capsys):
    collegamento_v4._stampa({"cartella": "x", "ramo": "altro", "commit": "abc1234", "data": "2026-10-30 10:00",
                             "messaggio": "m", "modifiche_locali": 1, "non_tracciati": 0})
    uscita = capsys.readouterr().out
    assert "!!! ATTENZIONE" in uscita and "«altro»" in uscita


def test_funzioni_private_di_v4_esistono(v4):
    """dataset usa vetrina._trigger_candidati e _tutti_i_filtri: se V4 le toglie, qui si vede."""
    from engine import vetrina
    import dataset
    vetrina._registra_condizioni()
    assert len(dataset.trigger_candidati(1)) > 0
    assert len(dataset.trigger_candidati(-1)) > 0
    assert len(dataset.filtri()) > 0
