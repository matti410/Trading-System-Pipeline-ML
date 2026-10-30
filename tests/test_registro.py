"""Il registro delle prove: solo in aggiunta, un file per macchina, mai dentro studio/."""
import json
from pathlib import Path

import pytest

import registro


def _studio_finto(cartella):
    s = Path(cartella) / "studio"
    (s / "BINANCE_PERP_BTCUSDT_M15").mkdir(parents=True)
    (s / "confine.json").write_text('{"confine": "2025-06-15T17:30:00+00:00"}', encoding="utf-8")
    (s / "BINANCE_PERP_BTCUSDT_M15" / "giornale_pc.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    return sorted((str(p), p.read_bytes(), p.stat().st_mtime_ns) for p in s.rglob("*") if p.is_file())


def test_solo_in_aggiunta_e_studio_intatto(tmp_path):
    prima_studio = _studio_finto(tmp_path)
    ds = "binance_perp_btcusdt_m15"
    registro.aggiungi(tmp_path, ds, {"tipo": "addestramento", "lato": "LONG", "H": 16}, macchina="pc")
    f = registro.file_giornale(tmp_path, ds, "pc")
    dopo_uno = f.read_bytes()
    registro.aggiungi(tmp_path, ds, {"tipo": "oos", "cosa": "tabella"}, macchina="pc")
    dopo_due = f.read_bytes()
    assert dopo_due.startswith(dopo_uno) and len(dopo_due) > len(dopo_uno)   # la riga di prima non cambia
    righe = [json.loads(r) for r in dopo_due.decode("utf-8").splitlines()]
    assert [r["tipo"] for r in righe] == ["addestramento", "oos"]
    assert righe[0]["dataset"] == "BINANCE_PERP_BTCUSDT_M15" and righe[0]["macchina"] == "pc"
    assert "studio" not in f.relative_to(tmp_path).parts
    assert f.relative_to(tmp_path).parts[0] == "registro_ml"
    assert sorted(
        (str(p), p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "studio").rglob("*") if p.is_file()
    ) == prima_studio


def test_un_file_per_macchina_e_conto(tmp_path):
    ds = "BYBIT_SPOT_ETHUSDT_H1"
    registro.aggiungi(tmp_path, ds, {"tipo": "addestramento", "collaudo": True}, macchina="pc")
    registro.aggiungi(tmp_path, ds, {"tipo": "oos", "collaudo": False}, macchina="colab")
    registro.aggiungi(tmp_path, ds, {"tipo": "oos", "collaudo": True}, macchina="colab")
    assert registro.file_giornale(tmp_path, ds, "pc").exists()
    assert registro.file_giornale(tmp_path, ds, "colab").exists()
    assert registro.conta(tmp_path, ds) == {"addestramento": 1, "addestramento_collaudo": 1,
                                            "oos": 2, "oos_collaudo": 1, "selezione": 0,
                                            "selezione_collaudo": 0, "confronto": 0, "confronto_collaudo": 0}
    assert len(registro.leggi(tmp_path, ds)) == 3


def test_voci_non_valide(tmp_path):
    with pytest.raises(ValueError):
        registro.aggiungi(tmp_path, "X_M15", {"tipo": "altro"}, macchina="pc")
    with pytest.raises(ValueError):
        registro.aggiungi(tmp_path, "X_M15", {"tipo": "oos"}, macchina="../studio")


def test_valori_numpy_e_pandas(tmp_path):
    import numpy as np
    import pandas as pd
    riga = registro.aggiungi(tmp_path, "X_M15", {"tipo": "oos", "a": np.float64(1.5), "b": np.int64(2),
                                                 "c": pd.Timestamp("2025-01-01", tz="UTC"), "d": float("nan")},
                             macchina="pc")
    assert riga["a"] == 1.5 and riga["b"] == 2 and riga["c"].startswith("2025-01-01") and riga["d"] is None


def test_versione_ml():
    v = registro.versione_ml()
    assert v["commit"] and isinstance(v["modifiche_locali"], bool)
