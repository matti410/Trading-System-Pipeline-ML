"""
Preparazione comune dei test: V4 collegato una volta, prima di ogni test.

Senza V4 i test che lo usano si saltano con il motivo. I test sui dati veri
usano BINANCE_PERP_BTCUSDT M15 dalla cartella dei dati del Lab: senza quel
CSV (o senza lo studio) si saltano.
"""
import os
import sys
from pathlib import Path

import pytest

RADICE = Path(__file__).resolve().parent.parent
if str(RADICE) not in sys.path:
    sys.path.insert(0, str(RADICE))

import collegamento_v4  # noqa: E402

try:
    INFO_V4 = collegamento_v4.collega(verbose=False)
    ERRORE_V4 = None
except Exception as e:          # V4 assente: si saltano i test che lo usano
    INFO_V4, ERRORE_V4 = None, f"{type(e).__name__}: {e}"

SYMBOL_VERO, TF_VERO = "BINANCE_PERP_BTCUSDT", "M15"


@pytest.fixture(scope="session")
def v4():
    if INFO_V4 is None:
        pytest.skip(f"V4 non collegato ({ERRORE_V4})")
    return INFO_V4


@pytest.fixture(scope="session")
def cartella_dati(v4):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        cartella = collegamento_v4.cartella_dati()
    csv = Path(cartella) / f"{SYMBOL_VERO}_{TF_VERO}.csv"
    if not csv.exists():
        pytest.skip(f"mancano i dati veri: {csv}")
    if not (Path(cartella) / "studio" / "confine.json").exists():
        pytest.skip(f"manca lo studio di V4 in {cartella}")
    return cartella


@pytest.fixture(scope="session")
def ds_vero(cartella_dati):
    import dataset
    return dataset.apri(SYMBOL_VERO, TF_VERO, cartella_dati, verbose=False)


def pytest_report_header(config):
    if INFO_V4 is None:
        return f"V4: non collegato ({ERRORE_V4})"
    return (f"V4: ramo {INFO_V4['ramo']} · commit {INFO_V4['commit']} · "
            f"modifiche non committate {INFO_V4['modifiche_locali']} · {os.path.basename(INFO_V4['cartella'])}")
