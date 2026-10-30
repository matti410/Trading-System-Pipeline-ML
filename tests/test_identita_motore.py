"""
Target = motore veloce di V4 = Lab, senza tolleranze (decisione D5 del 30/10/2026).

Per ogni trade che il motore veloce apre con uscita a tempo n_barre = H, il
target dello stesso trigger deve avere lo stesso prezzo d'ingresso, lo stesso
prezzo d'uscita e gli stessi bp netti, uguali al bit. I costi del motore si
aggiungono con la stessa catena del Lab (engine/exit_search_bt.py).
"""
import numpy as np
import pandas as pd
import pytest

from target import costruisci_target


def _trade_del_motore(df, segnale, direction, H, costi, quarantena):
    from engine.exchange import aggiungi_funding, aggiungi_spread_passo
    from engine.metriche import netti_in_unita
    from engine.motore_veloce import MotoreVeloce
    from engine.quarantena_trade import costo_extra_pips, maschera_segnali_puliti
    df_bt = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df_bt["sig"] = segnale.to_numpy() & maschera_segnali_puliti(df, quarantena)
    for col in ("sl_long", "tp_long", "sl_short", "tp_short"):
        df_bt[col] = np.nan
    lato = {"long_col": "sig"} if direction == 1 else {"short_col": "sig"}
    stats = MotoreVeloce(df_bt, cash=costi.get("cash", 10_000_000), spread=costi["spread"],
                         commission=costi["commission"], exclusive_orders=True).run(n_barre=H, **lato)
    tr = costo_extra_pips(stats["_trades"], df, quarantena, costi["spread"], costi.get("spread_rollover_pips"),
                          costi["pip_size"])
    tr = aggiungi_funding(tr, df, costi.get("funding"), costi["pip_size"])
    tr = aggiungi_spread_passo(tr, costi.get("spread_passo_pips"))
    _, netti = netti_in_unita(tr, costi["pip_size"], costi["commission"], "bp")
    return tr, netti


def _controlla_identita(df, colonna, direction, H, costi, quarantena):
    t = costruisci_target(df, [colonna], direction, H, costi=costi, quarantena=quarantena)
    tr, netti = _trade_del_motore(df, df[colonna], direction, H, costi, quarantena)
    assert len(tr) > 20, "troppi pochi trade per un test utile"
    assert set((tr["ExitBar"] - tr["EntryBar"]).tolist()) == {H}       # stessa durata: H barre
    segnale = df.index[tr["EntryBar"].to_numpy() - 1]                   # la barra del trigger
    dal_target = t.loc[segnale]
    assert np.array_equal(dal_target["ingresso"].to_numpy(), tr["EntryPrice"].to_numpy())
    assert np.array_equal(dal_target["uscita"].to_numpy(), tr["ExitPrice"].to_numpy())
    assert np.array_equal(dal_target["netto_bp"].to_numpy(), netti.to_numpy())
    return len(tr)


def _mercato_inventato(n=4000, seme=0):
    rng = np.random.default_rng(seme)
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    close = 20_000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    apertura = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.0003, n))
    alto = np.maximum(apertura, close) * (1 + rng.uniform(0, 0.001, n))
    basso = np.minimum(apertura, close) * (1 - rng.uniform(0, 0.001, n))
    df = pd.DataFrame({"Open": apertura.round(1), "High": alto.round(1), "Low": basso.round(1),
                       "Close": close.round(1), "Volume": 1.0}, index=idx)
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    df["E_L"] = rng.random(n) < 0.04
    df["E_S"] = rng.random(n) < 0.04
    funding = pd.DataFrame({"tasso": rng.normal(0.0001, 0.0003, n // 32), "intervallo_ore": 8.0},
                           index=idx[::32][: n // 32])
    return df, funding


@pytest.mark.parametrize("direction,colonna", [(1, "E_L"), (-1, "E_S")])
def test_identita_su_barre_inventate(v4, direction, colonna):
    df, funding = _mercato_inventato()
    costi = {"spread": 0.0, "spread_passo_pips": 0.1, "commission": 0.0005, "pip_size": 1.0,
             "spread_rollover_pips": None, "funding": funding}
    _controlla_identita(df, colonna, direction, 16, costi, quarantena=False)


def test_identita_sui_dati_veri(ds_vero):
    """BTC perpetual M15 di V4, con quarantena e funding veri, primo trigger long e primo short."""
    df = ds_vero.df.loc[: ds_vero.df_is.index[-1]]
    _controlla_identita(df, ds_vero.trigger_long[0], 1, 16, ds_vero.costi, quarantena=True)
    _controlla_identita(df, ds_vero.trigger_short[0], -1, 32, ds_vero.costi, quarantena=True)
