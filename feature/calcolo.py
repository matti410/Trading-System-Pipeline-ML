"""
Calcolo e collaudi del catalogo delle feature (lavoro C, 30/10/2026, docs/02 §C).

  calcola_feature(df, timeframe, trigger_lato=None, df_btc=None) -> DataFrame
  barre_finestra(ore, timeframe) -> int
  riscaldamento(feat, riscaldamento_v4=0) -> (Series, int)
  degenerazione_continue(feat, indice=None) -> DataFrame
  collaudo_lookahead(df, timeframe, trigger_lato=None, df_btc=None, ...) -> DataFrame
  elenco_catalogo(timeframe, feat=None, lookahead=None, degenerazione=None, riscaldamento_v4=0) -> DataFrame
  collaudo_catalogo(ds, df_btc=None, trigger_lato=None) -> (elenco, feat)

QUI NON SI TAGLIA NIENTE
------------------------
calcola_feature dice quante barre iniziali sono NaN per colonna e il riscaldamento
totale (quello delle feature + le barre che gli indicatori di V4 tolgono all'inizio),
ma non toglie righe e non tocca il target: la regola sui NaN si decide nel lavoro D.

DEGENERAZIONE DELLE COLONNE CONTINUE (docs/02 §C, test 2)
--------------------------------------------------------
verifica_degenerazione di V4 guarda solo le condizioni 0/1 del registro. Per le
feature continue lo stesso controllo, con le tre idee di V4 e le soglie del §C:
  MAI_CALCOLATA    tutta NaN dopo il riscaldamento;
  QUASI_COSTANTE   un solo valore in >= 95% delle barre;
  RARA             colonne di conteggio con meno di 30 valori diversi da zero.
Proposta per V4: estendere verifica_degenerazione alle colonne continue.
"""
import numpy as np
import pandas as pd

from feature.catalogo import CATALOGO, MIN_BARRE, MIN_PASSO_BARRE, Contesto

QUOTA_COSTANTE = 0.95
MIN_NON_NULLI = 30
COLONNE_GREZZE = ["Open", "High", "Low", "Close", "Volume"]


def barre_finestra(ore, timeframe: str) -> int:
    """Barre di `timeframe` in `ore` ore (stesso conto di indicatori.barre_per_durata di V4, senza minimo)."""
    from engine.studio import passo
    return int(round(float(ore) * 60.0 / (passo(timeframe) / pd.Timedelta(minutes=1))))


def _motivo_non_calcolata(voce, ctx):
    if voce["richiede"] == "btc" and ctx.btc is None:
        return "solo altcoin: serve BTC dello stesso exchange e mercato"
    if voce["richiede"] == "trigger" and ctx.trigger is None:
        return "servono i trigger del lato (passo 4)"
    corte = [f"{o} h = {ctx.n(o)} barre" for o in voce["finestre_ore"] if ctx.n(o) < MIN_BARRE]
    if corte:
        return f"finestra sotto {MIN_BARRE} barre ({', '.join(corte)})"
    if voce.get("passo_ore") is not None and ctx.n(voce["passo_ore"]) < MIN_PASSO_BARRE:
        return (f"passo sotto {MIN_PASSO_BARRE} barre ({voce['passo_ore']} h = "
                f"{ctx.n(voce['passo_ore'])} barre)")
    return None


def calcola_feature(df, timeframe, trigger_lato=None, df_btc=None, nomi=None, verbose=True) -> pd.DataFrame:
    """
    Le feature del catalogo su `df` (servono Open, High, Low, Close, Volume; indice UTC).

    trigger_lato  Series bool dei trigger del lato (famiglia 11); None = famiglia 11 non calcolata
    df_btc        le barre di BTC dello stesso exchange, mercato e timeframe (famiglia 10);
                  None = famiglia 10 non calcolata (per BTC stesso, o se manca)
    nomi          solo queste feature (default: tutto il catalogo)

    Ritorna un DataFrame con una colonna per feature calcolata, stesso indice di `df`.
    In .attrs["non_calcolate"] il motivo di ogni feature non calcolata. Le feature con
    una finestra sotto MIN_BARRE barre si stampano sempre.
    """
    ctx = Contesto(df, timeframe, trigger_lato=trigger_lato, df_btc=df_btc)
    colonne, non_calcolate = {}, {}
    for nome in (nomi or CATALOGO):
        voce = CATALOGO[nome]
        motivo = _motivo_non_calcolata(voce, ctx)
        if motivo:
            non_calcolate[nome] = motivo
            continue
        colonne[nome] = voce["funzione"](ctx).astype(float).rename(nome)
    feat = pd.DataFrame(colonne, index=df.index)
    feat.attrs["non_calcolate"] = non_calcolate
    corte = {n: m for n, m in non_calcolate.items() if m.startswith(("finestra", "passo"))}
    if corte:
        print(f"{timeframe}: {len(corte)} feature non calcolate per finestra o passo troppo corti: "
              + "; ".join(f"{n} ({m})" for n, m in corte.items()))
    if verbose:
        print(f"{timeframe}: {feat.shape[1]} feature calcolate su {len(CATALOGO) if nomi is None else len(nomi)} "
              f"del catalogo ({len(non_calcolate)} non calcolate)")
    return feat


def riscaldamento(feat: pd.DataFrame, riscaldamento_v4: int = 0):
    """
    (barre iniziali NaN per colonna, riscaldamento totale in barre). Il totale e' le
    barre che gli indicatori di V4 tolgono all'inizio + la colonna con piu' NaN iniziali.
    """
    nan_iniziali = {}
    for c in feat.columns:
        validi = np.flatnonzero(feat[c].notna().to_numpy())
        nan_iniziali[c] = int(validi[0]) if len(validi) else len(feat)
    s = pd.Series(nan_iniziali, name="barre_nan_iniziali", dtype=int)
    return s, int(riscaldamento_v4) + (int(s.max()) if len(s) else 0)


def degenerazione_continue(feat: pd.DataFrame, indice=None) -> pd.DataFrame:
    """Esito di degenerazione per colonna, sulle barre dopo il riscaldamento della colonna (e in `indice`)."""
    nan_iniziali, _ = riscaldamento(feat)
    righe = []
    for c in feat.columns:
        s = feat[c].iloc[nan_iniziali[c]:]
        if indice is not None:
            s = s.loc[s.index.intersection(indice)]
        validi = s.dropna()
        if len(validi) == 0:
            righe.append((c, "MAI_CALCOLATA", "tutta NaN dopo il riscaldamento"))
            continue
        quota = float(validi.value_counts(normalize=True).iloc[0])
        if CATALOGO.get(c, {}).get("conteggio") and int((validi != 0).sum()) < MIN_NON_NULLI:
            righe.append((c, "RARA", f"{int((validi != 0).sum())} valori diversi da zero (< {MIN_NON_NULLI})"))
        elif quota >= QUOTA_COSTANTE:
            righe.append((c, "QUASI_COSTANTE", f"un valore nel {quota:.1%} delle barre (>= {QUOTA_COSTANTE:.0%})"))
        else:
            righe.append((c, "OK", f"valore piu' frequente nel {quota:.1%} delle barre"))
    return pd.DataFrame(righe, columns=["nome", "degenerazione", "dettaglio_degenerazione"]).set_index("nome")


def collaudo_lookahead(df, timeframe, trigger_lato=None, df_btc=None, riscaldamento_tagli=None, seme=0,
                       verbose=False) -> pd.DataFrame:
    """
    verifica_lookahead di V4 sulle colonne del catalogo: le feature ricalcolate su
    ogni storico tagliato (uno per orario del giorno, tagli_standard di V4) devono
    essere uguali a quelle dello storico intero, barra per barra. BTC e trigger
    passano interi: se una feature usasse BTC o trigger futuri, il taglio lo mostra.
    Ritorna una riga per feature: esito_lookahead (OK / LOOKAHEAD), tagli, barre_diverse.
    """
    from engine.collaudo_catalogo import tagli_standard, verifica_lookahead
    grezzo = df[COLONNE_GREZZE]

    def prepara(d):
        return calcola_feature(d, timeframe, trigger_lato=trigger_lato, df_btc=df_btc, verbose=False)

    if riscaldamento_tagli is None:
        _, riscaldamento_tagli = riscaldamento(prepara(grezzo))
    tagli = tagli_standard(grezzo.index, seme=seme, riscaldamento=riscaldamento_tagli)
    import contextlib
    import io
    uscita = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with uscita:
        rep = verifica_lookahead(grezzo, prepara=prepara, tagli=tagli, tagli_per_condizione=0,
                                 riscaldamento=riscaldamento_tagli, verbose=verbose)
    rep = rep[rep["tipo"] == "indicatore"].rename(columns={"esito": "lookahead"})
    return rep.set_index("nome")[["lookahead", "tagli", "barre_diverse", "primo_caso"]]


def elenco_catalogo(timeframe, feat=None, lookahead=None, degenerazione=None, riscaldamento_v4=0) -> pd.DataFrame:
    """
    Una riga per feature del catalogo: famiglia, finestre in ore e in barre, formula e,
    se ci sono, calcolata sì/no, barre NaN iniziali e l'esito dei collaudi. «esito» riassume:
    non calcolata / LOOKAHEAD / DEGENERE (con il tipo) / OK.
    """
    non_calcolate = {} if feat is None else feat.attrs.get("non_calcolate", {})
    nan_iniziali = riscaldamento(feat, riscaldamento_v4)[0] if feat is not None else pd.Series(dtype=int)
    righe = []
    for nome, v in CATALOGO.items():
        r = {"nome": nome, "famiglia": v["famiglia"],
             "finestre_ore": ", ".join(str(o) for o in v["finestre_ore"]) or "-",
             "finestre_barre": ", ".join(str(barre_finestra(o, timeframe)) for o in v["finestre_ore"]) or "-",
             "formula": v["formula"]}
        if feat is not None:
            calcolata = nome in feat.columns
            r["calcolata"] = "sì" if calcolata else f"no: {non_calcolate.get(nome, 'non richiesta')}"
            r["barre_nan_iniziali"] = int(nan_iniziali[nome]) if calcolata else np.nan
            esito_l = lookahead["lookahead"].get(nome) if lookahead is not None and calcolata else None
            esito_d = degenerazione["degenerazione"].get(nome) if degenerazione is not None and calcolata else None
            r["lookahead"] = esito_l or ("-" if calcolata else "")
            r["degenerazione"] = esito_d or ("-" if calcolata else "")
            if not calcolata:
                r["esito"] = "non calcolata"
            elif esito_l == "LOOKAHEAD":
                r["esito"] = "LOOKAHEAD"
            elif esito_d not in (None, "OK"):
                r["esito"] = f"DEGENERE ({esito_d})"
            elif esito_l == "OK" and esito_d == "OK":
                r["esito"] = "OK"
            else:
                r["esito"] = "collaudo non fatto"
        righe.append(r)
    return pd.DataFrame(righe).set_index("nome")


def collaudo_catalogo(ds, df_btc=None, trigger_lato=None, verbose=True):
    """
    Il collaudo del catalogo su un dataset (cella dopo il passo 2 del notebook):
      - feature sullo storico intero;
      - degenerazione sulle sole barre dell'in-sample (l'OOS non entra);
      - lookahead su un pezzo dell'inizio dell'in-sample: il riscaldamento delle
        feature + altre 720 ore, con i tagli standard di V4.
    Ritorna (elenco, feat). Stampa il riscaldamento.
    """
    feat = calcola_feature(ds.df, ds.timeframe, trigger_lato=trigger_lato, df_btc=df_btc, verbose=verbose)
    nan_iniziali, totale = riscaldamento(feat, getattr(ds, "riscaldamento_v4", 0))
    deg = degenerazione_continue(feat, indice=ds.df_is.index)
    _, risc_feat = riscaldamento(feat)
    pezzo = ds.df.iloc[: risc_feat + barre_finestra(720, ds.timeframe)]
    if pezzo.index[-1] > ds.df_is.index[-1]:
        raise ValueError("il pezzo per il lookahead uscirebbe dall'in-sample")
    look = collaudo_lookahead(pezzo, ds.timeframe, trigger_lato=trigger_lato, df_btc=df_btc,
                              riscaldamento_tagli=risc_feat)
    if verbose:
        print(f"riscaldamento: feature {risc_feat} barre (la colonna con più NaN iniziali: "
              f"{nan_iniziali.idxmax()}) + indicatori di V4 {getattr(ds, 'riscaldamento_v4', 0)} barre "
              f"= {totale} barre. Qui non si taglia niente: la regola sui NaN è del lavoro D.")
        print(f"lookahead su {len(pezzo):,} barre ({pezzo.index[0]:%d/%m/%Y} -> {pezzo.index[-1]:%d/%m/%Y}), "
              f"{int(look['tagli'].max()) if len(look) else 0} tagli")
    elenco = elenco_catalogo(ds.timeframe, feat, lookahead=look, degenerazione=deg,
                             riscaldamento_v4=getattr(ds, "riscaldamento_v4", 0))
    return elenco, feat
