"""
Il dataset di lavoro, preso dal Lab (V4) come la sua schermata 1 (30/10/2026, docs/02 §A).

  apri(symbol, timeframe, cartella, aggiorna=False) -> Dataset
  trigger_candidati(direzione) / filtri() -> list[str]

DA DOVE VIENE OGNI COSA (tutto da V4, niente copie)
---------------------------------------------------
  dati e funding   vetrina.apri_dati: CSV in UTC, taglio al funding, carica_funding;
                   con aggiorna=True prima aggiorna_dati.aggiorna_per_lab (solo sul PC)
  indicatori       indicatori.aggiungi_indicatori
  confine IS/OOS   studio.leggi_confine, IN SOLA LETTURA, poi vetrina.dividi_dati
  costi            vetrina.costi_storico(symbol, df, funding, df_prezzo=df_is)
  trigger, filtri  il registro di V4 (vetrina._registra_condizioni), colonne
                   calcolate come il backtest del Lab: get_entry(nome)(df).astype(bool)

LO STUDIO DI V4 NON SI TOCCA
----------------------------
Il Lab entra nello studio con studio.entra, che puo' scrivere (apre lo studio se
manca, scrive impronta.json per un dataset nuovo). Qui no: si legge confine.json e
lo si passa ad apri_dati come data (modo "data" di V4), che divide senza scrivere.
L'impronta dell'in-sample si confronta con il file, se c'e', senza riscriverlo.
"""
import json
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Dataset:
    symbol: str
    timeframe: str
    cartella: str
    df: pd.DataFrame            # barre con indicatori e colonne E*/F* (bool)
    df_is: pd.DataFrame
    df_oos: pd.DataFrame
    confine: pd.Timestamp       # fine dell'ultima barra in-sample (UTC)
    funding: pd.DataFrame | None
    costi: dict                 # il dizionario dei motori del Lab (spread, commission, pip_size, ...)
    pip: float
    solo_long: bool
    trigger_long: list = field(default_factory=list)
    trigger_short: list = field(default_factory=list)
    filtri: list = field(default_factory=list)
    impronta: str = ""          # "ok" o "assente" (il dataset non e' ancora entrato nello studio)
    aggiornamento: object = None

    def costo_bp(self) -> dict:
        """Commissione andata e ritorno e spread di un passo al prezzo mediano dell'in-sample, in bp."""
        mediana = float(self.df_is["Close"].median())
        commissione = 2 * self.costi["commission"] * 10_000
        spread = self.costi["spread_passo_pips"] * self.pip / mediana * 10_000
        return {"commissione_bp": commissione, "spread_bp": spread, "totale_bp": commissione + spread}

    def riepilogo(self) -> pd.DataFrame:
        def giorno(t):
            return pd.Timestamp(t).strftime("%d/%m/%Y %H:%M")
        c = self.costo_bp()
        righe = [
            ("Symbol", self.symbol),
            ("Timeframe", self.timeframe),
            ("Barre", f"{len(self.df):,}"),
            ("Periodo", f"{giorno(self.df.index[0])} – {giorno(self.df.index[-1])} UTC"),
            ("Confine dello studio", f"{giorno(self.confine)} UTC (letto da V4, non modificato)"),
            ("In-sample", f"{len(self.df_is):,} barre ({len(self.df_is) / len(self.df):.1%})"),
            ("Out-of-sample", f"{len(self.df_oos):,} barre"),
            ("Costo per trade", f"{c['totale_bp']:.2f} bp al prezzo mediano dell'in-sample: commissione "
                                f"{c['commissione_bp']:.1f} bp + spread di un passo {c['spread_bp']:.2f} bp"),
            ("Funding", "sì, contato trade per trade" if self.funding is not None else "no"),
            ("Lati", "solo LONG (spot)" if self.solo_long else "LONG e SHORT"),
            ("Trigger", f"{len(self.trigger_long)} long · {len(self.trigger_short)} short"),
            ("Filtri", f"{len(self.filtri)} (F* e VWAP_*)"),
            ("Impronta dell'in-sample", {"ok": "uguale a quella dello studio",
                                         "assente": "non ancora nello studio: aprilo una volta nel Lab"}
             .get(self.impronta, self.impronta)),
        ]
        return pd.DataFrame(righe, columns=["", " "]).set_index("")


def trigger_candidati(direzione: int) -> list[str]:
    """I trigger del Lab per un lato (V4: vetrina._trigger_candidati, privata)."""
    from engine import vetrina
    return list(vetrina._trigger_candidati(direzione))


def filtri() -> list[str]:
    """I filtri F* del Lab (V4: vetrina._tutti_i_filtri, privata)."""
    from engine import vetrina
    return list(vetrina._tutti_i_filtri())


def leggi_confine(cartella) -> pd.Timestamp:
    """Il confine dello studio di V4, in sola lettura. Si ferma se lo studio non c'e'."""
    from engine import studio
    conf = studio.leggi_confine(cartella)
    if conf is None:
        raise FileNotFoundError(
            f"Non trovo lo studio di V4 in {studio.cartella_studio(cartella)}. Il confine dell'in-sample sta lì: "
            "apri prima lo studio nel Lab (schermata 1). Qui lo studio si legge soltanto.")
    t = pd.Timestamp(conf["confine"])
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def controlla_impronta(cartella, symbol, timeframe, grezzo, confine) -> str:
    """
    Confronta l'impronta dell'in-sample con quella dello studio, senza scrivere.
    "ok" se coincide, "assente" se il dataset non e' ancora entrato nello studio.
    Si ferma se e' diversa (le barre dell'in-sample sono cambiate).
    """
    from engine import studio
    from engine.splitting import split_al_confine
    ds = studio.nome_dataset(symbol, timeframe)
    f = studio.cartella_studio(cartella) / ds / "impronta.json"
    if not f.exists():
        return "assente"
    with open(f, encoding="utf-8") as fh:
        prima = json.load(fh)
    base = split_al_confine(grezzo, confine, studio.passo(timeframe))[0]
    adesso = studio.impronta(base)
    diverso = [k for k in ("barre", "prima", "ultima", "codice") if prima.get(k) != adesso.get(k)]
    if diverso:
        raise ValueError(f"L'in-sample di {ds} non è più quello dello studio (diversi: {', '.join(diverso)}). "
                         "Il CSV è cambiato: controllalo nel Lab prima di continuare.")
    return "ok"


def _stampa_aggiornamento(a) -> None:
    """
    L'esito di aggiorna_per_lab, in breve. vetrina.mostra_dati non si usa: con il
    confine passato come data direbbe «confine di collaudo», qui falso (e' quello
    dello studio).
    """
    if a is None:
        return
    if a == "non qui":
        print("Aggiorna i dati: ignorato (su Colab i dati non si scaricano).")
        return
    for parte in a.values():
        if parte["esito"] == "aggiornato":
            print(f"{parte['nome']}: +{parte['nuove']} nuove, fino al {parte.get('al')}")
        else:
            print(f"!!! {parte['nome']}: aggiornamento non riuscito ({parte['errore'][:160]}). "
                  "Lavoro sui dati che ci sono.")


def _colonne_condizioni(df, nomi_entry, nomi_filtri):
    from engine import registry
    nuove = {n: registry.get_entry(n)(df).astype(bool) for n in nomi_entry}
    nuove.update({n: registry.get_filter(n)(df).astype(bool) for n in nomi_filtri})
    return pd.concat([df, pd.DataFrame(nuove, index=df.index)], axis=1)


def apri(symbol: str, timeframe: str, cartella, aggiorna: bool = False, macchina: str | None = None,
         condizioni: bool = True, verbose: bool = True) -> Dataset:
    """
    Il dataset (symbol, timeframe) come lo prepara la schermata 1 del Lab, con il
    confine dello studio letto e mai scritto. condizioni=False salta il calcolo
    delle colonne E*/F* (serve ai test sui soli costi).
    """
    from engine import vetrina
    from engine.exchange import solo_long
    from engine.indicatori import aggiungi_indicatori
    from engine.simboli import da_exchange

    if not da_exchange(symbol):
        raise ValueError(f"{symbol}: qui si lavora solo con i symbol da exchange di V4 (engine/simboli.py).")
    confine = leggi_confine(cartella)
    d = vetrina.apri_dati(symbol, timeframe, cartella, confine=confine, aggiorna=aggiorna,
                          macchina=macchina, funding=True, verbose=verbose)
    impronta = controlla_impronta(cartella, symbol, timeframe, d.grezzo, confine)
    grezzo = d.df                                   # come vetrina.prepara: i costi sulle barre prima degli indicatori
    with vetrina._zitto():
        df = aggiungi_indicatori(d.df)
    df_is, df_oos = vetrina.dividi_dati(d, df)      # modo "data": split_al_confine, nessuna scrittura
    costi, _ = vetrina.costi_storico(symbol, grezzo, funding=d.funding, df_prezzo=df_is)

    lungo_solo = solo_long(symbol)
    vetrina._registra_condizioni()
    t_long = trigger_candidati(1)
    t_short = [] if lungo_solo else trigger_candidati(-1)
    f = filtri()
    if condizioni:
        df = _colonne_condizioni(df, t_long + t_short, f)
        df_is, df_oos = df.loc[df_is.index], df.loc[df_oos.index]

    if verbose:
        _stampa_aggiornamento(d.aggiornamento)
    return Dataset(symbol=symbol, timeframe=timeframe, cartella=str(cartella), df=df, df_is=df_is,
                   df_oos=df_oos, confine=confine, funding=d.funding, costi=costi,
                   pip=float(costi["pip_size"]), solo_long=lungo_solo, trigger_long=t_long,
                   trigger_short=t_short, filtri=f, impronta=impronta, aggiornamento=d.aggiornamento)
