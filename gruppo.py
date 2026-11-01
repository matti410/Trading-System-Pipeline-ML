"""
Scelta automatica del gruppo di trigger e di H per lato (docs/04, proposta docs/05 approvata da
Mattia il 01/11/2026).

  netti_train(ds, direzione, H, quota=0.8, quarantena=True) -> DataFrame
  ammessi(netti) -> DataFrame
  gruppo_per_H(netti, ammessi, min_per_meta) -> (gruppo, composizione)
  scegli_H(per_H) -> H | None
  proponi(ds, direzione, griglia_h, quota=0.8, min_per_meta=10, quarantena=True) -> Proposta

LE REGOLE (docs/04 §2 e §3, decisioni del 01/11/2026 in testa a docs/05)
------------------------------------------------------------------------
- Tutto sulle sole barre del train (pipeline.barre_train): i prezzi del test non entrano.
- I bp netti sono quelli di target.costruisci_target (ingresso Open i+1, uscita Open i+H+1,
  costi del Lab, quarantena), con TUTTI i trigger del lato insieme: una barra in cui scattano
  piu' trigger e' un solo trade. Restano solo i trade con l'uscita dentro il train.
- Le due meta' del train si dividono per tempo, a meta' delle barre del train.
- Ammesso: media netta > 0 in tutto il train e in ciascuna meta'.
- Gruppo: gli ammessi per media netta decrescente (a parita', piu' occorrenze); un trigger entra
  se le sue occorrenze nuove (barre non gia' coperte) sono almeno min_per_meta in ciascuna meta'
  e hanno media netta > 0 in tutto il train e in ciascuna meta'.
- Totale(H) = somma, sulle barre del gruppo, di netto_bp x peso di unicita' NON normalizzato
  (pesi.pesi_unicita, sulle sole barre del gruppo): media netta x occorrenze effettive.
- H scelto: il Totale piu' alto fra gli H con un gruppo; a parita' l'H piu' corto. Nessun
  gruppo: None (il lato e' proposto vuoto).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pesi import pesi_unicita
from target import costruisci_target

NOMI_LATO = {1: "LONG", -1: "SHORT"}
ENTRA = "entra"


def _barre_train(ds, quota):
    from pipeline import barre_train
    return barre_train(ds, quota)


def _trigger_del_lato(ds, direzione):
    if direzione not in NOMI_LATO:
        raise ValueError("direzione deve essere 1 (LONG) o -1 (SHORT)")
    if direzione == -1 and ds.solo_long:
        raise ValueError("Sullo spot lo short non esiste: niente gruppo SHORT.")
    return list(ds.trigger_long if direzione == 1 else ds.trigger_short)


def netti_train(ds, direzione: int, H: int, quota: float = 0.8, quarantena=True) -> pd.DataFrame:
    """
    I bp netti di ogni barra del train con almeno un trigger del lato (un trade per barra), la
    meta' del train (1 o 2) e una colonna vero/falso per trigger. Indice: la barra del trigger.
    """
    trigger = _trigger_del_lato(ds, direzione)
    tr = _barre_train(ds, quota)
    t = costruisci_target(tr, trigger, direzione, int(H), costi=ds.costi, quarantena=quarantena)
    pos = tr.index.get_indexer(t.index)
    out = pd.DataFrame({"netto_bp": t["netto_bp"].to_numpy(float),
                        "meta": np.where(pos < len(tr) // 2, 1, 2)}, index=t.index)
    vere = tr.loc[t.index, trigger].astype(bool)
    return pd.concat([out, vere], axis=1)


def _trigger(netti):
    return [c for c in netti.columns if c not in ("netto_bp", "meta")]


def _medie(netto, meta):
    """Occorrenze e media netta in tutto il train e in ciascuna meta' (NaN senza occorrenze)."""
    def media(x):
        return float(x.mean()) if len(x) else float("nan")
    uno, due = netto[meta == 1], netto[meta == 2]
    return {"occorrenze": len(netto), "media_netta_bp": media(netto),
            "occorrenze_meta_1": len(uno), "media_meta_1_bp": media(uno),
            "occorrenze_meta_2": len(due), "media_meta_2_bp": media(due)}


def _positive(r):
    return all(r[k] > 0 for k in ("media_netta_bp", "media_meta_1_bp", "media_meta_2_bp"))


def ammessi(netti: pd.DataFrame) -> pd.DataFrame:
    """Per trigger: occorrenze e medie nette (train, meta' 1, meta' 2) e `ammesso` (tutte e tre > 0)."""
    netto, meta = netti["netto_bp"].to_numpy(float), netti["meta"].to_numpy()
    righe = {}
    for nome in _trigger(netti):
        m = netti[nome].to_numpy(bool)
        r = _medie(netto[m], meta[m])
        righe[nome] = {**r, "ammesso": _positive(r)}
    tab = pd.DataFrame.from_dict(righe, orient="index",
                                 columns=["occorrenze", "media_netta_bp", "occorrenze_meta_1", "media_meta_1_bp",
                                          "occorrenze_meta_2", "media_meta_2_bp", "ammesso"])
    tab.index.name = "trigger"
    return tab


def gruppo_per_H(netti: pd.DataFrame, ammessi: pd.DataFrame, min_per_meta: int):
    """
    Il gruppo di un H, un trigger alla volta. Ritorna (gruppo, composizione): per ogni ammesso,
    nell'ordine in cui si prova, le occorrenze nuove (in tutto e per meta'), le loro medie nette
    e l'esito («entra» o il motivo).
    """
    netto, meta = netti["netto_bp"].to_numpy(float), netti["meta"].to_numpy()
    candidati = ammessi[ammessi["ammesso"].astype(bool)]
    ordine = candidati.sort_values(["media_netta_bp", "occorrenze"], ascending=[False, False], kind="stable").index
    coperto = np.zeros(len(netti), dtype=bool)
    gruppo, righe = [], {}
    for nome in ordine:
        nuove = netti[nome].to_numpy(bool) & ~coperto
        r = _medie(netto[nuove], meta[nuove])
        if min(r["occorrenze_meta_1"], r["occorrenze_meta_2"]) < min_per_meta:
            esito = f"nuove sotto {min_per_meta} in una metà"
        elif not _positive(r):
            esito = "media delle nuove ≤ 0 nel train o in una metà"
        else:
            esito = ENTRA
            gruppo.append(nome)
            coperto |= nuove
        righe[nome] = {"nuove": r["occorrenze"], "media_nuove_bp": r["media_netta_bp"],
                       "nuove_meta_1": r["occorrenze_meta_1"], "media_meta_1_bp": r["media_meta_1_bp"],
                       "nuove_meta_2": r["occorrenze_meta_2"], "media_meta_2_bp": r["media_meta_2_bp"],
                       "esito": esito}
    composizione = pd.DataFrame.from_dict(righe, orient="index",
                                          columns=["nuove", "media_nuove_bp", "nuove_meta_1", "media_meta_1_bp",
                                                   "nuove_meta_2", "media_meta_2_bp", "esito"])
    composizione.index.name = "trigger"
    return gruppo, composizione


def scegli_H(per_H: pd.DataFrame):
    """L'H con il Totale piu' alto fra quelli con un gruppo; a parita' il piu' corto. Nessun gruppo: None."""
    con_gruppo = per_H[per_H["trigger_nel_gruppo"] > 0]
    if con_gruppo.empty:
        return None
    migliore = con_gruppo["Totale_bp"].max()
    return int(min(h for h, t in con_gruppo["Totale_bp"].items() if t == migliore))


@dataclass
class Proposta:
    direzione: int
    griglia: list
    min_per_meta: int
    quota: float
    per_H: pd.DataFrame             # una riga per H: gruppo, occorrenze, effettive, media netta, Totale
    H: int | None                   # l'H scelto (None: nessun gruppo, lato proposto vuoto)
    gruppo: list                    # il gruppo dell'H scelto
    ammessi: dict = field(default_factory=dict)        # H -> tabella di ammessi()
    composizione: dict = field(default_factory=dict)   # H -> composizione di gruppo_per_H()

    @property
    def nome(self):
        return NOMI_LATO[self.direzione]


def _controlla_griglia(griglia_h):
    griglia = list(griglia_h or [])
    if not griglia:
        raise ValueError("GRIGLIA_H vuota: scrivi almeno un H in barre.")
    for h in griglia:
        if isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"Gli H di GRIGLIA_H devono essere numeri interi di barre >= 1, non {h!r}.")
    if len(set(griglia)) != len(griglia):
        raise ValueError("GRIGLIA_H ha H ripetuti.")
    return sorted(int(h) for h in griglia)


def proponi(ds, direzione: int, griglia_h, quota: float = 0.8, min_per_meta: int = 10,
            quarantena=True) -> Proposta:
    """La proposta di un lato: per ogni H della griglia ammessi, gruppo e Totale, poi l'H scelto."""
    griglia = _controlla_griglia(griglia_h)
    if isinstance(min_per_meta, bool) or not isinstance(min_per_meta, (int, np.integer)) or min_per_meta < 0:
        raise ValueError(f"MIN_NUOVE_PER_META deve essere un intero >= 0, non {min_per_meta!r}.")
    tr = _barre_train(ds, quota)
    righe, tutti_ammessi, composizioni, gruppi = {}, {}, {}, {}
    for H in griglia:
        netti = netti_train(ds, direzione, H, quota=quota, quarantena=quarantena)
        tutti_ammessi[H] = ammessi(netti)
        gruppi[H], composizioni[H] = gruppo_per_H(netti, tutti_ammessi[H], int(min_per_meta))
        barre = netti.index[netti[gruppi[H]].any(axis=1)] if gruppi[H] else netti.index[:0]
        netto = netti.loc[barre, "netto_bp"]
        peso = pesi_unicita(barre, tr, H, normalizza=False) if len(barre) else netto * 0.0
        righe[H] = {"trigger_nel_gruppo": len(gruppi[H]), "gruppo": ", ".join(gruppi[H]),
                    "occorrenze": len(barre), "effettive": float(peso.sum()),
                    "media_netta_bp": float(netto.mean()) if len(barre) else float("nan"),
                    "Totale_bp": float((netto * peso).sum())}
    per_H = pd.DataFrame.from_dict(righe, orient="index")
    per_H.index.name = "H"
    H = scegli_H(per_H)
    return Proposta(direzione=direzione, griglia=griglia, min_per_meta=int(min_per_meta), quota=quota,
                    per_H=per_H, H=H, gruppo=list(gruppi[H]) if H is not None else [],
                    ammessi=tutti_ammessi, composizione=composizioni)
