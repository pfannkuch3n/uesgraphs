"""Einzelwiderstände (ζ) aus gezählten Formteilen statt aus einer Längenformel.

Bisher kam der Summenbeiwert eines Rohres aus
:meth:`SystemModelHeating.estimate_xi` — einer reinen Längenheuristik
("4 Bögen je 25 m"). Die Annahme steht auf dem Kopf: wie viele Bögen und
T-Stücke ein Strang hat, hängt an der Verlegung, nicht an der Länge.

Dieses Modul dreht die Richtung um:

1. :func:`derive_from_geometry` zählt je **Knoten** die Formteile, die dort
   geometrisch stehen müssen (Knotengrad, Ablenkwinkel, DN-Wechsel).
2. :func:`node_zeta_to_edges` verteilt die Knotenverluste auf die **Kanten** —
   weder pandapipes noch Modelica kennen Knotenverluste. Die Verteilung ist so
   gewählt, dass jeder Weg *durch* den Knoten genau den richtigen Verlust
   trägt.
3. :func:`apply_fittings` schreibt das Ergebnis als Kantenattribute.

Konventionen
------------
* Zählungen und ζ gelten **je Strang**, nicht je Graben. ``hydraulics.py``
  legt ζ identisch auf Vor- und Rücklauf; damit ist der symmetrische Graben
  korrekt abgebildet.
* Kantenattribute: ``sum_zetas`` (float), ``zeta_quelle``
  ("auto" | "plan" | "geschaetzt" | "default") und ``zeta_herkunft``
  (Liste von Dicts, JSON-serialisierbar, für den Hover in der Analyse-App).
  In ``zeta_herkunft`` ist ``zeta`` der Anteil **je Stück**, es gilt also
  ``sum_zetas == Σ anzahl * zeta`` über alle Einträge.
* Knotenattribute: ``formteile`` (Zählungen), ``formteile_quelle`` und bei
  Grad-2-Knoten ``bogen_winkel_deg``.
* Bestehende Attribute werden nie überschrieben, solange ``overwrite=False``
  — gleiches Muster wie :mod:`uesgraphs.heatnetsim.pipe_specs`.

Quellen der ζ-Werte stehen je Wert im :data:`KATALOG` und sind über
``uesgraphs/data/fittings_katalog.toml`` überschreibbar.
"""

import hashlib
import json
import logging
import math
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from uesgraphs.utilities import set_up_file_logger


#: Ab diesem Ablenkwinkel gilt ein Grad-2-Knoten als echter Bogen. Darunter ist
#: er ein Vermessungspunkt der Trasse und **kein** Formteil. Das ist der
#: wichtigste Stellhebel des Moduls.
BOGEN_WINKEL_SCHWELLE_DEG = 5.0

#: Annahme für den Hausanschluss (Grad-1-Gebäudeknoten): eine Absperrung je
#: Strang. Unbelegt, aber sparsamer als die alte Heuristik (2 Schieber je Rohr).
SCHIEBER_JE_HAUSANSCHLUSS = 1

#: ζ je Formteil mit Quelle je Wert. Startwerte aus ``estimate_xi``
#: (``systemmodels_pp/systemmodelheating.py``), dort belegt über Horlacher2016
#: und Böhmer. Das T-Stück ist aufgeteilt — die alte Sammelkonstante 0,25 kennt
#: den Unterschied zwischen Durchgang und Abzweig nicht.
KATALOG = {
    "bogen": {
        "zeta": 0.08,
        "quelle": "Horlacher2016, Rohrleitungen 2, S.519-521 (90°-Bogen, "
                  "grosser Kruemmungsradius); Startwert aus estimate_xi",
    },
    "t_durchgang": {
        "zeta": 0.30,
        "quelle": "Horlacher2016, Rohrleitungen 2, S.519-521 (Trennung, "
                  "Durchgang); alte Sammelkonstante 0,25 liegt dazwischen",
    },
    "t_abzweig": {
        "zeta": 1.00,
        "quelle": "Horlacher2016, Rohrleitungen 2, S.519-521 (Trennung, "
                  "Abzweig 90°); alte Sammelkonstante 0,25 liegt dazwischen",
    },
    "schieber": {
        "zeta": 0.05,
        "quelle": "Boehmer, Fernwaerme, S.46 (Flachschieber offen); "
                  "Startwert aus estimate_xi",
    },
    # Der Querschnittssprung wird nicht aus dem Katalog gelesen, sondern je
    # Knoten aus dem Flaechenverhaeltnis gerechnet (siehe zeta_querschnitt).
    # Der Eintrag steht hier nur, damit die Quelle dokumentiert ist.
    "querschnittssprung": {
        "zeta": None,
        "quelle": "Borda-Carnot (Erweiterung) bzw. Idelchik (Verengung), "
                  "je Knoten aus dem Flaechenverhaeltnis gerechnet",
    },
    # Kompensatoren liegen in der Trasse und sind aus der Geometrie NICHT
    # ableitbar — der Eintrag existiert nur für die händische Erfassung.
    "kompensator": {
        "zeta": 0.50,
        "quelle": "Annahme (unbelegt) für Axial-/Wellrohrkompensator; "
                  "über fittings_katalog.toml anpassbar",
    },
}

#: Optionale Überschreibung des Katalogs.
KATALOG_PFAD = Path(__file__).parent / "data" / "fittings_katalog.toml"

#: Dateiname der händisch erfassten Einbauteile, abgelegt im Projekt-Root
#: (neben ``project.toml``).
MANUAL_DATEINAME = "fittings_manual.json"

_GUELTIGE_QUELLEN = ("auto", "plan", "geschaetzt", "default")

#: Formteiltypen, die an der **Kante** erfasst werden können. Der Schieber steht
#: bewusst in beiden Listen: er sitzt mal an einem Knoten (Hausanschluss,
#: Abzweig), mal mitten im Rohr — dort ist er **additiv** zu dem, was aus den
#: Knoten kommt.
KANTEN_TYPEN = ("kompensator", "schieber")

#: Formteiltypen, die es **nur** an der Kante gibt. Nur diese werden aus einer
#: Knoten-Erfassung herausgefiltert; ein am Knoten erfasster Schieber muss
#: weiterhin am Knoten landen.
NUR_KANTEN_TYPEN = ("kompensator",)

#: Freies ζ, das direkt auf das Rohr addiert wird (keine Stückzahl).
FREIES_ZETA_FELD = "sonstiges_zeta"

#: Optionale Abschnitte der Erfassungsdatei. Sie werden nur gelesen und
#: geschrieben, wenn sie vorhanden sind — bestehende Dateien ohne sie bleiben
#: gültig, und eine Datei ohne Rohr-Eigenschaften bekommt die Schlüssel nicht
#: nachträglich untergeschoben.
#: * ``eigenschaften`` — je Kante ein Dict frei definierter Eigenschaften
#:   (Baujahr, Material …), Schlüssel wie bei ``kanten``.
#: * ``felder`` — die Definitionen dieser Eigenschaften (Liste von Dicts). Sie
#:   werden hier nur durchgereicht; was ein gültiges Feld ist, entscheidet die
#:   erfassende Oberfläche.
OPTIONALE_ABSCHNITTE = ("eigenschaften", "felder")


# -- Katalog -----------------------------------------------------------------

def lade_katalog(pfad=None, logger=None):
    """Liest den ζ-Katalog und legt ihn über die eingebauten Startwerte.

    Parameters
    ----------
    pfad : str oder Path, optional
        TOML-Datei mit einem Abschnitt je Formteiltyp
        (``[bogen]`` / ``zeta = 0.08`` / ``quelle = "..."``). Ohne Angabe wird
        :data:`KATALOG_PFAD` genommen, falls vorhanden.
    logger : logging.Logger, optional

    Returns
    -------
    dict
        ``{typ: {"zeta": float, "quelle": str}}`` — Kopie, das Modul-Original
        bleibt unverändert.
    """
    katalog = {typ: dict(werte) for typ, werte in KATALOG.items()}

    if pfad is None:
        pfad = KATALOG_PFAD if KATALOG_PFAD.exists() else None
    if pfad is None:
        return katalog

    pfad = Path(pfad)
    if not pfad.exists():
        raise FileNotFoundError(f"Formteil-Katalog nicht gefunden: {pfad}")

    with open(pfad, "rb") as f:
        daten = tomllib.load(f)

    for typ, werte in daten.items():
        if not isinstance(werte, dict):
            continue
        eintrag = katalog.setdefault(typ, {"zeta": None, "quelle": "unbekannt"})
        if "zeta" in werte:
            eintrag["zeta"] = float(werte["zeta"])
        if "quelle" in werte:
            eintrag["quelle"] = str(werte["quelle"])

    if logger is not None:
        logger.info("Formteil-Katalog aus %s gelesen (%d Typen).", pfad, len(daten))
    return katalog


def _zeta(katalog, typ):
    """ζ eines Formteiltyps aus dem Katalog, 0.0 wenn unbekannt."""
    eintrag = katalog.get(typ) or {}
    wert = eintrag.get("zeta")
    return 0.0 if wert is None else float(wert)


# -- Händisch erfasste Einbauteile -------------------------------------------

def datei_hash(pfad):
    """sha256 einer Datei als Hex, ``None`` wenn sie nicht existiert."""
    p = Path(pfad)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def quellen_hashes(quellen):
    """``{logischer Name: sha256}`` für die Eingangsdateien.

    *quellen* ist ``{"network.geojson": pfad, "buildings.geojson": pfad, …}`` —
    der logische Name bleibt stabil, auch wenn die Datei anders heißt.
    """
    return {name: datei_hash(pfad) for name, pfad in (quellen or {}).items()}


def kanten_schluessel(u, v):
    """Kanten-Schlüssel der Erfassungsdatei, z.B. ``"1042->1044"``."""
    return f"{u}->{v}"


def leere_manual():
    """Leerer Erfassungsstand — dieselbe Form wie :func:`load_manual`."""
    return {"quellen_hash": {}, "knoten": {}, "kanten": {}, "warnung": None}


def load_manual(pfad, quellen=None, logger=None):
    """Liest ``fittings_manual.json`` und prüft die Eingangsdaten.

    **Warum die Prüfung:** Knotennummern sind nur so lange stabil, wie die
    GeoJSONs unverändert sind — ``UESGraph.add_network_node`` vergibt sie in
    Feature-Reihenfolge. Ändert sich eine Quelldatei, kann die Zuordnung der
    erfassten Einbauteile verrutschen. Deshalb wird der beim Speichern
    festgehaltene ``quellen_hash`` beim Laden gegengeprüft und eine Abweichung
    **laut gemeldet**, statt sie stillschweigend anzuwenden.

    Parameters
    ----------
    pfad : str oder Path
        Datei; existiert sie nicht, kommt ein leerer Stand zurück.
    quellen : dict, optional
        ``{logischer Name: pfad}`` der Eingangsdateien für den Abgleich. Ohne
        Angabe wird nicht geprüft.

    Returns
    -------
    dict
        ``{"quellen_hash": {...}, "knoten": {...}, "kanten": {...},
        "warnung": str oder None}``. ``warnung`` ist der Text der
        Hash-Abweichung — der Aufrufer zeigt ihn an.
    """
    if logger is None:
        logger = set_up_file_logger(f"{__name__}.load_manual", level=logging.INFO)

    daten = leere_manual()
    p = Path(pfad)
    if not p.is_file():
        return daten

    with open(p, "r", encoding="utf-8") as f:
        roh = json.load(f)

    daten["quellen_hash"] = dict(roh.get("quellen_hash") or {})
    for bereich in ("knoten", "kanten"):
        daten[bereich] = {str(k): dict(v)
                          for k, v in (roh.get(bereich) or {}).items()
                          if isinstance(v, dict)}
    # Optionale Abschnitte nur übernehmen, wenn sie in der Datei stehen —
    # sonst wüchse jede alte Datei beim ersten Speichern um leere Schlüssel.
    if isinstance(roh.get("eigenschaften"), dict):
        daten["eigenschaften"] = {str(k): dict(v)
                                  for k, v in roh["eigenschaften"].items()
                                  if isinstance(v, dict)}
    if isinstance(roh.get("felder"), list):
        daten["felder"] = [dict(f) for f in roh["felder"] if isinstance(f, dict)]

    if quellen:
        jetzt = quellen_hashes(quellen)
        abweichend = sorted(name for name, alt in daten["quellen_hash"].items()
                            if jetzt.get(name) != alt)
        if abweichend:
            daten["warnung"] = (
                "Die Eingangsdaten haben sich seit dem Erfassen geändert "
                f"({', '.join(abweichend)}). Knotennummern werden in "
                "Feature-Reihenfolge vergeben — die Zuordnung der Einbauteile "
                "kann verrutscht sein. Bitte prüfen, bevor die Werte weiter "
                "benutzt werden."
            )
            logger.warning(daten["warnung"])

    logger.info("Einbauteile gelesen aus %s: %d Knoten, %d Kanten.",
                p, len(daten["knoten"]), len(daten["kanten"]))
    return daten


def save_manual(pfad, daten, quellen=None, logger=None):
    """Schreibt ``fittings_manual.json``.

    *quellen* (``{logischer Name: pfad}``) wird zu ``quellen_hash`` verrechnet;
    ohne Angabe bleibt der Hash aus *daten* stehen. Geschrieben wird UTF-8 mit
    Einrückung, damit die Datei im Diff lesbar bleibt.
    """
    if logger is None:
        logger = set_up_file_logger(f"{__name__}.save_manual", level=logging.INFO)

    inhalt = {
        "quellen_hash": (quellen_hashes(quellen) if quellen is not None
                         else dict(daten.get("quellen_hash") or {})),
        "knoten": {str(k): dict(v) for k, v in (daten.get("knoten") or {}).items()},
        "kanten": {str(k): dict(v) for k, v in (daten.get("kanten") or {}).items()},
    }
    # Optionale Abschnitte werden geschrieben, sobald der Schlüssel vorhanden
    # ist — auch leer. So bleibt "der Nutzer hat alle Felder gelöscht" erhalten,
    # während eine Datei, die nie Eigenschaften kannte, unverändert bleibt.
    if isinstance(daten.get("eigenschaften"), dict):
        inhalt["eigenschaften"] = {str(k): dict(v)
                                   for k, v in daten["eigenschaften"].items()}
    if isinstance(daten.get("felder"), list):
        inhalt["felder"] = [dict(f) for f in daten["felder"]]
    p = Path(pfad)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(inhalt, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    logger.info("Einbauteile geschrieben nach %s: %d Knoten, %d Kanten.",
                p, len(inhalt["knoten"]), len(inhalt["kanten"]))
    return p


# -- Geometrie ---------------------------------------------------------------

def ablenkwinkel_deg(p_mitte, p_1, p_2):
    """Ablenkwinkel im Knoten zwischen zwei Nachbarn, in Grad.

    0° heißt geradeaus (kollinear), 90° ein rechtwinkliger Bogen. Erwartet
    Objekte mit ``.x``/``.y`` (shapely Point), wie sie unter dem Knotenattribut
    ``position`` liegen.

    Returns
    -------
    float oder None
        ``None``, wenn eine der beiden Kanten die Länge null hat.
    """
    v1 = (p_1.x - p_mitte.x, p_1.y - p_mitte.y)
    v2 = (p_2.x - p_mitte.x, p_2.y - p_mitte.y)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0.0 or n2 == 0.0:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    # Der Winkel zwischen den beiden Kanten ist 180° bei geradem Durchlauf;
    # die Ablenkung ist der Rest zu 180°.
    return 180.0 - math.degrees(math.acos(cos))


def zeta_querschnitt(d_klein, d_gross):
    """ζ eines Querschnittssprungs, gemittelt über beide Strömungsrichtungen.

    Im Vorlauf ist der Sprung eine Verengung, im Rücklauf eine Erweiterung.
    Da ζ identisch auf Vor- und Rücklauf gelegt wird (Konvention "je Strang"),
    ist der Mittelwert beider Richtungen die passende Repräsentation. Beide
    Beiwerte sind auf die Geschwindigkeit im **kleinen** Querschnitt bezogen.

    Erweiterung (Borda-Carnot): ``(1 - A_klein/A_gross)**2``
    Verengung (Idelchik, scharfkantig): ``0.5 * (1 - A_klein/A_gross)``
    """
    if not d_klein or not d_gross or d_klein <= 0 or d_gross <= 0:
        return 0.0
    if d_klein > d_gross:
        d_klein, d_gross = d_gross, d_klein
    flaechenverhaeltnis = (d_klein / d_gross) ** 2
    zeta_erw = (1.0 - flaechenverhaeltnis) ** 2
    zeta_ver = 0.5 * (1.0 - flaechenverhaeltnis)
    return 0.5 * (zeta_erw + zeta_ver)


def _kante_durchmesser(edge_data):
    """Innendurchmesser einer Kante in m; ``diameter`` zuerst, dann DN.

    Lookup-Muster wie in ``pipe_specs``: erst direkt, dann im ``attr_dict``,
    wo uesgraphs die rohen GeoJSON-Eigenschaften ablegt.
    """
    for schluessel, faktor in (("diameter", 1.0), ("DN", 1e-3)):
        wert = edge_data.get(schluessel)
        if wert is None:
            nested = edge_data.get("attr_dict") or {}
            wert = nested.get(schluessel)
        if wert is None:
            continue
        if isinstance(wert, str):
            wert = wert.strip().upper().lstrip("DN").strip()
        try:
            return float(wert) * faktor
        except (TypeError, ValueError):
            continue
    return None


def _kanten_daten(graph, u, v):
    return graph.edges[u, v]


def _knoten_struktur(graph, node, winkel_schwelle_deg):
    """Formteile, Kantenrollen und Ablenkwinkel eines Knotens.

    Returns
    -------
    dict
        ``formteile``  — ``{typ: anzahl}``, nur vorhandene Typen
        ``rollen``     — ``{nachbar: typ}``: welche Rolle die Kante
                         ``(node, nachbar)`` im Knoten spielt
        ``winkel``     — Ablenkwinkel in Grad (nur Grad 2), sonst ``None``
        ``zeta_extra`` — ``{typ: zeta}`` für Beiwerte, die nicht aus dem
                         Katalog kommen (Querschnittssprung)
    """
    nachbarn = list(graph.neighbors(node))
    grad = len(nachbarn)
    nd = graph.nodes[node]
    struktur = {"formteile": {}, "rollen": {}, "winkel": None, "zeta_extra": {}}

    if grad == 0:
        return struktur

    if grad == 1:
        # Grad 1 am Gebäude ist ein Hausanschluss — Absperrung als Annahme.
        # Freie Enden im Netz (Blindstopfen) bekommen nichts.
        if nd.get("node_type") == "building":
            struktur["formteile"]["schieber"] = SCHIEBER_JE_HAUSANSCHLUSS
            struktur["rollen"][nachbarn[0]] = "schieber"
        return struktur

    pos = nd.get("position")
    positionen = {n: graph.nodes[n].get("position") for n in nachbarn}

    if grad == 2:
        a, b = nachbarn
        winkel = None
        if pos is not None and positionen[a] is not None and positionen[b] is not None:
            winkel = ablenkwinkel_deg(pos, positionen[a], positionen[b])
        struktur["winkel"] = winkel
        if winkel is not None and winkel >= winkel_schwelle_deg:
            struktur["formteile"]["bogen"] = 1
            struktur["rollen"][a] = "bogen"
            struktur["rollen"][b] = "bogen"
        # Ein DN-Wechsel mitten im Strang ist ein Reduzierstück.
        d_a = _kante_durchmesser(_kanten_daten(graph, node, a))
        d_b = _kante_durchmesser(_kanten_daten(graph, node, b))
        if d_a and d_b and abs(d_a - d_b) > 1e-9:
            struktur["formteile"]["querschnittssprung"] = 1
            struktur["zeta_extra"]["querschnittssprung"] = zeta_querschnitt(d_a, d_b)
            struktur["rollen"].setdefault(a, "querschnittssprung")
            struktur["rollen"].setdefault(b, "querschnittssprung")
        return struktur

    # Grad >= 3: T-Stück. Durchgang ist das kollinearste Kantenpaar, alles
    # Übrige ist Abzweig. Ein DN-Wechsel ist hier im T-Beiwert enthalten und
    # wird nicht zusätzlich gezählt.
    durchgang = _kollinearstes_paar(pos, nachbarn, positionen)
    if durchgang is None:
        durchgang = (nachbarn[0], nachbarn[1])

    for n in nachbarn:
        struktur["rollen"][n] = "t_durchgang" if n in durchgang else "t_abzweig"
    struktur["formteile"]["t_durchgang"] = 1
    struktur["formteile"]["t_abzweig"] = grad - 2
    return struktur


def _kollinearstes_paar(pos, nachbarn, positionen):
    """Das Nachbarpaar mit der kleinsten Ablenkung (= der Durchgang)."""
    if pos is None:
        return None
    bestes, bester_winkel = None, None
    for i, a in enumerate(nachbarn):
        for b in nachbarn[i + 1:]:
            if positionen.get(a) is None or positionen.get(b) is None:
                continue
            winkel = ablenkwinkel_deg(pos, positionen[a], positionen[b])
            if winkel is None:
                continue
            if bester_winkel is None or winkel < bester_winkel:
                bestes, bester_winkel = (a, b), winkel
    return bestes


# -- Schritt 1: Zählungen ----------------------------------------------------

def derive_from_geometry(
    graph,
    winkel_schwelle_deg=BOGEN_WINKEL_SCHWELLE_DEG,
    overwrite=False,
    logger=None,
):
    """Zählt je Knoten die Formteile aus der Geometrie.

    Abgeleitet wird aus Knotengrad, Ablenkwinkel und DN-Wechsel:

    * **Grad 1 am Gebäude** → Hausanschluss, ein Schieber als Annahme.
    * **Grad 2** → Bogen, sofern der Ablenkwinkel über ``winkel_schwelle_deg``
      liegt; darunter ist es ein Vermessungspunkt der Trasse. Zusätzlich ein
      Querschnittssprung, wenn sich der Durchmesser ändert.
    * **Grad ≥ 3** → T-Stück: das kollinearste Kantenpaar ist der Durchgang,
      jede weitere Kante ein Abzweig.

    Gibt **Zählungen** zurück, keine ζ — die Bewertung macht
    :func:`node_zeta_to_edges`.

    Parameters
    ----------
    graph : uesgraphs.UESGraph
        Wird in-place ergänzt: ``formteile``, ``formteile_quelle`` und bei
        Grad-2-Knoten ``bogen_winkel_deg``.
    winkel_schwelle_deg : float
        Schwelle "echter Bogen vs. Vermessungspunkt".
    overwrite : bool
        Wenn False, bleiben bereits gesetzte ``formteile`` unangetastet.

    Returns
    -------
    dict
        ``{knoten: {typ: anzahl}}`` für alle bearbeiteten Knoten.
    """
    if logger is None:
        logger = set_up_file_logger(
            f"{__name__}.derive_from_geometry", level=logging.INFO
        )

    zaehlungen = {}
    n_gesetzt, n_behalten, n_ohne_position = 0, 0, 0

    for node, nd in graph.nodes(data=True):
        if "formteile" in nd and not overwrite:
            zaehlungen[node] = dict(nd["formteile"])
            n_behalten += 1
            continue

        struktur = _knoten_struktur(graph, node, winkel_schwelle_deg)
        if nd.get("position") is None and len(list(graph.neighbors(node))) >= 2:
            n_ohne_position += 1

        nd["formteile"] = dict(struktur["formteile"])
        nd["formteile_quelle"] = "auto"
        if struktur["winkel"] is not None:
            nd["bogen_winkel_deg"] = float(struktur["winkel"])
        zaehlungen[node] = dict(struktur["formteile"])
        n_gesetzt += 1

    if n_ohne_position:
        logger.warning(
            "%d Knoten ohne 'position' — dort konnten Winkel und Durchgang "
            "nicht bestimmt werden.", n_ohne_position,
        )
    logger.info(
        "Formteile abgeleitet: %d Knoten gesetzt, %d vorhandene behalten.",
        n_gesetzt, n_behalten,
    )
    return zaehlungen


# -- Schritt 2: Knotenverluste auf Kanten ------------------------------------

def _verteile_grad3(p_ab, p_ac, p_bc):
    """Exakte Lösung des Dreiecksystems für einen Grad-3-Knoten.

    ``a + b = p_ab``, ``a + c = p_ac``, ``b + c = p_bc`` — eindeutig lösbar,
    damit trägt jeder Weg durch den Knoten exakt seinen Verlust.
    """
    a = 0.5 * (p_ab + p_ac - p_bc)
    b = 0.5 * (p_ab + p_bc - p_ac)
    c = 0.5 * (p_ac + p_bc - p_ab)
    return a, b, c


def _verteile_kleinste_quadrate(nachbarn, paar_verluste):
    """Kleinste Quadrate mit ζ ≥ 0 für Knoten ab Grad 4.

    Eine Zeile je Kantenpaar (``x_i + x_j = ζ_weg``), gelöst mit
    nicht-negativen kleinsten Quadraten. Gibt Anteile und Restfehler zurück.
    """
    import numpy as np
    from scipy.optimize import nnls

    index = {n: i for i, n in enumerate(nachbarn)}
    A, b = [], []
    for (i, j), verlust in paar_verluste.items():
        zeile = [0.0] * len(nachbarn)
        zeile[index[i]] = 1.0
        zeile[index[j]] = 1.0
        A.append(zeile)
        b.append(verlust)
    x, rest = nnls(np.array(A), np.array(b))
    return {n: float(x[index[n]]) for n in nachbarn}, float(rest)


def _paar_verluste(struktur, katalog):
    """ζ je Weg durch den Knoten, ``{(nachbar_i, nachbar_j): zeta}``.

    Zwei Anteile, die sich addieren:

    * der **T-Anteil**, der davon abhängt, welche Rolle die beiden Kanten im
      Knoten spielen (Durchgang↔Durchgang vs. irgendetwas↔Abzweig);
    * ein **Zusatz** aus allen übrigen Formteilen des Knotens (Bogen,
      Schieber, Querschnittssprung, Kompensator …) — die liegen in *jedem*
      Weg durch den Knoten.

    Bei rein automatisch abgeleiteten Knoten ist immer genau einer der beiden
    Anteile besetzt (Grad ≥ 3 nur T, Grad ≤ 2 nur Zusatz). Getrennt gerechnet
    werden sie, damit **händisch erfasste** Zusatzteile an einem T-Knoten
    (z.B. ein Schieber im Abzweig) mitzählen.
    """
    rollen = struktur["rollen"]
    nachbarn = list(rollen)
    extra = struktur["zeta_extra"]
    formteile = struktur["formteile"]

    zusatz = 0.0
    for typ, anzahl in formteile.items():
        if typ in ("t_durchgang", "t_abzweig"):
            continue
        zusatz += anzahl * extra.get(typ, _zeta(katalog, typ))
    hat_t = bool(formteile.get("t_durchgang") or formteile.get("t_abzweig"))

    verluste = {}
    for i, a in enumerate(nachbarn):
        for b in nachbarn[i + 1:]:
            t_anteil = 0.0
            if hat_t:
                if rollen[a] == "t_durchgang" and rollen[b] == "t_durchgang":
                    t_anteil = _zeta(katalog, "t_durchgang")
                elif "t_durchgang" in (rollen[a], rollen[b]) or \
                     "t_abzweig" in (rollen[a], rollen[b]):
                    t_anteil = _zeta(katalog, "t_abzweig")
            verluste[(a, b)] = t_anteil + zusatz
    return verluste


def node_zeta_to_edges(graph, katalog=None,
                       winkel_schwelle_deg=BOGEN_WINKEL_SCHWELLE_DEG,
                       logger=None):
    """Verteilt die Knotenverluste auf die anliegenden Kanten.

    Weder pandapipes noch Modelica kennen Knotenverluste — der Verlust muss auf
    die Rohre. Verteilt wird so, dass jeder Weg durch den Knoten **exakt**
    seinen ζ trägt:

    * **Grad 1** → der ganze Knotenverlust auf die eine Kante.
    * **Grad 2** → halbe-halbe (unterbestimmt, jeder Weg stimmt trotzdem).
    * **Grad 3** → eindeutig lösbar: Durchgangskanten je ``ζ_durchgang/2``,
      der Abzweig ``ζ_abzweig − ζ_durchgang/2``.
    * **Grad ≥ 4** → überbestimmt, kleinste Quadrate mit ζ ≥ 0; der Restfehler
      wird geloggt.

    Returns
    -------
    dict
        ``{(u, v): {"sum_zetas": float, "zeta_herkunft": [ {...}, ... ]}}``
        mit ``(u, v)`` genau so, wie die Kante im Graphen liegt.
    """
    if logger is None:
        logger = set_up_file_logger(
            f"{__name__}.node_zeta_to_edges", level=logging.INFO
        )
    if katalog is None:
        katalog = lade_katalog()

    anteile = {}   # frozenset({u, v}) -> [summe, herkunft]
    rest_gesamt, n_geklemmt = 0.0, 0

    for node in graph.nodes():
        struktur = _knoten_struktur(graph, node, winkel_schwelle_deg)
        nd = graph.nodes[node]
        if nd.get("formteile") is not None and nd.get("formteile_quelle") not in (
            None, "auto"
        ):
            # Von Hand erfasste Zählungen haben Vorrang vor der Geometrie. Die
            # Kantenrollen bleiben die geometrischen; Kanten, die die Geometrie
            # gar nicht bedacht hat (Knoten unter der Bogenschwelle), kommen als
            # "sonstiges" dazu, sonst verpufft die Erfassung dort.
            struktur["formteile"] = dict(nd["formteile"])
            for nachbar in graph.neighbors(node):
                struktur["rollen"].setdefault(nachbar, "sonstiges")
        if not struktur["formteile"]:
            continue

        rollen = struktur["rollen"]
        nachbarn = list(rollen)
        extra = struktur["zeta_extra"]

        # herkunft: je Nachbarkante eine Liste (typ, anzahl, anteil)
        herkunft = {n: [] for n in nachbarn}

        if len(nachbarn) <= 2:
            # Grad 1: alles auf die eine Kante. Grad 2: halbe-halbe — der Weg
            # durch den Knoten trägt damit trotzdem den vollen Verlust.
            teiler = float(len(nachbarn))
            geloest = {n: 0.0 for n in nachbarn}
            for typ, anzahl in struktur["formteile"].items():
                zeta_typ = anzahl * extra.get(typ, _zeta(katalog, typ))
                for n in nachbarn:
                    geloest[n] += zeta_typ / teiler
                    herkunft[n].append((typ, anzahl, zeta_typ / teiler))
        elif len(nachbarn) == 3:
            verluste = _paar_verluste(struktur, katalog)
            a, b, c = nachbarn
            geloest = dict(zip(
                (a, b, c),
                _verteile_grad3(
                    verluste[_paar(verluste, a, b)],
                    verluste[_paar(verluste, a, c)],
                    verluste[_paar(verluste, b, c)],
                ),
            ))
            for n in nachbarn:
                herkunft[n].append((rollen[n], 1, geloest[n]))
        else:
            verluste = _paar_verluste(struktur, katalog)
            geloest, rest = _verteile_kleinste_quadrate(nachbarn, verluste)
            rest_gesamt += rest
            for n in nachbarn:
                herkunft[n].append((rollen[n], 1, geloest[n]))

        for n, wert in geloest.items():
            if wert < 0.0:
                n_geklemmt += 1
                wert = 0.0
                herkunft[n] = [(typ, anzahl, 0.0) for typ, anzahl, _ in herkunft[n]]
            if wert == 0.0:
                continue
            schluessel = frozenset((node, n))
            eintrag = anteile.setdefault(schluessel, [0.0, []])
            eintrag[0] += wert
            for typ, anzahl, anteil in herkunft[n]:
                if anteil == 0.0 or not anzahl:
                    continue
                # ``zeta`` ist der Anteil **je Stück**; der Beitrag des
                # Eintrags zu sum_zetas ist anzahl * zeta.
                eintrag[1].append({
                    "typ": typ,
                    "knoten": node,
                    "anzahl": int(anzahl),
                    "zeta": round(float(anteil) / int(anzahl), 9),
                })

    if n_geklemmt:
        logger.warning(
            "%d Kantenanteile waren negativ und wurden auf 0 geklemmt "
            "(ζ_abzweig < ζ_durchgang/2?).", n_geklemmt,
        )
    if rest_gesamt:
        logger.info(
            "Kleinste Quadrate an Knoten ab Grad 4: Restfehler %.4f gesamt.",
            rest_gesamt,
        )

    ergebnis = {}
    for u, v in graph.edges():
        summe, herkunft = anteile.get(frozenset((u, v)), (0.0, []))
        ergebnis[(u, v)] = {
            "sum_zetas": float(summe),
            "zeta_herkunft": list(herkunft),
        }
    return ergebnis


def _paar(verluste, a, b):
    """Schlüssel des Kantenpaares in ``verluste``, egal in welcher Reihenfolge."""
    return (a, b) if (a, b) in verluste else (b, a)


# -- Schritt 3: auf den Graphen schreiben ------------------------------------

def _knoten_index(graph):
    """``{str(knoten): knoten}`` — die Erfassungsdatei kennt nur Zeichenketten."""
    return {str(n): n for n in graph.nodes()}


def uebernimm_manuelle_knoten(graph, manual, logger=None):
    """Legt die erfassten Knoten-Zählungen über die abgeleiteten.

    Läuft **zwischen** :func:`derive_from_geometry` und
    :func:`node_zeta_to_edges`: die Zählung kommt aus der Erfassung, die
    Verteilung auf die Kanten macht weiterhin die Geometrie.

    Returns
    -------
    int
        Anzahl übernommener Knoten.
    """
    eintraege = (manual or {}).get("knoten") or {}
    if not eintraege:
        return 0
    index = _knoten_index(graph)
    n_ok, unbekannt = 0, []
    for schluessel, eintrag in eintraege.items():
        node = index.get(str(schluessel))
        if node is None:
            unbekannt.append(schluessel)
            continue
        zaehlungen = {}
        for typ, wert in eintrag.items():
            if typ == "quelle" or typ in NUR_KANTEN_TYPEN:
                continue
            try:
                anzahl = int(wert)
            except (TypeError, ValueError):
                continue
            if anzahl > 0:
                zaehlungen[typ] = anzahl
        graph.nodes[node]["formteile"] = zaehlungen
        graph.nodes[node]["formteile_quelle"] = str(eintrag.get("quelle") or "plan")
        n_ok += 1
    if unbekannt and logger is not None:
        logger.warning(
            "%d erfasste Knoten gibt es im Graphen nicht (%s …) — die "
            "Eingangsdaten passen vermutlich nicht zur Erfassung.",
            len(unbekannt), ", ".join(sorted(unbekannt)[:5]),
        )
    return n_ok


def uebernimm_manuelle_kanten(graph, manual, katalog, logger=None):
    """Addiert die erfassten Kanten-Einbauteile auf ``sum_zetas``.

    An der Kante sitzt alles, was *keinem* Knoten zuzuordnen ist: Kompensatoren
    in der Trasse, Absperrschieber mitten im Rohr, dazu ein freies
    ``sonstiges_zeta``. Gezählt wird **jeder** Formteiltyp, der im Katalog einen
    festen ζ-Wert hat — die Funktion kennt keine feste Typenliste, ein neuer
    Eintrag im Katalog genügt.

    Die Kantenteile kommen **auf** den aus der Geometrie verteilten
    Knotenanteil obendrauf, stehen als eigener Eintrag in ``zeta_herkunft``
    (mit ``knoten: None``) und setzen ``zeta_quelle`` auf die erfasste Quelle.
    Der Vertrag ``sum_zetas == Σ anzahl * zeta`` bleibt damit erhalten.

    Returns
    -------
    int
        Anzahl geänderter Kanten.
    """
    eintraege = (manual or {}).get("kanten") or {}
    if not eintraege:
        return 0
    index = _knoten_index(graph)
    n_ok, unbekannt, unbekannte_typen = 0, [], set()
    for schluessel, eintrag in eintraege.items():
        teile = str(schluessel).split("->")
        u = index.get(teile[0].strip()) if len(teile) == 2 else None
        v = index.get(teile[1].strip()) if len(teile) == 2 else None
        if u is None or v is None or not graph.has_edge(u, v):
            unbekannt.append(schluessel)
            continue
        zusatz, herkunft = 0.0, []
        for typ, wert in eintrag.items():
            if typ == "quelle":
                continue
            if typ == FREIES_ZETA_FELD:
                try:
                    z = float(wert)
                except (TypeError, ValueError):
                    continue
                if z:
                    zusatz += z
                    herkunft.append({"typ": "sonstiges", "knoten": None,
                                     "anzahl": 1, "zeta": round(z, 9)})
                continue
            # Nur Typen mit festem ζ aus dem Katalog. Ein Querschnittssprung
            # (ζ = None, wird je Knoten gerechnet) oder ein Tippfehler soll
            # auffallen statt still einen Eintrag mit ζ = 0 zu erzeugen.
            if (katalog.get(typ) or {}).get("zeta") is None:
                unbekannte_typen.add(str(typ))
                continue
            try:
                anzahl = int(wert)
            except (TypeError, ValueError):
                continue
            if anzahl <= 0:
                continue
            z = _zeta(katalog, typ)
            zusatz += anzahl * z
            herkunft.append({"typ": typ, "knoten": None,
                             "anzahl": anzahl, "zeta": round(z, 9)})
        data = graph.edges[u, v]
        basis = data.get("sum_zetas")
        data["sum_zetas"] = float(basis or 0.0) + zusatz
        data["zeta_herkunft"] = list(data.get("zeta_herkunft") or []) + herkunft
        data["zeta_quelle"] = str(eintrag.get("quelle") or "plan")
        n_ok += 1
    if unbekannt and logger is not None:
        logger.warning(
            "%d erfasste Kanten gibt es im Graphen nicht (%s …).",
            len(unbekannt), ", ".join(sorted(unbekannt)[:5]),
        )
    if unbekannte_typen and logger is not None:
        logger.warning(
            "An Kanten erfasste Formteiltypen ohne festes ζ im Katalog "
            "ignoriert: %s.", ", ".join(sorted(unbekannte_typen)),
        )
    return n_ok


# -- Rohr-Eigenschaften ------------------------------------------------------

def uebernimm_eigenschaften(graph, manual, logger=None):
    """Schreibt die erfassten Rohr-Eigenschaften als Kantenattribute.

    Der Abschnitt ``eigenschaften`` der Erfassungsdatei ist ``{"u->v":
    {feldname: wert}}``. Welche Felder es gibt, entscheidet die erfassende
    Oberfläche (Abschnitt ``felder``) — hier wird nichts geprüft und nichts auf
    Physik abgebildet, die Werte werden nur an den Graphen gereicht.

    Geschrieben wird je Kante ein Attribut je Feld **und** das Dict unter
    ``eigenschaften``. Letzteres ist nicht nur Bequemlichkeit: beim erneuten
    Anwenden auf denselben Graphen müssen die Attribute eines inzwischen
    gelöschten Feldes wieder verschwinden, sonst bliebe ein alter Wert stehen.

    Returns
    -------
    int
        Anzahl geänderter Kanten.
    """
    eintraege = (manual or {}).get("eigenschaften") or {}
    index = _knoten_index(graph)
    n_ok, unbekannt = 0, []

    # Erst überall abräumen, was der letzte Lauf gesetzt hat.
    for u, v, data in graph.edges(data=True):
        for feld in (data.pop("eigenschaften", None) or {}):
            data.pop(feld, None)

    for schluessel, eintrag in eintraege.items():
        teile = str(schluessel).split("->")
        u = index.get(teile[0].strip()) if len(teile) == 2 else None
        v = index.get(teile[1].strip()) if len(teile) == 2 else None
        if u is None or v is None or not graph.has_edge(u, v):
            unbekannt.append(schluessel)
            continue
        werte = {str(k): w for k, w in eintrag.items() if w is not None}
        if not werte:
            continue
        data = graph.edges[u, v]
        data["eigenschaften"] = dict(werte)
        data.update(werte)
        n_ok += 1

    if unbekannt and logger is not None:
        logger.warning(
            "%d Kanten mit erfassten Eigenschaften gibt es im Graphen nicht "
            "(%s …).", len(unbekannt), ", ".join(sorted(unbekannt)[:5]),
        )
    return n_ok


def apply_fittings(
    graph,
    katalog=None,
    overwrite=False,
    winkel_schwelle_deg=BOGEN_WINKEL_SCHWELLE_DEG,
    manual=None,
    logger=None,
):
    """Leitet Formteile ab, verteilt sie auf die Kanten und schreibt sie.

    Setzt je Kante ``sum_zetas`` (float, Σζ je Strang), ``zeta_quelle``
    ("auto") und ``zeta_herkunft`` (Liste von Dicts, JSON-serialisierbar:
    ``{"typ", "knoten", "anzahl", "zeta"}``, wobei ``zeta`` je Stück gilt und
    ``sum_zetas == Σ anzahl * zeta``). Je Knoten kommen ``formteile``,
    ``formteile_quelle`` und ggf. ``bogen_winkel_deg`` dazu.

    Parameters
    ----------
    graph : uesgraphs.UESGraph
        Wird in-place geändert und zum Verketten zurückgegeben.
    katalog : dict, optional
        Ergebnis von :func:`lade_katalog`. Ohne Angabe wird der Katalog
        (inklusive TOML-Überschreibung) geladen.
    overwrite : bool, default False
        Wenn False, bleiben Kanten mit bereits gesetztem ``sum_zetas``
        unangetastet — von Hand erfasste oder aus Plänen übernommene Werte
        ("plan", "geschaetzt") überleben also einen erneuten Lauf.
    winkel_schwelle_deg : float
        Schwelle "echter Bogen vs. Vermessungspunkt", siehe
        :data:`BOGEN_WINKEL_SCHWELLE_DEG`.
    manual : dict, optional
        Händisch erfasste Einbauteile aus :func:`load_manual`
        (``{"knoten": …, "kanten": …, "eigenschaften": …}``). **Manuell schlägt
        auto:** erfasste Knoten-Zählungen ersetzen die abgeleiteten, erfasste
        Kantenteile (jeder Katalogtyp mit festem ζ, dazu ``sonstiges_zeta``)
        kommen obendrauf. Erfasste Rohr-Eigenschaften (``eigenschaften``)
        landen als Kantenattribute, ohne auf Physik abgebildet zu werden.

    Returns
    -------
    graph : uesgraphs.UESGraph
    """
    if logger is None:
        logger = set_up_file_logger(f"{__name__}.apply_fittings", level=logging.INFO)
    if katalog is None:
        katalog = lade_katalog(logger=logger)

    derive_from_geometry(
        graph, winkel_schwelle_deg=winkel_schwelle_deg,
        overwrite=overwrite, logger=logger,
    )
    n_manuell_knoten = uebernimm_manuelle_knoten(graph, manual, logger=logger)
    kanten = node_zeta_to_edges(
        graph, katalog=katalog,
        winkel_schwelle_deg=winkel_schwelle_deg, logger=logger,
    )

    n_gesetzt, n_behalten = 0, 0
    for (u, v), werte in kanten.items():
        data = graph.edges[u, v]
        if "sum_zetas" in data and not overwrite:
            n_behalten += 1
            continue
        data["sum_zetas"] = werte["sum_zetas"]
        data["zeta_herkunft"] = werte["zeta_herkunft"]
        data["zeta_quelle"] = "auto"
        n_gesetzt += 1

    n_manuell_kanten = uebernimm_manuelle_kanten(graph, manual, katalog, logger=logger)
    n_eigenschaften = uebernimm_eigenschaften(graph, manual, logger=logger)

    logger.info(
        "Einzelwiderstände gesetzt: %d Kanten geschrieben, %d vorhandene behalten "
        "(händisch: %d Knoten, %d Kanten, %d Kanten mit Eigenschaften).",
        n_gesetzt, n_behalten, n_manuell_knoten, n_manuell_kanten,
        n_eigenschaften,
    )
    return graph
