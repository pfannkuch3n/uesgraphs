"""Tests für uesgraphs.fittings.

Der eine Test, der wirklich zählt, ist ``test_t_knoten_...``: an einem
künstlichen T-Knoten muss für jeden der drei Wege gelten
``Σ Kantenanteile == ζ_weg``. Der Rest sind schmale Tests für Winkel,
Katalog-Laden und den Attribut-Vertrag.
"""

import logging

import networkx as nx
import pytest
from shapely.geometry import Point

from uesgraphs.fittings import (
    KATALOG,
    ablenkwinkel_deg,
    apply_fittings,
    derive_from_geometry,
    lade_katalog,
    node_zeta_to_edges,
    zeta_querschnitt,
)


LOGGER = logging.getLogger("test_fittings")


def _graph(nodes, edges):
    """Kleiner nx-Graph mit shapely-Positionen, wie ihn uesgraphs liefert."""
    g = nx.Graph()
    for name, (x, y), attrs in nodes:
        g.add_node(name, position=Point(x, y), **attrs)
    for u, v, attrs in edges:
        g.add_edge(u, v, **attrs)
    return g


@pytest.fixture
def t_graph():
    """T-Knoten: A---M---B gerade, C zweigt rechtwinklig ab. Alles DN 100."""
    nodes = [
        ("M", (0.0, 0.0), {"node_type": "network_heating"}),
        ("A", (-10.0, 0.0), {"node_type": "network_heating"}),
        ("B", (10.0, 0.0), {"node_type": "network_heating"}),
        ("C", (0.0, 10.0), {"node_type": "network_heating"}),
    ]
    edges = [
        ("M", "A", {"diameter": 0.1, "length": 10.0}),
        ("M", "B", {"diameter": 0.1, "length": 10.0}),
        ("M", "C", {"diameter": 0.1, "length": 10.0}),
    ]
    return _graph(nodes, edges)


# -- Winkel ------------------------------------------------------------------

def test_ablenkwinkel_gerade_ist_null():
    assert ablenkwinkel_deg(Point(0, 0), Point(-1, 0), Point(1, 0)) == pytest.approx(0.0)


def test_ablenkwinkel_rechter_winkel():
    assert ablenkwinkel_deg(Point(0, 0), Point(-1, 0), Point(0, 1)) == pytest.approx(90.0)


def test_ablenkwinkel_ohne_laenge_ist_none():
    assert ablenkwinkel_deg(Point(0, 0), Point(0, 0), Point(1, 0)) is None


# -- Katalog -----------------------------------------------------------------

def test_katalog_laedt_mit_quelle_je_wert():
    katalog = lade_katalog()
    for typ in ("bogen", "t_durchgang", "t_abzweig", "schieber"):
        assert katalog[typ]["zeta"] > 0
        assert katalog[typ]["quelle"]


def test_katalog_toml_ueberschreibt(tmp_path):
    pfad = tmp_path / "katalog.toml"
    pfad.write_text('[bogen]\nzeta = 0.5\nquelle = "Testwert"\n', encoding="utf-8")
    katalog = lade_katalog(pfad)
    assert katalog["bogen"]["zeta"] == 0.5
    assert katalog["bogen"]["quelle"] == "Testwert"
    # unberührte Typen behalten den Startwert
    assert katalog["schieber"]["zeta"] == KATALOG["schieber"]["zeta"]
    # das Modul-Original bleibt unangetastet
    assert KATALOG["bogen"]["zeta"] == 0.08


def test_katalog_fehlender_pfad_wirft(tmp_path):
    with pytest.raises(FileNotFoundError):
        lade_katalog(tmp_path / "gibtsnicht.toml")


# -- Der Kern: Verteilung am T-Knoten ----------------------------------------

def test_t_knoten_jeder_weg_traegt_seinen_zeta(t_graph):
    """Σ der Kantenanteile == ζ des Weges — für alle drei Wege durch das T."""
    katalog = lade_katalog()
    kanten = node_zeta_to_edges(t_graph, katalog=katalog, logger=LOGGER)

    def anteil(u, v):
        return kanten.get((u, v), kanten.get((v, u)))["sum_zetas"]

    a, b, c = anteil("M", "A"), anteil("M", "B"), anteil("M", "C")
    zeta_d = katalog["t_durchgang"]["zeta"]
    zeta_ab = katalog["t_abzweig"]["zeta"]

    assert a + b == pytest.approx(zeta_d)    # Durchgang A -> B
    assert a + c == pytest.approx(zeta_ab)   # Abzweig  A -> C
    assert b + c == pytest.approx(zeta_ab)   # Abzweig  B -> C


def test_t_knoten_rollen_und_herkunft(t_graph):
    derive_from_geometry(t_graph, logger=LOGGER)
    assert t_graph.nodes["M"]["formteile"] == {"t_durchgang": 1, "t_abzweig": 1}
    assert t_graph.nodes["M"]["formteile_quelle"] == "auto"

    kanten = node_zeta_to_edges(t_graph, logger=LOGGER)
    herkunft = kanten[("M", "C")]["zeta_herkunft"]
    assert [e["typ"] for e in herkunft] == ["t_abzweig"]
    assert herkunft[0]["knoten"] == "M"
    # Herkunft muss die Summe erklären: zeta gilt je Stück
    assert sum(e["anzahl"] * e["zeta"] for e in herkunft) == pytest.approx(
        kanten[("M", "C")]["sum_zetas"]
    )


# -- Grad 2: Bogen und Schwelle ----------------------------------------------

def _grad2_graph(winkel_punkt):
    nodes = [
        ("M", (0.0, 0.0), {"node_type": "network_heating"}),
        ("A", (-10.0, 0.0), {"node_type": "network_heating"}),
        ("B", winkel_punkt, {"node_type": "network_heating"}),
    ]
    edges = [
        ("M", "A", {"diameter": 0.1, "length": 10.0}),
        ("M", "B", {"diameter": 0.1, "length": 10.0}),
    ]
    return _graph(nodes, edges)


def test_grad2_echter_bogen_wird_halbe_halbe_verteilt():
    g = _grad2_graph((0.0, 10.0))  # 90° Ablenkung
    apply_fittings(g, logger=LOGGER)
    assert g.nodes["M"]["formteile"] == {"bogen": 1}
    assert g.nodes["M"]["bogen_winkel_deg"] == pytest.approx(90.0)
    zeta_bogen = lade_katalog()["bogen"]["zeta"]
    assert g.edges["M", "A"]["sum_zetas"] == pytest.approx(zeta_bogen / 2)
    assert g.edges["M", "B"]["sum_zetas"] == pytest.approx(zeta_bogen / 2)


def test_grad2_unter_schwelle_ist_vermessungspunkt():
    g = _grad2_graph((10.0, 0.3))  # ~1.7° Ablenkung
    apply_fittings(g, logger=LOGGER)
    assert g.nodes["M"]["formteile"] == {}
    assert g.edges["M", "A"]["sum_zetas"] == 0.0


def test_grad2_dn_wechsel_gibt_querschnittssprung():
    g = _grad2_graph((10.0, 0.0))  # gerade, also kein Bogen
    g.edges["M", "B"]["diameter"] = 0.05
    apply_fittings(g, logger=LOGGER)
    assert g.nodes["M"]["formteile"] == {"querschnittssprung": 1}
    erwartet = zeta_querschnitt(0.05, 0.1)
    assert g.edges["M", "A"]["sum_zetas"] == pytest.approx(erwartet / 2)


def test_zeta_querschnitt_ist_null_ohne_sprung():
    assert zeta_querschnitt(0.1, 0.1) == pytest.approx(0.0)


# -- Attribut-Vertrag --------------------------------------------------------

def test_apply_fittings_setzt_vertrag(t_graph):
    t_graph.nodes["C"]["node_type"] = "building"
    apply_fittings(t_graph, logger=LOGGER)
    for u, v, data in t_graph.edges(data=True):
        assert isinstance(data["sum_zetas"], float)
        assert data["zeta_quelle"] == "auto"
        assert isinstance(data["zeta_herkunft"], list)
        for eintrag in data["zeta_herkunft"]:
            assert set(eintrag) == {"typ", "knoten", "anzahl", "zeta"}
    # Der Hausanschluss bringt den Schieber komplett auf seine eine Kante
    zeta_schieber = lade_katalog()["schieber"]["zeta"]
    herkunft = t_graph.edges["M", "C"]["zeta_herkunft"]
    schieber = [e for e in herkunft if e["typ"] == "schieber"]
    assert len(schieber) == 1
    assert schieber[0]["zeta"] == pytest.approx(zeta_schieber)


def test_herkunft_zeta_ist_je_stueck(t_graph):
    """Invariante des Attribut-Vertrags: sum_zetas == Σ anzahl * zeta."""
    t_graph.nodes["C"]["node_type"] = "building"
    apply_fittings(t_graph, logger=LOGGER)
    for _, _, data in t_graph.edges(data=True):
        assert sum(
            e["anzahl"] * e["zeta"] for e in data["zeta_herkunft"]
        ) == pytest.approx(data["sum_zetas"])


def test_herkunft_bei_anzahl_groesser_eins(t_graph, monkeypatch):
    """Zwei Schieber → anzahl=2 und zeta bleibt der Einzelwert."""
    import uesgraphs.fittings as fittings
    monkeypatch.setattr(fittings, "SCHIEBER_JE_HAUSANSCHLUSS", 2)
    t_graph.nodes["C"]["node_type"] = "building"
    fittings.apply_fittings(t_graph, logger=LOGGER)
    schieber = [e for e in t_graph.edges["M", "C"]["zeta_herkunft"]
                if e["typ"] == "schieber"]
    assert len(schieber) == 1
    assert schieber[0]["anzahl"] == 2
    assert schieber[0]["zeta"] == pytest.approx(KATALOG["schieber"]["zeta"])


def test_herkunft_ist_json_serialisierbar(t_graph):
    import json
    apply_fittings(t_graph, logger=LOGGER)
    for _, _, data in t_graph.edges(data=True):
        json.dumps(data["zeta_herkunft"])


def test_apply_fittings_ueberschreibt_nicht(t_graph):
    t_graph.edges["M", "A"]["sum_zetas"] = 42.0
    t_graph.edges["M", "A"]["zeta_quelle"] = "plan"
    apply_fittings(t_graph, logger=LOGGER)
    assert t_graph.edges["M", "A"]["sum_zetas"] == 42.0
    assert t_graph.edges["M", "A"]["zeta_quelle"] == "plan"


def test_apply_fittings_overwrite_true(t_graph):
    t_graph.edges["M", "A"]["sum_zetas"] = 42.0
    apply_fittings(t_graph, overwrite=True, logger=LOGGER)
    assert t_graph.edges["M", "A"]["sum_zetas"] != 42.0


# -- Grad 4: kleinste Quadrate -----------------------------------------------

def test_grad4_bleibt_nicht_negativ_und_ist_symmetrisch():
    nodes = [
        ("M", (0.0, 0.0), {"node_type": "network_heating"}),
        ("A", (-10.0, 0.0), {"node_type": "network_heating"}),
        ("B", (10.0, 0.0), {"node_type": "network_heating"}),
        ("C", (0.0, 10.0), {"node_type": "network_heating"}),
        ("D", (0.0, -10.0), {"node_type": "network_heating"}),
    ]
    edges = [("M", n, {"diameter": 0.1, "length": 10.0}) for n in "ABCD"]
    g = _graph(nodes, edges)
    apply_fittings(g, logger=LOGGER)
    werte = {n: g.edges["M", n]["sum_zetas"] for n in "ABCD"}
    assert all(w >= 0.0 for w in werte.values())
    assert werte["A"] == pytest.approx(werte["B"])
    assert werte["C"] == pytest.approx(werte["D"])


# -- Händisch erfasste Einbauteile -------------------------------------------

def test_manuell_schlaegt_auto(t_graph):
    """Erfasste Zählungen ersetzen die abgeleiteten — am selben Knoten."""
    auto = dict(t_graph.edges["M", "C"])
    apply_fittings(t_graph, logger=LOGGER)
    zeta_auto = t_graph.edges["M", "C"]["sum_zetas"]

    g2 = _graph(
        [("M", (0.0, 0.0), {"node_type": "network_heating"}),
         ("A", (-10.0, 0.0), {"node_type": "network_heating"}),
         ("B", (10.0, 0.0), {"node_type": "network_heating"}),
         ("C", (0.0, 10.0), {"node_type": "network_heating"})],
        [("M", "A", {"diameter": 0.1, "length": 10.0}),
         ("M", "B", {"diameter": 0.1, "length": 10.0}),
         ("M", "C", {"diameter": 0.1, "length": 10.0})],
    )
    manual = {"knoten": {"M": {"t_durchgang": 1, "t_abzweig": 1,
                              "schieber": 2, "quelle": "plan"}}}
    apply_fittings(g2, manual=manual, logger=LOGGER)

    assert g2.nodes["M"]["formteile_quelle"] == "plan"
    assert g2.nodes["M"]["formteile"]["schieber"] == 2
    # Die zwei erfassten Schieber liegen in JEDEM Weg durch den Knoten, also
    # muss der Gesamtverlust über dem rein abgeleiteten liegen.
    assert g2.edges["M", "C"]["sum_zetas"] > zeta_auto
    assert auto.get("sum_zetas") is None          # Fixture war unberührt


def test_manuell_ohne_formteile_setzt_knoten_auf_null(t_graph):
    """Alles auf 0 erfasst heißt: hier steht nichts — der T-Verlust entfällt."""
    manual = {"knoten": {"M": {"t_durchgang": 0, "t_abzweig": 0, "quelle": "plan"}}}
    apply_fittings(t_graph, manual=manual, logger=LOGGER)
    assert all(t_graph.edges["M", n]["sum_zetas"] == 0.0 for n in "ABC")


def test_manuelle_kante_kommt_obendrauf(t_graph):
    apply_fittings(t_graph, logger=LOGGER)
    basis = t_graph.edges["M", "A"]["sum_zetas"]

    g2 = _graph(
        [("M", (0.0, 0.0), {"node_type": "network_heating"}),
         ("A", (-10.0, 0.0), {"node_type": "network_heating"}),
         ("B", (10.0, 0.0), {"node_type": "network_heating"}),
         ("C", (0.0, 10.0), {"node_type": "network_heating"})],
        [("M", "A", {"diameter": 0.1, "length": 10.0}),
         ("M", "B", {"diameter": 0.1, "length": 10.0}),
         ("M", "C", {"diameter": 0.1, "length": 10.0})],
    )
    manual = {"kanten": {"M->A": {"kompensator": 2, "sonstiges_zeta": 0.4,
                                 "quelle": "geschaetzt"}}}
    apply_fittings(g2, manual=manual, logger=LOGGER)
    data = g2.edges["M", "A"]
    erwartet = basis + 2 * KATALOG["kompensator"]["zeta"] + 0.4
    assert data["sum_zetas"] == pytest.approx(erwartet)
    assert data["zeta_quelle"] == "geschaetzt"
    assert sum(e["anzahl"] * e["zeta"] for e in data["zeta_herkunft"]) == pytest.approx(
        data["sum_zetas"])


def test_speichern_laden_ist_roundtrip(tmp_path):
    from uesgraphs.fittings import load_manual, save_manual

    quelle = tmp_path / "network.geojson"
    quelle.write_text('{"type": "FeatureCollection"}', encoding="utf-8")
    daten = {"knoten": {"1042": {"schieber": 2, "bogen": 1, "quelle": "plan"}},
             "kanten": {"1042->1044": {"kompensator": 1, "sonstiges_zeta": 0.4,
                                       "quelle": "geschaetzt"}}}
    pfad = tmp_path / "fittings_manual.json"
    save_manual(pfad, daten, quellen={"network.geojson": quelle}, logger=LOGGER)

    zurueck = load_manual(pfad, quellen={"network.geojson": quelle}, logger=LOGGER)
    assert zurueck["knoten"] == daten["knoten"]
    assert zurueck["kanten"] == daten["kanten"]
    assert zurueck["warnung"] is None
    assert zurueck["quellen_hash"]["network.geojson"]


def test_hash_abweichung_warnt_laut(tmp_path, caplog):
    """Ändern sich die GeoJSONs, können die Knotennummern verrutschen — das
    muss beim Laden auffallen, nicht stillschweigend angewandt werden."""
    from uesgraphs.fittings import load_manual, save_manual

    quelle = tmp_path / "network.geojson"
    quelle.write_text('{"type": "FeatureCollection"}', encoding="utf-8")
    pfad = tmp_path / "fittings_manual.json"
    save_manual(pfad, {"knoten": {"1042": {"bogen": 1}}},
                quellen={"network.geojson": quelle}, logger=LOGGER)

    quelle.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        zurueck = load_manual(pfad, quellen={"network.geojson": quelle}, logger=LOGGER)

    assert zurueck["warnung"] is not None
    assert "network.geojson" in zurueck["warnung"]
    assert any("network.geojson" in r.message % r.args if r.args else
               "network.geojson" in r.message for r in caplog.records)
    assert zurueck["knoten"]                       # Daten kommen trotzdem mit


def test_load_manual_ohne_datei_ist_leer(tmp_path):
    from uesgraphs.fittings import load_manual

    leer = load_manual(tmp_path / "gibtsnicht.json", logger=LOGGER)
    assert leer == {"quellen_hash": {}, "knoten": {}, "kanten": {}, "warnung": None}


# -- Schieber am Rohr --------------------------------------------------------

def _t_graph_neu():
    return _graph(
        [("M", (0.0, 0.0), {"node_type": "network_heating"}),
         ("A", (-10.0, 0.0), {"node_type": "network_heating"}),
         ("B", (10.0, 0.0), {"node_type": "network_heating"}),
         ("C", (0.0, 10.0), {"node_type": "network_heating"})],
        [("M", "A", {"diameter": 0.1, "length": 10.0}),
         ("M", "B", {"diameter": 0.1, "length": 10.0}),
         ("M", "C", {"diameter": 0.1, "length": 10.0})],
    )


def test_schieber_an_der_kante_kommt_obendrauf(t_graph):
    """Ein Schieber im Rohr ist additiv zu dem, was aus den Knoten kommt."""
    apply_fittings(t_graph, logger=LOGGER)
    basis = t_graph.edges["M", "A"]["sum_zetas"]

    g2 = _t_graph_neu()
    manual = {"kanten": {"M->A": {"schieber": 2, "quelle": "plan"}}}
    apply_fittings(g2, manual=manual, logger=LOGGER)

    data = g2.edges["M", "A"]
    assert data["sum_zetas"] == pytest.approx(
        basis + 2 * KATALOG["schieber"]["zeta"])
    # eigener Eintrag in der Herkunft, erkennbar an knoten=None
    kanten_eintraege = [e for e in data["zeta_herkunft"]
                        if e["typ"] == "schieber" and e["knoten"] is None]
    assert kanten_eintraege == [{"typ": "schieber", "knoten": None,
                                 "anzahl": 2, "zeta": KATALOG["schieber"]["zeta"]}]
    # Vertrag
    assert sum(e["anzahl"] * e["zeta"] for e in data["zeta_herkunft"]) == \
        pytest.approx(data["sum_zetas"])


def test_schieber_am_knoten_bleibt_am_knoten(t_graph):
    """Der Schieber steht jetzt auch in KANTEN_TYPEN — am Knoten muss er
    trotzdem weiterhin ankommen."""
    manual = {"knoten": {"M": {"t_durchgang": 1, "t_abzweig": 1, "schieber": 3,
                               "quelle": "plan"}}}
    apply_fittings(t_graph, manual=manual, logger=LOGGER)
    assert t_graph.nodes["M"]["formteile"]["schieber"] == 3


def test_kantentyp_ohne_festes_zeta_wird_ignoriert(caplog):
    """``querschnittssprung`` hat kein festes ζ — an der Kante ist er ein
    Erfassungsfehler und soll auffallen, nicht still 0 beitragen."""
    g = _t_graph_neu()
    manual = {"kanten": {"M->A": {"querschnittssprung": 2, "quatsch": 1,
                                  "quelle": "plan"}}}
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        apply_fittings(g, manual=manual, logger=LOGGER)
    data = g.edges["M", "A"]
    assert all(e["typ"] not in ("querschnittssprung", "quatsch")
               for e in data["zeta_herkunft"])
    assert any("querschnittssprung" in str(r.msg) % r.args if r.args
               else "querschnittssprung" in str(r.msg) for r in caplog.records)


# -- Rohr-Eigenschaften ------------------------------------------------------

def test_eigenschaften_landen_als_kantenattribute():
    g = _t_graph_neu()
    manual = {"eigenschaften": {"M->A": {"year_built": 1987,
                                         "material": "steel"}}}
    apply_fittings(g, manual=manual, logger=LOGGER)
    data = g.edges["M", "A"]
    assert data["year_built"] == 1987
    assert data["material"] == "steel"
    assert data["eigenschaften"] == {"year_built": 1987, "material": "steel"}
    assert "year_built" not in g.edges["M", "B"]


def test_geloeschte_eigenschaft_verschwindet_vom_graphen():
    """Derselbe Graph wird beim Erfassen mehrfach durchgerechnet — ein
    entfernter Wert darf nicht als Attribut liegen bleiben."""
    g = _t_graph_neu()
    apply_fittings(g, manual={"eigenschaften": {"M->A": {"material": "steel"}}},
                   overwrite=True, logger=LOGGER)
    apply_fittings(g, manual={"eigenschaften": {}}, overwrite=True, logger=LOGGER)
    assert "material" not in g.edges["M", "A"]
    assert not g.edges["M", "A"].get("eigenschaften")


def test_optionale_abschnitte_sind_rueckwaertskompatibel(tmp_path):
    """Alte Dateien laden weiter, und Speichern erfindet keine Schlüssel."""
    import json

    from uesgraphs.fittings import load_manual, save_manual

    pfad = tmp_path / "fittings_manual.json"
    pfad.write_text(json.dumps({"quellen_hash": {}, "knoten": {},
                                "kanten": {"1->2": {"kompensator": 1}}}),
                    encoding="utf-8")
    alt = load_manual(pfad, logger=LOGGER)
    assert "eigenschaften" not in alt and "felder" not in alt

    save_manual(pfad, alt, logger=LOGGER)
    assert set(json.loads(pfad.read_text(encoding="utf-8"))) == {
        "quellen_hash", "knoten", "kanten"}


def test_eigenschaften_und_felder_sind_roundtrip(tmp_path):
    from uesgraphs.fittings import load_manual, save_manual

    daten = {"knoten": {}, "kanten": {},
             "eigenschaften": {"1->2": {"year_built": 1990,
                                        "material": "plastic"}},
             "felder": [{"name": "year_built", "type": "int"},
                        {"name": "material", "type": "choice",
                         "values": ["steel", "plastic"]}]}
    pfad = tmp_path / "fittings_manual.json"
    save_manual(pfad, daten, logger=LOGGER)
    zurueck = load_manual(pfad, logger=LOGGER)
    assert zurueck["eigenschaften"] == daten["eigenschaften"]
    assert zurueck["felder"] == daten["felder"]


def test_leere_felderliste_ueberlebt(tmp_path):
    """Alle Felder gelöscht ist eine Entscheidung — sie darf nicht dazu
    führen, dass die Voreinstellung wieder auftaucht."""
    from uesgraphs.fittings import load_manual, save_manual

    pfad = tmp_path / "fittings_manual.json"
    save_manual(pfad, {"knoten": {}, "kanten": {}, "felder": []}, logger=LOGGER)
    assert load_manual(pfad, logger=LOGGER)["felder"] == []
