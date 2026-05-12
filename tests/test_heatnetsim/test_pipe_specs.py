"""Tests for uesgraphs.heatnetsim.pipe_specs."""

import logging

import pytest

from uesgraphs.heatnetsim.pipe_specs import (
    _parse_dn,
    _read_pipe_specs,
    apply_pipe_specs_to_graph,
)

from tests.test_heatnetsim.conftest import FakeGraph


MINI_CATALOG = (
    "DN,wall_thickness,inner_diameter,d_ins\n"
    "DN50,0.0032,0.0539,0.0324\n"
    "DN80,0.0032,0.0825,0.0356\n"
    "DN125,0.0036,0.1325,0.0427\n"
    "DN200,0.0045,0.2101,0.0480\n"
)


@pytest.fixture
def mini_catalog_csv(tmp_path):
    path = tmp_path / "mini_catalog.csv"
    path.write_text(MINI_CATALOG)
    return path


@pytest.fixture
def dn_graph():
    """5-node graph where each edge carries a DN matching MINI_CATALOG entries."""
    nodes = [
        ("S",  {"node_type": "building", "is_supply_heating": True, "name": "ebu1"}),
        ("N1", {"node_type": "network_heating"}),
        ("N2", {"node_type": "network_heating"}),
        ("D1", {"node_type": "building", "name": "bldg_a"}),
        ("D2", {"node_type": "building", "name": "bldg_b"}),
    ]
    edges = [
        ("S",  "N1", {"DN": "DN200"}),                # nested in attr_dict elsewhere
        ("N1", "N2", {"DN": 125}),                    # bare int
        ("N1", "D1", {"attr_dict": {"DN": "80"}}),    # nested string
        ("N2", "D2", {"DN": "DN50"}),
    ]
    return FakeGraph(nodes, edges)


def test_parse_dn_normalises_variants():
    assert _parse_dn("DN200") == 200
    assert _parse_dn("200") == 200
    assert _parse_dn(200) == 200
    assert _parse_dn(" DN 200 ") == 200
    assert _parse_dn(200.0) == 200


def test_parse_dn_returns_none_for_garbage():
    assert _parse_dn(None) is None
    assert _parse_dn("foo") is None
    assert _parse_dn(float("nan")) is None


def test_apply_sets_missing_attrs(dn_graph, mini_catalog_csv, caplog):
    with caplog.at_level(logging.INFO):
        apply_pipe_specs_to_graph(
            dn_graph, mini_catalog_csv,
            logger=logging.getLogger("test_apply"),
        )

    # DN200 edge gets 0.2101 diameter and 0.0480 d_ins
    e = dn_graph.edges[("S", "N1")]
    assert e["diameter"] == pytest.approx(0.2101)
    assert e["dIns"] == pytest.approx(0.0480)
    assert e["wall_thickness"] == pytest.approx(0.0045)

    # DN80 edge (DN nested in attr_dict)
    e = dn_graph.edges[("N1", "D1")]
    assert e["diameter"] == pytest.approx(0.0825)
    assert e["dIns"] == pytest.approx(0.0356)


def test_apply_preserves_existing_by_default(dn_graph, mini_catalog_csv):
    dn_graph.edges[("S", "N1")]["diameter"] = 0.9  # pre-existing, not from catalog

    apply_pipe_specs_to_graph(
        dn_graph, mini_catalog_csv,
        logger=logging.getLogger("test_preserve"),
    )

    assert dn_graph.edges[("S", "N1")]["diameter"] == 0.9
    # other attrs that weren't pre-set should still be filled
    assert dn_graph.edges[("S", "N1")]["dIns"] == pytest.approx(0.0480)


def test_apply_overwrite_replaces(dn_graph, mini_catalog_csv):
    dn_graph.edges[("S", "N1")]["diameter"] = 0.9

    apply_pipe_specs_to_graph(
        dn_graph, mini_catalog_csv, overwrite=True,
        logger=logging.getLogger("test_overwrite"),
    )

    assert dn_graph.edges[("S", "N1")]["diameter"] == pytest.approx(0.2101)


def test_apply_skips_unknown_dn(dn_graph, mini_catalog_csv, caplog):
    dn_graph.edges[("S", "N1")]["DN"] = "DN999"  # not in catalog

    with caplog.at_level(logging.WARNING):
        apply_pipe_specs_to_graph(
            dn_graph, mini_catalog_csv,
            logger=logging.getLogger("test_unknown"),
        )

    # unknown DN edge gets nothing
    assert "diameter" not in dn_graph.edges[("S", "N1")]
    # other edges still resolved
    assert dn_graph.edges[("N1", "N2")]["diameter"] == pytest.approx(0.1325)
    assert any("DN=999" in rec.getMessage() for rec in caplog.records)


def test_apply_skips_edges_without_dn(mini_catalog_csv):
    nodes = [
        ("S", {"node_type": "building", "is_supply_heating": True}),
        ("D1", {"node_type": "building"}),
    ]
    edges = [("S", "D1", {})]  # no DN at all
    graph = FakeGraph(nodes, edges)

    apply_pipe_specs_to_graph(
        graph, mini_catalog_csv,
        logger=logging.getLogger("test_no_dn"),
    )

    assert "diameter" not in graph.edges[("S", "D1")]


def test_read_specs_drops_unparseable_dn(tmp_path, caplog):
    csv = tmp_path / "broken.csv"
    csv.write_text(
        "DN,inner_diameter\n"
        "DN50,0.05\n"
        "foo,0.10\n"
        "DN100,0.10\n"
    )

    with caplog.at_level(logging.WARNING):
        df = _read_pipe_specs(csv, dn_column="DN", logger=logging.getLogger("t"))

    assert sorted(df.index.tolist()) == [50, 100]
    assert any("unparseable DN" in rec.getMessage() for rec in caplog.records)


def test_read_specs_first_wins_on_duplicate_dn(tmp_path, caplog):
    csv = tmp_path / "dup.csv"
    csv.write_text(
        "DN,inner_diameter\n"
        "DN50,0.05\n"
        "DN50,0.99\n"
    )

    with caplog.at_level(logging.WARNING):
        df = _read_pipe_specs(csv, dn_column="DN", logger=logging.getLogger("t"))

    assert df.loc[50, "inner_diameter"] == 0.05
    assert any("duplicate DN" in rec.getMessage() for rec in caplog.records)


def test_apply_warns_and_returns_when_no_usable_columns(dn_graph, tmp_path, caplog):
    """Catalog with only DN column - nothing to map."""
    csv = tmp_path / "only_dn.csv"
    csv.write_text("DN\nDN50\nDN200\n")

    with caplog.at_level(logging.WARNING):
        result = apply_pipe_specs_to_graph(
            dn_graph, csv,
            logger=logging.getLogger("test_empty_mapping"),
        )

    assert result is dn_graph
    assert any("No usable columns" in rec.getMessage() for rec in caplog.records)
