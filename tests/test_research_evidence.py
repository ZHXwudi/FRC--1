from pathlib import Path

from frc_lab.research import load_research_catalog, validate_research_catalog


ROOT = Path(__file__).resolve().parents[1]


def test_research_catalog_has_valid_provenance() -> None:
    catalog = load_research_catalog(ROOT / "data" / "research_evidence.json")
    assert validate_research_catalog(catalog, ROOT) == []


def test_fast_equilibrium_paper_journal_is_aip_advances() -> None:
    catalog = load_research_catalog(ROOT / "data" / "research_evidence.json")
    paper = next(
        item for item in catalog["papers"] if item["doi"] == "10.1063/5.0152318"
    )
    assert paper["journal"] == "AIP Advances"
    assert paper["author_role"] == "共同作者（第 3/4 作者）"


def test_only_licensed_assets_are_reused() -> None:
    catalog = load_research_catalog(ROOT / "data" / "research_evidence.json")
    reused = [paper for paper in catalog["papers"] if paper["figure_reused"]]
    assert {paper["id"] for paper in reused} == {"fydev_fair4rs"}
    assert all("CC BY" in paper["license"] for paper in reused)
