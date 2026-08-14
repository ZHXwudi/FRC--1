"""Structured provenance for the research-alignment view."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_research_catalog(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_research_catalog(catalog: dict[str, Any], root: str | Path) -> list[str]:
    """Return provenance problems without mutating the catalog."""
    root_path = Path(root)
    errors: list[str] = []
    papers = catalog.get("papers", [])
    dois = [paper.get("doi", "").lower() for paper in papers]
    ids = [paper.get("id", "") for paper in papers]

    if not papers:
        errors.append("research catalog has no papers")
    if len(dois) != len(set(dois)):
        errors.append("paper DOI values must be unique")
    if len(ids) != len(set(ids)):
        errors.append("paper IDs must be unique")

    required = {
        "id",
        "title_en",
        "title_zh",
        "doi",
        "journal",
        "year",
        "author_role",
        "evidence_level",
        "verified_claim",
        "project_relation",
        "must_not_claim",
        "license",
        "figure_reused",
    }
    for paper in papers:
        missing = sorted(required - paper.keys())
        if missing:
            errors.append(f"{paper.get('id', '<unknown>')} missing: {', '.join(missing)}")
        if paper.get("evidence_level") != "A":
            errors.append(f"{paper.get('id', '<unknown>')} is not primary-evidence level A")
        if not str(paper.get("doi", "")).startswith("10."):
            errors.append(f"{paper.get('id', '<unknown>')} has an invalid DOI")

    paper_ids = set(ids)
    asset_paper_ids: set[str] = set()
    for asset in catalog.get("assets", []):
        if asset.get("paper_id") not in paper_ids:
            errors.append(f"{asset.get('id', '<unknown>')} references an unknown paper")
        asset_paper_ids.add(asset.get("paper_id", ""))
        if not str(asset.get("source_url", "")).startswith("https://"):
            errors.append(f"{asset.get('id', '<unknown>')} lacks an HTTPS source URL")
        if asset.get("modified") is not False:
            errors.append(f"{asset.get('id', '<unknown>')} must declare its modification state")
        asset_path = root_path / asset.get("path", "")
        if not asset_path.is_file():
            errors.append(f"missing asset: {asset_path}")
            continue
        digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        if digest != asset.get("sha256"):
            errors.append(f"SHA-256 mismatch: {asset_path}")
        if "CC BY" not in asset.get("license", ""):
            errors.append(f"asset lacks a reusable license: {asset_path}")

    reused_paper_ids = {paper["id"] for paper in papers if paper.get("figure_reused")}
    if reused_paper_ids != asset_paper_ids:
        errors.append("reused-paper flags and attributed assets must match")

    return errors
