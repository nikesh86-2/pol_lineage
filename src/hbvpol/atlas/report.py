"""Self-contained HTML report for the residue atlas.

Deliberately dependency-free: a plain string template plus :mod:`html` escaping
is enough, so the report renders even in a bare environment.  When
``atlas.interactive`` is set a py3Dmol viewer stub is embedded (the JS is pulled
from a CDN *in the browser*; nothing is fetched at build time).
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from ..config import get
from ..pipeline import get_logger

__all__ = ["build_report", "render_table"]

logger = get_logger("atlas.report")

_MAX_TABLE_ROWS = 100


def render_table(frame: pd.DataFrame, max_rows: int = _MAX_TABLE_ROWS) -> str:
    """Render a DataFrame as a plain HTML table (escaped, truncated)."""
    if frame is None or len(frame) == 0:
        return "<p class='empty'>No rows.</p>"
    frame = frame.head(max_rows)
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<table><thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def _summary_counts(atlas: pd.DataFrame, targets: pd.DataFrame) -> list[tuple[str, object]]:
    counts: list[tuple[str, object]] = [
        ("Atlas rows", len(atlas) if atlas is not None else 0),
        ("Ranked targets", len(targets) if targets is not None else 0),
    ]
    if atlas is not None and "lineage" in atlas.columns:
        counts.append(("Lineages", int(atlas["lineage"].nunique())))
    if atlas is not None and "passes_criteria" in atlas.columns:
        counts.append(("Mechanistic candidates", int(atlas["passes_criteria"].fillna(False).astype(bool).sum())))
    return counts


def build_report(atlas: pd.DataFrame, targets: pd.DataFrame, config: dict, out_html) -> Path:
    """Write a self-contained HTML report and return its path."""
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    counts = _summary_counts(atlas, targets)
    summary_items = "".join(
        f"<li><span class='k'>{html.escape(name)}</span><span class='v'>{html.escape(str(value))}</span></li>"
        for name, value in counts
    )

    target_cols = ["rank", "lineage", "pol_position", "score", "domain", "pol_aa"]
    target_view = (
        targets[[c for c in target_cols if c in targets.columns]]
        if targets is not None and len(targets)
        else targets
    )
    atlas_view = atlas.head(_MAX_TABLE_ROWS) if atlas is not None else pd.DataFrame()

    if bool(get(config, "atlas.interactive", False)):
        viewer = (
            "<h2>Interactive structure</h2>"
            "<div id='viewer' style='width:100%;height:420px;position:relative;'></div>"
            "<script src='https://3dmol.org/build/3Dmol-min.js'></script>\n"
            "<script>if (window.$3Dmol) { var v=$3Dmol.createViewer('viewer',{backgroundColor:'white'}); }</script>"
        )
    else:
        viewer = ""

    title = html.escape(str(get(config, "project.name", "hbv-pol-lineage")))
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — residue atlas</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
  h1, h2 {{ font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #8884; padding: 4px 8px; text-align: left; }}
  th {{ background: #8881; }}
  ul.summary {{ list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 1rem; }}
  ul.summary li {{ border: 1px solid #8884; border-radius: 6px; padding: 0.5rem 0.9rem; }}
  ul.summary .k {{ display: block; font-size: 0.75rem; opacity: 0.7; }}
  ul.summary .v {{ font-size: 1.4rem; font-weight: 600; }}
  .empty {{ opacity: 0.6; }}
</style>
</head>
<body>
<h1>{title} — residue atlas</h1>
<ul class="summary">{summary_items}</ul>
<h2>Ranked validation targets</h2>
{render_table(target_view)}
<h2>Atlas (first {_MAX_TABLE_ROWS} rows)</h2>
{render_table(atlas_view, max_rows=len(atlas_view) if atlas_view is not None else 0)}
{viewer}
</body>
</html>
"""
    out.write_text(document, encoding="utf-8")
    logger.info("wrote atlas report to %s", out)
    return out
