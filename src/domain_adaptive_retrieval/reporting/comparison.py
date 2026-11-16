"""Human-readable per-domain evaluation summaries."""

from pathlib import Path


def make_comparison(
    metrics: dict[str, dict[str, float]], *, protocol: str, capped: bool
) -> str:
    """Render one protocol-aware markdown table, without cross-domain aggregation."""
    columns = sorted({column for row in metrics.values() for column in row})
    lines = [
        "# Held-out retrieval comparison",
        "",
        f"Protocol: `{protocol}`",
        "",
        f"Coverage: **{'subset/capped run' if capped else 'all available held-out queries'}**",
        "",
        "| System | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for name, row in sorted(metrics.items()):
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(f"{row[k]:.4f}" if k in row else "n/a" for k in columns)
            + " |"
        )
    lines.extend(
        [
            "",
            "Results use known relevance judgments. Unjudged documents are treated as zero gain; they are not proven irrelevant.",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison(
    path: Path, metrics: dict[str, dict[str, float]], *, protocol: str, capped: bool
) -> None:
    """Save the benchmark as an easy-to-read Markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        make_comparison(metrics, protocol=protocol, capped=capped), encoding="utf-8"
    )
