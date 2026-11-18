"""Generate a small public results bundle and separate local model release copies."""

import csv
import io
import os
import shutil
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.reporting.artifacts import write_json, write_text
from domain_adaptive_retrieval.reporting.evidence import (
    DOMAINS,
    PRIMARY,
    SCOPE,
    TITLES,
    EvaluationRun,
    read_json,
    scan_runs,
    sha256_file,
)
from domain_adaptive_retrieval.reporting.publication import (
    LABELS,
    domain_report,
    evaluated_model_card,
    readme_section,
)

START = "<!-- BEGIN MODEL LINKS -->"
END = "<!-- END MODEL LINKS -->"
LEGACY_START = "<!-- BEGIN MEASURED RESULTS -->"
LEGACY_END = "<!-- END MEASURED RESULTS -->"


def plot_domain(path: Path, domain: str, runs: list[EvaluationRun]) -> None:
    """Save component comparisons on a fixed 0–1 scale with original variant labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    evaluated = [run for run in runs if run.status == "evaluated"]
    if not evaluated:
        return
    with plt.rc_context({"svg.hashsalt": "domain-adaptive-retrieval", "font.size": 10}):
        fig, axes = plt.subplots(
            1, len(evaluated), figsize=(5 * len(evaluated), 3.8), squeeze=False
        )
        for ax, run in zip(axes[0], evaluated, strict=True):
            metrics = PRIMARY[domain]
            positions = np.arange(len(metrics))
            for shift, label, scores, color in (
                (-0.18, "Base", run.base, "#315b8a"),
                (0.18, "Tuned", run.tuned, "#c76a24"),
            ):
                bars = ax.bar(
                    positions + shift,
                    [scores[m] for m in metrics],
                    0.34,
                    label=label,
                    color=color,
                )
                ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
            ax.set_xticks(positions, [LABELS[m] for m in metrics])
            ax.set_ylim(0, 1.08)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            scope = run.protocol.replace("_", " ")
            if run.protocol == "held_out_corpus":
                scope = f"held-out pool: {run.split_queries['test']:,} documents"
            ax.set_title(
                f"{TITLES[domain]} {run.model_type} · {run.query_count:,} queries\n{scope}"
            )
            ax.set_ylabel("Score (higher is better)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(
                loc="upper left", bbox_to_anchor=(0, -0.14), ncols=2, frameon=False
            )
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path.with_suffix(".svg"), metadata={"Date": None}, bbox_inches="tight"
        )
        fig.savefig(
            path.with_suffix(".png"),
            dpi=160,
            metadata={"Software": "Matplotlib"},
            bbox_inches="tight",
        )
        plt.close(fig)


def _csv(runs: list[EvaluationRun]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "run",
            "variant",
            "metric",
            "value",
            "tuned_minus_base",
            "delta_ci_low",
            "delta_ci_high",
            "test_queries",
            "protocol",
        ]
    )
    for run in runs:
        for variant, scores in run.systems.items():
            for metric, value in sorted(scores.items()):
                paired = variant == run.tuned_label
                delta = value - run.base[metric] if paired else ""
                ci = run.bootstrap.get(metric, {}).get(
                    "paired_bootstrap_95ci", ["", ""]
                )
                writer.writerow(
                    [
                        run.id,
                        variant,
                        metric,
                        value,
                        delta,
                        ci[0] if paired else "",
                        ci[1] if paired else "",
                        run.query_count,
                        run.protocol,
                    ]
                )
    return stream.getvalue()


def _relative(target: Path, parent: Path) -> str:
    return Path(os.path.relpath(target, parent)).as_posix()


def update_readme(path: Path, section: str) -> None:
    """Replace model links and migrate old score blocks, preserving other edits."""
    content = path.read_text(encoding="utf-8")
    for start, end in ((START, END), (LEGACY_START, LEGACY_END)):
        if (
            content.count(start) != content.count(end)
            or content.count(start) > 1
            or (start in content and content.index(end) < content.index(start))
        ):
            raise ValueError("README contains unmatched or duplicate model markers")
    if LEGACY_START in content:
        before, remaining = content.split(LEGACY_START, 1)
        _, after = remaining.split(LEGACY_END, 1)
        replacement = "" if START in content else f"{START}\n{END}"
        content = before + replacement + after
    block = f"{START}\n{section.rstrip()}\n{END}"
    if START in content:
        before, remaining = content.split(START, 1)
        _, after = remaining.split(END, 1)
        content = before + block + after
    else:
        first_section = content.find("\n## ")
        if first_section < 0:
            content = content.rstrip() + "\n\n" + block + "\n"
        else:
            content = (
                content[:first_section].rstrip()
                + "\n\n"
                + block
                + "\n"
                + content[first_section:]
            )
    write_text(path, content)


def package_model(source: Path, target: Path, run: EvaluationRun) -> None:
    """Copy the selected model, never modifying the evidence used for evaluation."""
    if target.exists():
        marker = target / "release_manifest.json"
        if (
            not marker.is_file()
            or read_json(marker).get("format") != "dar-local-release-v1"
        ):
            raise FileExistsError(
                f"Refusing to replace an unrelated directory: {target}"
            )
        shutil.rmtree(target)
    shutil.copytree(
        source / "model",
        target,
        ignore=shutil.ignore_patterns("README.md", "*:Zone.Identifier"),
    )
    write_text(target / "README.md", evaluated_model_card(run, packaged=True))
    write_json(target / "evaluation/run.json", run.model_dump(mode="json"))
    for evidence in run.evidence:
        if evidence.path.startswith("model/"):
            continue
        destination = target / "evaluation/source" / evidence.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / evidence.path, destination)
    plot_domain(target / "evaluation" / run.domain, run.domain, [run])
    manifest: dict[str, Any] = {
        "format": "dar-local-release-v1",
        "repo_id": run.repo_id,
        "status": "prepared_locally",
        "source_run": run.source,
        "source_checkpoint_fingerprint": run.checkpoint_fingerprint,
        "files": {
            path.relative_to(target).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(target.rglob("*"))
            if path.is_file()
        },
    }
    write_json(target / "release_manifest.json", manifest)


def _collection(runs: list[EvaluationRun], namespace: str) -> dict[str, Any]:
    available = [run for run in runs if run.status == "evaluated"]
    available.sort(
        key=lambda run: (
            run.id != "legal/embedding",
            run.model_type != "embedding",
            run.domain,
        )
    )
    return {
        "title": "Domain Adaptive Retrieval",
        "owner": namespace,
        "status": "draft_not_published",
        "description": f"Five domains, {len(available)} evaluated research checkpoints. "
        "Every card includes its baseline, evaluation scope and measured tradeoffs.",
        "items": [
            {"repo_id": run.repo_id, "type": "model", "note": outcome_note(run)}
            for run in available
        ],
    }


def outcome_note(run: EvaluationRun) -> str:
    metric = PRIMARY[run.domain][0]
    delta = run.tuned[metric] - run.base[metric]
    return f"{LABELS[metric]} {run.tuned[metric]:.4f} (base {run.base[metric]:.4f}; Δ {delta:+.4f}). {SCOPE[run.domain]}"


def build_suite(
    *,
    root: Path,
    output: Path,
    cards_output: Path,
    package_output: Path | None = None,
    namespace: str = "nehal-vaghasiya",
    readme: Path | None = None,
    featured_model: str = "legal/embedding",
) -> list[EvaluationRun]:
    """Feature one evaluated model while retaining all Hugging Face release assets."""
    destinations = [output, cards_output]
    if package_output is not None:
        destinations.append(package_output)
    if readme is not None:
        destinations.append(readme)
    for destination in destinations:
        if destination.resolve().is_relative_to(
            root.resolve()
        ) or root.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Report destinations must be separate from source outputs")
    for first in (output, cards_output):
        if package_output is not None and (
            first.resolve().is_relative_to(package_output.resolve())
            or package_output.resolve().is_relative_to(first.resolve())
        ):
            raise ValueError("Documentation and package destinations must be separate")
    runs = scan_runs(root, namespace)
    invalid = [run.id for run in runs if run.status == "invalid"]
    if invalid:
        raise ValueError("Invalid evidence: " + ", ".join(invalid))
    featured = next((run for run in runs if run.id == featured_model), None)
    if featured is None or featured.status != "evaluated":
        raise ValueError(
            f"Featured model has no validated evaluation: {featured_model}"
        )
    public_runs = [
        featured.model_copy(
            update={
                "systems": {
                    featured.base_label: featured.base,
                    featured.tuned_label: featured.tuned,
                }
            }
        )
    ]
    output.mkdir(parents=True, exist_ok=True)
    cards_output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "results.json",
        {
            "schema_version": 1,
            "featured_model": featured_model,
            "runs": [run.model_dump(mode="json") for run in public_runs],
        },
    )
    write_text(output / "metrics.csv", _csv(public_runs))
    for domain in DOMAINS:
        if domain != featured.domain:
            for stale in (
                output / f"{domain}.md",
                output / "assets" / f"{domain}.svg",
                output / "assets" / f"{domain}.png",
            ):
                stale.unlink(missing_ok=True)
    plot_domain(output / "assets" / featured.domain, featured.domain, public_runs)
    report = domain_report(featured.domain, public_runs).replace(
        "../model-cards/", _relative(cards_output, output) + "/"
    )
    write_text(output / f"{featured.domain}.md", report)
    available = [run for run in runs if run.status == "evaluated"]
    index = (
        "\n\n".join(
            [
                f"# {TITLES[featured.domain]} {featured.model_type} evaluation",
                f"[Protocol, metrics and chart]({featured.domain}.md) · "
                f"[Hugging Face model card]({_relative(cards_output, output)}/"
                f"{featured.domain}-{featured.model_type}.md)",
                SCOPE[featured.domain],
                "[Full-precision data and source digests](results.json) · "
                "[Metrics CSV](metrics.csv)",
            ]
        )
        + "\n"
    )
    write_text(output / "README.md", index)
    card_index = [
        "# Hugging Face model cards",
        "Prepared locally for "
        + inline_namespace(namespace)
        + ". Model IDs are planned; repositories and collection have not been published.",
        "| Model | Prepared model ID |",
        "|---|---|",
    ]
    for run in available:
        card = evaluated_model_card(
            run, public_report=run.id == featured_model
        ).replace(
            f"../results/{run.domain}.md",
            _relative(output / f"{run.domain}.md", cards_output),
        )
        write_text(cards_output / f"{run.domain}-{run.model_type}.md", card)
        card_index.append(
            f"| [{TITLES[run.domain]} {run.model_type}]({run.domain}-{run.model_type}.md) | "
            f"{inline_namespace(run.repo_id)} |"
        )
        if package_output is not None:
            package_model(
                root / run.domain / run.model_type,
                package_output / run.repo_id.split("/")[-1],
                run,
            )
    card_index.extend(
        [
            "",
            "[Collection draft](collection.json). Each local package contains inference "
            "files, a model card, evaluation evidence and checksums.",
            "",
            "Reproduce a local package with the report command's "
            "`--package-output dist/huggingface` option. Run the card's inference example "
            "from inside its package directory. Source dataset/model revisions were not pinned.",
        ]
    )
    write_text(cards_output / "README.md", "\n".join(card_index) + "\n")
    write_json(cards_output / "collection.json", _collection(runs, namespace))
    if readme is not None:
        section = readme_section(public_runs, _relative(output, readme.parent))
        section = section.replace(
            _relative(output, readme.parent) + "/../model-cards/",
            _relative(cards_output, readme.parent) + "/",
        )
        update_readme(readme, section)
    return runs


def inline_namespace(namespace: str) -> str:
    return "`" + namespace + "`"
