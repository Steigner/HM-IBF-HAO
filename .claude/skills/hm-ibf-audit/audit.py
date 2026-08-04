"""Classify whether a source tree implements a heterogeneous-migration island framework.

The auditor scans a repository for the structural traits that separate an HM-IBF
(heterogeneous migration island-based framework) from an ordinary island / multi-population
metaheuristic, and reports a verdict per trait together with file and line evidence.

The discriminating trait is *heterogeneous migration*: neighbouring islands may work at
different dimensions, so a migrant is re-expressed by a transform along a shared geometric
backbone instead of being copied index-to-index.

The traits themselves live in :mod:`criteria`; this module is only the scanning engine.
Run ``python3 audit.py --help`` for usage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from criteria import CRITERIA, DEVIATIONS
from model import ABSENT, PARTIAL, PRESENT, CriterionResult, Deviation, Hit, Report, SignalResult

#: File extensions worth scanning for structural signals.
SOURCE_SUFFIXES = frozenset(
    {".rs", ".py", ".m", ".jl", ".cpp", ".cc", ".hpp", ".h", ".java", ".go", ".ts", ".md"}
)

#: Documentation suffixes. A prose description of a trait is corroboration, never proof, so
#: hits in these files can never on their own satisfy a signal.
DOC_SUFFIXES = frozenset({".md"})

#: Directory names never scanned: build output, vendored code and benchmark payloads.
#:
#: ``.claude`` is skipped because it holds agent tooling - including this auditor, whose own
#: pattern strings would otherwise match every signal it looks for.
SKIPPED_DIRECTORIES = frozenset(
    {
        ".claude",
        ".git",
        ".direnv",
        ".ruff_cache",
        ".pytest_cache",
        "target",
        "build",
        "dist",
        "node_modules",
        "external",
        "instances",
        "result",
        "venv",
        ".venv",
    }
)

#: Maximum evidence lines retained per signal, counted separately for code and documentation.
MAX_EVIDENCE = 3

#: Prefixes marking a whole-line comment. Deviation probes ignore such lines so that a note
#: explaining an anti-pattern is not mistaken for the anti-pattern itself.
COMMENT_PREFIXES = ("//", "#", "*", "/*", "%", "--")


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield every scannable source file below `root`.

    Args:
        root: Directory to walk.

    Yields:
        Paths of files whose suffix is in :data:`SOURCE_SUFFIXES` and that live outside
        :data:`SKIPPED_DIRECTORIES`.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if SKIPPED_DIRECTORIES.intersection(path.relative_to(root).parts[:-1]):
            continue
        yield path


def scan(root: Path) -> tuple[dict[str, list[Hit]], dict[str, list[Hit]]]:
    """Search every configured pattern across the source tree.

    Args:
        root: Repository root to audit.

    Returns:
        Two mappings from pattern string to evidence. The first covers every line and keeps
        code and documentation hits capped independently. The second holds only executable
        lines - no documentation files and no whole-line comments - and drives the deviation
        probes, so prose describing an anti-pattern cannot trigger it.
    """
    patterns = {signal.pattern for criterion in CRITERIA for signal in criterion.signals}
    patterns.update(deviation.pattern for deviation in DEVIATIONS)
    compiled = {pattern: re.compile(pattern) for pattern in patterns}
    any_line: dict[str, list[Hit]] = {pattern: [] for pattern in patterns}
    code_only: dict[str, list[Hit]] = {pattern: [] for pattern in patterns}

    for path in iter_source_files(root):
        relative = path.relative_to(root).as_posix()
        is_doc = path.suffix in DOC_SUFFIXES
        content = path.read_text(encoding="utf-8", errors="replace")
        for number, text in enumerate(content.splitlines(), start=1):
            stripped = text.strip()
            is_comment = stripped.startswith(COMMENT_PREFIXES)
            for pattern, regex in compiled.items():
                if not regex.search(text):
                    continue
                hit = Hit(relative, number, stripped[:110], is_doc)
                # Cap code and documentation evidence separately, so that alphabetically
                # early docs cannot crowd out the code that actually proves the trait.
                same_kind = sum(1 for other in any_line[pattern] if other.is_doc == is_doc)
                if same_kind < MAX_EVIDENCE:
                    any_line[pattern].append(hit)
                if not is_doc and not is_comment and len(code_only[pattern]) < MAX_EVIDENCE:
                    code_only[pattern].append(hit)
    return any_line, code_only


def evaluate(found: dict[str, list[Hit]]) -> list[CriterionResult]:
    """Turn raw pattern hits into per-criterion verdicts.

    Args:
        found: The all-lines mapping produced by :func:`scan`.

    Returns:
        One result per criterion, in declaration order.
    """
    return [
        CriterionResult(
            criterion=criterion,
            signals=[
                SignalResult(name=signal.name, hits=found[signal.pattern])
                for signal in criterion.signals
            ],
        )
        for criterion in CRITERIA
    ]


def collect_deviations(found: dict[str, list[Hit]]) -> list[tuple[Deviation, list[Hit]]]:
    """Select the deviations whose trigger condition holds.

    Args:
        found: The code-only mapping produced by :func:`scan`; documentation and comments are
            excluded so that prose describing an anti-pattern does not trigger it.

    Returns:
        Pairs of triggered deviation and its supporting evidence (empty for absence triggers).
    """
    triggered = []
    for deviation in DEVIATIONS:
        hits = found[deviation.pattern]
        expects_hits = deviation.trigger == "present"
        if expects_hits == bool(hits):
            triggered.append((deviation, hits if expects_hits else []))
    return triggered


def classify(results: list[CriterionResult]) -> tuple[str, str]:
    """Derive the overall classification from the per-criterion verdicts.

    Args:
        results: The per-criterion verdicts.

    Returns:
        A ``(label, rationale)`` pair.
    """
    verdict = {result.criterion.key: result.verdict for result in results}
    core = ("topology", "heterogeneous_migration", "backbone", "dimension_agnostic")
    design = ("transform_catalog", "two_level_design", "freeze_inference")

    if verdict["topology"] == ABSENT:
        return "NOT AN ISLAND MODEL", "No island graph was found; the HM-IBF traits do not apply."
    if verdict["heterogeneous_migration"] == ABSENT or verdict["backbone"] == ABSENT:
        return (
            "HOMOGENEOUS ISLAND MODEL - NOT HM-IBF",
            "Islands and migration exist, but migrants are not re-expressed across dimensions. "
            "This is the homogeneous special case (tau = identity).",
        )
    if all(verdict[key] == PRESENT for key in core + design):
        return "HM-IBF - FULL", "Every structural trait of the framework is present."
    if all(verdict[key] == PRESENT for key in core):
        weak = [key for key in design if verdict[key] != PRESENT]
        return (
            "HM-IBF - CORE COMPLETE",
            "Heterogeneous migration is fully implemented; the design loop is incomplete: "
            f"{', '.join(weak)}.",
        )
    weak = [key for key in core if verdict[key] != PRESENT]
    return (
        "PARTIAL HM-IBF",
        f"Heterogeneous migration is started but unfinished: {', '.join(weak)}.",
    )


def audit(root: Path) -> Report:
    """Audit a repository against every HM-IBF criterion.

    Args:
        root: Repository root to audit.

    Returns:
        The complete audit outcome.
    """
    any_line, code_only = scan(root)
    results = evaluate(any_line)
    deviations = collect_deviations(code_only)
    label, rationale = classify(results)
    return Report(results=results, deviations=deviations, label=label, rationale=rationale)


def render(report: Report) -> str:
    """Render the human-readable report.

    Args:
        report: The audit outcome.

    Returns:
        The formatted report.
    """
    mark = {PRESENT: "[x]", PARTIAL: "[~]", ABSENT: "[ ]"}
    lines = ["", "=" * 78, f"VERDICT: {report.label}", "=" * 78, report.rationale, ""]

    for result in report.results:
        criterion = result.criterion
        lines.append(
            f"{mark[result.verdict]} {criterion.title}"
            f"   ({result.found_count}/{len(criterion.signals)} signals,"
            f" {criterion.min_strong} needed)"
        )
        lines.append(f"      {criterion.question}")
        for signal in result.signals:
            glyph = "+" if signal.found else ("?" if signal.documented_only else "-")
            suffix = "   (documented, not implemented)" if signal.documented_only else ""
            lines.append(f"      {glyph} {signal.name}{suffix}")
            hit = signal.evidence
            if hit is not None:
                lines.append(f"          {hit.path}:{hit.line}: {hit.text}")
        lines.append("")

    lines.extend(("-" * 78, "DEVIATIONS AND REVIEW POINTERS", "-" * 78))
    if not report.deviations:
        lines.append("None.")
    for deviation, hits in report.deviations:
        lines.append(f"[{deviation.severity}] {deviation.key}")
        lines.append(f"    {deviation.message}")
        lines.extend(f"      {hit.path}:{hit.line}: {hit.text}" for hit in hits)
        lines.append("")
    return "\n".join(lines)


def build_payload(report: Report) -> dict:
    """Assemble the machine-readable report.

    Args:
        report: The audit outcome.

    Returns:
        A JSON-serialisable dictionary.
    """
    return {
        "verdict": report.label,
        "rationale": report.rationale,
        "criteria": [
            {
                "key": result.criterion.key,
                "title": result.criterion.title,
                "verdict": result.verdict,
                "signals_found": result.found_count,
                "signals_total": len(result.criterion.signals),
                "signals": [
                    {
                        "name": signal.name,
                        "found": signal.found,
                        "documented_only": signal.documented_only,
                        "evidence": [
                            {"path": hit.path, "line": hit.line, "text": hit.text}
                            for hit in signal.hits
                        ],
                    }
                    for signal in result.signals
                ],
            }
            for result in report.results
        ],
        "deviations": [
            {
                "key": deviation.key,
                "severity": deviation.severity,
                "message": deviation.message,
                "evidence": [{"path": hit.path, "line": hit.line} for hit in hits],
            }
            for deviation, hits in report.deviations
        ],
    }


def self_test() -> int:
    """Check the classifier against synthetic trees with known verdicts.

    Returns:
        Process exit code: 0 when every expectation holds, 1 otherwise.
    """
    homogeneous = (
        "use petgraph::DiGraph;\n"
        "fn island_builders() {}\n"
        "fn node_weight() {}\n"
        "fn migration_builders() {}\n"
        "// migrants are copied verbatim between islands\n"
    )
    # A README may describe every trait of the framework while no code implements one.
    prose = (
        "# Design\n"
        "Islands form a DiGraph with node_weight and edge_weight labels.\n"
        "Migration resamples along the backbone arc-length in [0, 1] using project_onto,\n"
        "with PCHIP, Akima, CubicSpline, Douglas-Peucker, VSpline and TVDenoise,\n"
        "tuned by IRACE over a hyper-heuristic search with z-score aggregation.\n"
    )
    cases = {
        "plain_solver": ({"lib.rs": "def solve(x):\n    return min(x)\n"}, "NOT AN ISLAND MODEL"),
        "homogeneous": ({"lib.rs": homogeneous}, "HOMOGENEOUS ISLAND MODEL - NOT HM-IBF"),
        "prose_only": ({"README.md": prose}, "NOT AN ISLAND MODEL"),
    }
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, (files, expected) in cases.items():
            root = Path(tmp) / name
            root.mkdir()
            for filename, source in files.items():
                (root / filename).write_text(source, encoding="utf-8")
            actual = audit(root).label
            status = "ok" if actual == expected else "FAIL"
            failures += actual != expected
            print(f"[{status}] {name}: expected {expected!r}, got {actual!r}")
    print("self-test passed" if not failures else f"{failures} self-test failure(s)")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Run the auditor from the command line.

    Args:
        argv: Argument vector; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code. 0 for an HM-IBF verdict, 2 otherwise, 1 on a failed self-test.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root", nargs="?", default=".", type=Path, help="repository root to audit (default: .)"
    )
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    parser.add_argument("--self-test", action="store_true", help="verify the classifier itself")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    report = audit(root)
    print(json.dumps(build_payload(report), indent=2) if args.json else render(report))
    return 0 if report.label.startswith("HM-IBF") else 2


if __name__ == "__main__":
    sys.exit(main())
