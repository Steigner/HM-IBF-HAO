"""Resolve the retarget change surface against a working tree and check it afterwards.

Two modes:

* ``plan`` anchors every catalog entry to a live `file:line` and prints the ordered edit
  surface, so the retarget never runs on line numbers that have drifted.
* ``check`` searches the tree for horizontal-alignment assumptions that must not survive a
  finished retarget - above all the terrain objective and the perpendicular-offset
  encoding, which keep producing plausible numbers about the wrong problem.

The change surface itself lives in :mod:`sites`; this module is only the resolver. Run
``python3 retarget.py --help`` for usage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from model import AMBIGUOUS, MISSING, RESOLVED, Hit, Plan, Residue, SiteResult
from sites import LAYERS, RESIDUES, SITES

#: File extensions searched when hunting for leftover alignment assumptions.
SOURCE_SUFFIXES = frozenset({".rs", ".py", ".toml", ".conf", ".sh", ".bat", ".md"})

#: Directory names never searched: build output, vendored code and benchmark payloads.
#:
#: ``.claude`` is skipped because it holds this catalog, whose own pattern strings would
#: otherwise match every residue it looks for.
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

#: Maximum evidence lines retained per residue.
MAX_EVIDENCE = 5

#: Longest source line reproduced in the report.
MAX_LINE_WIDTH = 100

#: Width of the report's rules.
RULE_WIDTH = 78


def find_anchor(root: Path, path: str, anchor: str) -> list[Hit]:
    """Locate a site's defining line inside one file.

    Args:
        root: Repository root.
        path: Repository-relative path of the file to search.
        anchor: Regular expression matched against each line.

    Returns:
        Every matching line in file order; empty when the file is absent or nothing matched.
    """
    target = root / path
    if not target.is_file():
        return []

    regex = re.compile(anchor)
    content = target.read_text(encoding="utf-8", errors="replace")
    return [
        Hit(path, number, text.strip()[:MAX_LINE_WIDTH])
        for number, text in enumerate(content.splitlines(), start=1)
        if regex.search(text)
    ]


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield every searchable source file below `root`.

    Args:
        root: Directory to walk.

    Yields:
        Paths whose suffix is in :data:`SOURCE_SUFFIXES` and that live outside
        :data:`SKIPPED_DIRECTORIES`.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if SKIPPED_DIRECTORIES.intersection(path.relative_to(root).parts[:-1]):
            continue
        yield path


def find_residues(root: Path) -> list[tuple[Residue, list[Hit]]]:
    """Search the tree for alignment assumptions that a finished retarget must have removed.

    Args:
        root: Repository root to search.

    Returns:
        Pairs of residue and its capped evidence, for the residues that matched at all.
    """
    compiled = {residue.key: re.compile(residue.pattern) for residue in RESIDUES}
    found: dict[str, list[Hit]] = {residue.key: [] for residue in RESIDUES}

    for path in iter_source_files(root):
        relative = path.relative_to(root).as_posix()
        content = path.read_text(encoding="utf-8", errors="replace")
        for number, text in enumerate(content.splitlines(), start=1):
            for key, regex in compiled.items():
                if len(found[key]) < MAX_EVIDENCE and regex.search(text):
                    found[key].append(Hit(relative, number, text.strip()[:MAX_LINE_WIDTH]))

    return [(residue, found[residue.key]) for residue in RESIDUES if found[residue.key]]


def build_plan(root: Path, with_residues: bool) -> Plan:
    """Resolve the whole catalog against a tree.

    Args:
        root: Repository root.
        with_residues: Whether to also search for leftover alignment assumptions.

    Returns:
        The resolved plan.
    """
    results = [SiteResult(site, find_anchor(root, site.path, site.anchor)) for site in SITES]
    residues = find_residues(root) if with_residues else []
    return Plan(results=results, residues=residues)


def render_plan(plan: Plan) -> str:
    """Render the ordered edit surface.

    Args:
        plan: The resolved plan.

    Returns:
        The formatted report.
    """
    mark = {RESOLVED: "[x]", AMBIGUOUS: "[~]", MISSING: "[ ]"}
    lines = [
        "",
        "=" * RULE_WIDTH,
        f"RETARGET SURFACE: {len(SITES)} sites across {len(LAYERS)} layers",
        "=" * RULE_WIDTH,
    ]

    for layer in LAYERS:
        results = plan.by_layer(layer.key)
        lines.extend(("", "-" * RULE_WIDTH, layer.title, "-" * RULE_WIDTH, layer.note, ""))
        for result in results:
            site = result.site
            anchor = _format_anchor(result)
            lines.append(f"{mark[result.status]} {site.key}: {site.title}")
            lines.append(f"      {anchor}")
            lines.append(f"      change:   {site.change}")
            if site.contract:
                lines.append(f"      contract: {site.contract}")
            lines.append("")

    lines.extend(_render_residues(plan))
    return "\n".join(lines)


def _format_anchor(result: SiteResult) -> str:
    """Describe where a site resolved to.

    Args:
        result: The resolved site.

    Returns:
        A `path:line` reference, or an explanation of why the anchor did not resolve.
    """
    if result.status == RESOLVED:
        return f"{result.hits[0].path}:{result.hits[0].line}"
    if result.status == AMBIGUOUS:
        return "ambiguous: " + ", ".join(f"{hit.path}:{hit.line}" for hit in result.hits)
    return f"NOT FOUND in {result.site.path} (already retargeted, moved, or renamed)"


def _render_residues(plan: Plan) -> list[str]:
    """Render the leftover-assumption section.

    Args:
        plan: The resolved plan.

    Returns:
        The section's lines; a single note when residues were not searched for.
    """
    lines = ["-" * RULE_WIDTH, "ALIGNMENT ASSUMPTIONS STILL PRESENT", "-" * RULE_WIDTH]
    if not plan.residues:
        lines.append("None. Every alignment assumption in the catalog has been replaced.")
        return lines

    for residue, hits in plan.residues:
        lines.append(f"[{residue.severity}] {residue.key}")
        lines.append(f"    {residue.message}")
        lines.extend(f"      {hit.path}:{hit.line}: {hit.text}" for hit in hits)
        lines.append("")
    return lines


def build_payload(plan: Plan) -> dict:
    """Assemble the machine-readable plan.

    Args:
        plan: The resolved plan.

    Returns:
        A JSON-serialisable dictionary.
    """
    return {
        "layers": [
            {
                "key": layer.key,
                "title": layer.title,
                "note": layer.note,
                "sites": [
                    {
                        "key": result.site.key,
                        "title": result.site.title,
                        "path": result.site.path,
                        "status": result.status,
                        "lines": [hit.line for hit in result.hits],
                        "change": result.site.change,
                        "contract": result.site.contract,
                    }
                    for result in plan.by_layer(layer.key)
                ],
            }
            for layer in LAYERS
        ],
        "residues": [
            {
                "key": residue.key,
                "severity": residue.severity,
                "message": residue.message,
                "evidence": [{"path": hit.path, "line": hit.line} for hit in hits],
            }
            for residue, hits in plan.residues
        ],
    }


def self_test() -> int:
    """Check the resolver against synthetic trees with known outcomes.

    Returns:
        Process exit code: 0 when every expectation holds, 1 otherwise.
    """
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        empty = build_plan(root, with_residues=True)
        failures += _expect(
            "an empty tree resolves no site",
            all(result.status == MISSING for result in empty.results),
        )
        failures += _expect("an empty tree has no residue", not empty.residues)

        source = root / "hm-ibf-hao" / "src" / "alignment"
        source.mkdir(parents=True)
        (source / "problem.rs").write_text(
            "pub fn control_polygon(&self) {}\nlet clothoid = 1;\n", encoding="utf-8"
        )
        seeded = build_plan(root, with_residues=True)
        failures += _expect(
            "a present anchor resolves to its line",
            next(r for r in seeded.results if r.site.key == "block_size").hits[0].line == 1,
        )
        failures += _expect(
            "a leftover clothoid decoder is reported",
            any(residue.key == "clothoid_decoder" for residue, _ in seeded.residues),
        )

        (source / "problem.rs").write_text(
            "pub fn control_polygon(&self) {}\npub fn control_polygon(&mut self) {}\n",
            encoding="utf-8",
        )
        duplicated = build_plan(root, with_residues=False)
        failures += _expect(
            "a duplicated anchor is reported as ambiguous",
            next(r for r in duplicated.results if r.site.key == "block_size").status == AMBIGUOUS,
        )

    print("self-test passed" if not failures else f"{failures} self-test failure(s)")
    return 1 if failures else 0


def _expect(description: str, condition: bool) -> int:
    """Report one self-test expectation.

    Args:
        description: What the expectation checks.
        condition: Whether it held.

    Returns:
        0 when the expectation held, 1 otherwise.
    """
    print(f"[{'ok' if condition else 'FAIL'}] {description}")
    return 0 if condition else 1


def main(argv: list[str] | None = None) -> int:
    """Run the resolver from the command line.

    Args:
        argv: Argument vector; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code. 0 when the requested mode succeeded, 2 when `check` found a
        high-severity leftover or `plan` could not resolve a site, 1 on a failed self-test.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root", nargs="?", default=".", type=Path, help="repository root (default: .)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report alignment assumptions a finished retarget must have removed",
    )
    parser.add_argument("--json", action="store_true", help="emit the machine-readable plan")
    parser.add_argument("--self-test", action="store_true", help="verify the resolver itself")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    plan = build_plan(root, with_residues=args.check)
    print(json.dumps(build_payload(plan), indent=2) if args.json else render_plan(plan))

    if args.check:
        return 2 if any(residue.severity == "high" for residue, _ in plan.residues) else 0
    return 2 if plan.unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
