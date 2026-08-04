"""Data model shared by the retarget catalog and the resolver that anchors it to a tree.

The types here carry no domain knowledge: :mod:`sites` fills them with the change surface of
the pipeline, and :mod:`retarget` resolves them against a working copy so the plan always
carries live `file:line` anchors instead of line numbers that rot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The anchor matched exactly one line; the site is unambiguous.
RESOLVED = "RESOLVED"
#: The anchor matched several lines; every match is reported for the caller to disambiguate.
AMBIGUOUS = "AMBIGUOUS"
#: The anchor matched nothing: the file is gone, or the site has already been retargeted.
MISSING = "MISSING"


@dataclass(frozen=True)
class Layer:
    """One band of the change surface, ordered by how much judgement it needs.

    Attributes:
        key: Stable machine-readable identifier, referenced by :attr:`Site.layer`.
        title: Heading shown in the report.
        note: One-paragraph statement of what editing this band means.
    """

    key: str
    title: str
    note: str


@dataclass(frozen=True)
class Site:
    """One place the retarget has to touch, and the invariant it must not break.

    Attributes:
        key: Stable machine-readable identifier.
        layer: Key of the owning :class:`Layer`.
        path: Repository-relative path of the file holding the site.
        anchor: Regular expression locating the site's defining line.
        title: What the site *is* in general terms, not in alignment terms.
        change: The edit the retarget must make here.
        contract: The invariant that must still hold afterwards, or an empty string.
    """

    key: str
    layer: str
    path: str
    anchor: str
    title: str
    change: str
    contract: str = ""


@dataclass(frozen=True)
class Residue:
    """A domain assumption that must not survive a finished retarget.

    Attributes:
        key: Stable machine-readable identifier.
        pattern: Regular expression searched across the tree's source lines.
        severity: ``"high"`` (silently corrupts results) or ``"note"`` (needs a human look).
        message: What the leftover means and how to resolve it.
    """

    key: str
    pattern: str
    severity: str
    message: str


@dataclass
class Hit:
    """A single matched source line.

    Attributes:
        path: Repository-relative path of the file.
        line: One-based line number.
        text: The stripped, length-capped source line.
    """

    path: str
    line: int
    text: str


@dataclass
class SiteResult:
    """Outcome of anchoring one site to the tree.

    Attributes:
        site: The catalog entry that was resolved.
        hits: Every line the anchor matched, in file order.
    """

    site: Site
    hits: list[Hit] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Classification of the anchor's match count.

        Returns:
            :data:`RESOLVED`, :data:`AMBIGUOUS` or :data:`MISSING`.
        """
        if len(self.hits) == 1:
            return RESOLVED
        return AMBIGUOUS if self.hits else MISSING


@dataclass
class Plan:
    """The complete change surface resolved against one tree.

    Attributes:
        results: Per-site outcomes, in catalog order.
        residues: Domain assumptions still present, with their evidence.
    """

    results: list[SiteResult]
    residues: list[tuple[Residue, list[Hit]]]

    def by_layer(self, layer: str) -> list[SiteResult]:
        """Select the results belonging to one layer.

        Args:
            layer: The layer key to filter by.

        Returns:
            The matching results, in catalog order.
        """
        return [result for result in self.results if result.site.layer == layer]

    @property
    def unresolved(self) -> list[SiteResult]:
        """The sites whose anchor did not match exactly one line.

        Returns:
            Every result that is not :data:`RESOLVED`.
        """
        return [result for result in self.results if result.status != RESOLVED]
