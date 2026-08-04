"""Data model shared by the HM-IBF auditor's knowledge base and its scanning engine.

The types here carry no detection logic: :mod:`criteria` fills them with the traits to look
for, and :mod:`audit` populates them with evidence from a source tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The trait is backed by implementation code.
PRESENT = "PRESENT"
#: Some markers of the trait exist, but not enough to call it implemented.
PARTIAL = "PARTIAL"
#: No marker of the trait was found.
ABSENT = "ABSENT"


@dataclass(frozen=True)
class Signal:
    """One detectable structural marker of an HM-IBF trait.

    Attributes:
        name: Human-readable description of what the marker demonstrates.
        pattern: Regular expression searched line by line across the source tree.
    """

    name: str
    pattern: str


@dataclass(frozen=True)
class Criterion:
    """One HM-IBF trait, together with the signals that prove it.

    Attributes:
        key: Stable machine-readable identifier.
        title: Heading shown in the report.
        question: The question the criterion answers.
        min_strong: Number of distinct signals required for a `PRESENT` verdict.
        signals: The markers searched for.
    """

    key: str
    title: str
    question: str
    min_strong: int
    signals: tuple[Signal, ...]


@dataclass(frozen=True)
class Deviation:
    """An anti-pattern, or a missing safeguard, worth reporting to the reviewer.

    Attributes:
        key: Stable machine-readable identifier.
        pattern: Regular expression whose presence or absence triggers the message.
        trigger: Either ``"present"`` or ``"absent"``.
        severity: Either ``"high"`` (likely defect) or ``"note"`` (needs a human look).
        message: Explanation and remedy shown in the report.
    """

    key: str
    pattern: str
    trigger: str
    severity: str
    message: str


@dataclass
class Hit:
    """A single matched source line.

    Attributes:
        path: Repository-relative path of the file.
        line: One-based line number.
        text: The stripped, length-capped source line.
        is_doc: Whether the match came from a documentation file.
    """

    path: str
    line: int
    text: str
    is_doc: bool = False


@dataclass
class SignalResult:
    """Outcome of searching for one signal.

    Attributes:
        name: The signal's description.
        hits: The evidence found, capped per file kind.
    """

    name: str
    hits: list[Hit] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """Whether the signal is backed by implementation code.

        Returns:
            True when at least one non-documentation line matched.
        """
        return any(not hit.is_doc for hit in self.hits)

    @property
    def documented_only(self) -> bool:
        """Whether the trait is only described in prose.

        Returns:
            True when documentation matched but no code did.
        """
        return bool(self.hits) and not self.found

    @property
    def evidence(self) -> Hit | None:
        """The strongest single piece of evidence.

        Returns:
            The first code hit, else the first documentation hit, else None.
        """
        return next((hit for hit in self.hits if not hit.is_doc), next(iter(self.hits), None))


@dataclass
class CriterionResult:
    """Verdict for a single criterion.

    Attributes:
        criterion: The evaluated criterion.
        signals: Per-signal outcomes in declaration order.
    """

    criterion: Criterion
    signals: list[SignalResult]

    @property
    def found_count(self) -> int:
        """Number of distinct signals backed by code.

        Returns:
            The count of matched signals.
        """
        return sum(1 for signal in self.signals if signal.found)

    @property
    def verdict(self) -> str:
        """Classification of the criterion.

        Returns:
            :data:`PRESENT`, :data:`PARTIAL` or :data:`ABSENT`.
        """
        if self.found_count >= self.criterion.min_strong:
            return PRESENT
        return PARTIAL if self.found_count else ABSENT


@dataclass
class Report:
    """The complete outcome of one audit.

    Attributes:
        results: Per-criterion verdicts in declaration order.
        deviations: Triggered deviations with their evidence.
        label: The overall classification label.
        rationale: The one-paragraph justification of `label`.
    """

    results: list[CriterionResult]
    deviations: list[tuple[Deviation, list[Hit]]]
    label: str
    rationale: str
