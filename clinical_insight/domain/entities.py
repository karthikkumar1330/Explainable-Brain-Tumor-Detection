from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ClinicalInsight:
    """Dataclass holding synthesized clinical AI insights, warnings, and regulatory disclaimers."""
    summary_narrative: str
    key_findings: List[str]
    recommendations: List[str]
    disclaimer: str

    def to_markdown(self) -> str:
        """Helper to render insights to markdown format."""
        findings_str = "\n".join(f"- {f}" for f in self.key_findings)
        recs_str = "\n".join(f"- {r}" for r in self.recommendations)
        return (
            f"### AI Clinical Summary\n{self.summary_narrative}\n\n"
            f"### Key Findings\n{findings_str}\n\n"
            f"### Recommended Actions\n{recs_str}\n\n"
            f"> [!WARNING]\n"
            f"> **Disclaimer:** {self.disclaimer}\n"
        )
