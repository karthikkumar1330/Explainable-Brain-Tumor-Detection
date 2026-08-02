from dataclasses import dataclass
from enum import Enum


class SeverityCategory(Enum):
    """Enumeration representing LOW, MEDIUM, or HIGH severity classes."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass(frozen=True)
class SeverityAssessment:
    """Dataclass holding severity classification outputs, rule details, and disclaimer."""
    category: SeverityCategory
    rule_description: str
    educational_disclaimer: str
