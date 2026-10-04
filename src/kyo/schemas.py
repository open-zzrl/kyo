from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DecisionResult:
    decision: str
    confidence: float
    scores: Dict[str, float]
    fallback_to_llm: bool
    latency_ms: float
    raw_payload: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        status = " [ESCALATE_TO_LLM]" if self.fallback_to_llm else " [LOCAL_EXEC]"
        return (
            f"<DecisionResult: choice='{self.decision}', "
            f"confidence={self.confidence * 100:.1f}%, "
            f"latency={self.latency_ms:.2f}ms{status}>"
        )
