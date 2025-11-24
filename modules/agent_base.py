"""Base agent abstractions and canonical data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .config import AppConfig, ContentProfile


class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class GateDecision(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ContentBlockType(str, Enum):
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    PARAGRAPH = "paragraph"
    LIST = "list"
    FAQ = "faq"
    METADATA = "metadata"


class ContentBlock(BaseModel):
    block_id: str
    type: ContentBlockType
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class DocumentPayload(BaseModel):
    raw_text: str
    blocks: List[ContentBlock]
    profile: ContentProfile
    metadata: Dict[str, str] = Field(default_factory=dict)


class OptimizationFeedback(BaseModel):
    element_identified: str
    current_issue: str
    improvement_mandate: str
    optimized_version: str
    severity: Severity = Severity.MEDIUM
    impact_score: int = Field(50, ge=0, le=100)


class AgentScore(BaseModel):
    value: int = Field(0, ge=0, le=100)
    label: str = "AI Readiness"
    rationale: Optional[str] = None


class AgentPassResult(BaseModel):
    feedback: List[OptimizationFeedback] = Field(default_factory=list)
    optimized_blocks: List[ContentBlock] = Field(default_factory=list)
    score_delta: int = 0


class AgentResult(BaseModel):
    stage_name: str
    decision: GateDecision
    score: AgentScore
    feedback: List[OptimizationFeedback]
    optimized_blocks: List[ContentBlock]
    summary: Optional[str] = None


class AgentContext(BaseModel):
    document: DocumentPayload
    config: AppConfig
    notes: Dict[str, str] = Field(default_factory=dict)


class OptimizationAgent(ABC):
    """Base template for all optimization agents."""

    stage_name: str = "Base Agent"

    def __init__(self, config: AppConfig):
        self.config = config

    def run(self, payload: DocumentPayload) -> AgentResult:
        context = AgentContext(document=payload, config=self.config)
        structural_pass = self.structural_pass(context)
        copy_pass = self.copy_pass(context, structural_pass)

        combined_feedback = structural_pass.feedback + copy_pass.feedback
        optimized_blocks = self._merge_blocks(
            payload.blocks,
            structural_pass.optimized_blocks,
            copy_pass.optimized_blocks,
        )

        score_value = max(
            0,
            min(100, structural_pass.score_delta + copy_pass.score_delta),
        )
        decision = self.decide_gate(score_value, combined_feedback)

        return AgentResult(
            stage_name=self.stage_name,
            decision=decision,
            score=AgentScore(
                value=score_value,
                rationale=self.score_note(payload),
            ),
            feedback=combined_feedback,
            optimized_blocks=optimized_blocks,
            summary=self.summarize(combined_feedback, decision),
        )

    def decide_gate(
        self,
        score_value: int,
        feedback: List[OptimizationFeedback],
    ) -> GateDecision:
        if score_value >= 80:
            return GateDecision.PASSED
        if any(item.severity == Severity.CRITICAL for item in feedback):
            return GateDecision.FAILED
        if self.config.mode == self.config.mode.LITE:
            return GateDecision.SKIPPED
        return GateDecision.FAILED

    def _merge_blocks(
        self,
        original: List[ContentBlock],
        structural: List[ContentBlock],
        copy: List[ContentBlock],
    ) -> List[ContentBlock]:
        replacements = {
            block.block_id: block
            for block in structural + copy
        }
        merged = []
        for block in original:
            merged.append(replacements.get(block.block_id, block))
        return merged

    def score_note(self, payload: DocumentPayload) -> str:
        return (
            f"Profile: {payload.profile.value}; "
            f"Mode: {self.config.mode.value}"
        )

    def summarize(
        self,
        feedback: List[OptimizationFeedback],
        decision: GateDecision,
    ) -> str:
        counts = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 0,
            Severity.MEDIUM: 0,
            Severity.LOW: 0,
        }
        for item in feedback:
            counts[item.severity] += 1
        return (
            f"Decision: {decision.value}. Issues -> "
            f"Critical {counts[Severity.CRITICAL]}, "
            f"High {counts[Severity.HIGH]}, "
            f"Medium {counts[Severity.MEDIUM]}, "
            f"Low {counts[Severity.LOW]}."
        )

    @abstractmethod
    def structural_pass(self, context: AgentContext) -> AgentPassResult:
        """Analyze structure, headings, and intent."""

    @abstractmethod
    def copy_pass(
        self,
        context: AgentContext,
        structural_result: AgentPassResult,
    ) -> AgentPassResult:
        """Analyze paragraphs, sentences, and claims."""
