"""Stage 3: NLP Stylist agent scaffolding."""

from __future__ import annotations

import re
from typing import List, Optional

from .agent_base import (
    AgentContext,
    AgentPassResult,
    ContentBlock,
    ContentBlockType,
    OptimizationAgent,
    OptimizationFeedback,
    Severity,
)
from .config import AppConfig
from ..utils.llm_handler import ChatMessage, OpenRouterClient

MAX_SENTENCE_LEN = 30


class NLPStylistAgent(OptimizationAgent):
    """Refactors sentences so they are extraction-friendly for NLP models."""

    stage_name = "Stage 3 · NLP Stylist"

    def __init__(
        self,
        config: AppConfig,
        llm_client: Optional[OpenRouterClient] = None,
    ) -> None:
        super().__init__(config)
        self.llm = llm_client

    def structural_pass(self, context: AgentContext) -> AgentPassResult:
        feedback: List[OptimizationFeedback] = []
        optimized_blocks: List[ContentBlock] = []

        for block in context.document.blocks:
            if block.type != ContentBlockType.PARAGRAPH:
                continue
            sentences = self._sentences(block.text)
            long_sentences = [
                sentence
                for sentence in sentences
                if len(sentence.split()) > MAX_SENTENCE_LEN
            ]
            passive_sentences = [
                sentence
                for sentence in sentences
                if self._looks_passive(sentence)
            ]

            if long_sentences:
                feedback.append(
                    self._issue(
                        element=f"Paragraph {block.block_id}",
                        issue="Contains sentences longer than 30 words.",
                        mandate=(
                            "Split sentences so each carries one "
                            "subject-verb-object idea."
                        ),
                        optimized=long_sentences[0],
                        severity=Severity.MEDIUM,
                    )
                )
            if passive_sentences:
                feedback.append(
                    self._issue(
                        element=f"Paragraph {block.block_id}",
                        issue=(
                            "Relies on passive voice, which hinders "
                            "AI extraction."
                        ),
                        mandate=(
                            "Rewrite sentences so the subject performs "
                            "the action directly."
                        ),
                        optimized=passive_sentences[0],
                        severity=Severity.MEDIUM,
                    )
                )

        score_delta = max(0, 90 - 10 * len(feedback))
        return AgentPassResult(
            feedback=feedback,
            optimized_blocks=optimized_blocks,
            score_delta=score_delta,
        )

    def copy_pass(
        self,
        context: AgentContext,
        structural_result: AgentPassResult,
    ) -> AgentPassResult:
        feedback: List[OptimizationFeedback] = []
        optimized_blocks: List[ContentBlock] = []

        for block in context.document.blocks:
            if block.type != ContentBlockType.PARAGRAPH:
                continue
            if self._needs_density_upgrade(block.text):
                rewritten = self._rewrite_dense(block.text)
                optimized_blocks.append(
                    ContentBlock(
                        block_id=block.block_id,
                        type=block.type,
                        text=rewritten,
                        metadata=block.metadata,
                    )
                )
                feedback.append(
                    self._issue(
                        element=f"Paragraph {block.block_id}",
                        issue=(
                            "Sentences lack entity-rich, active "
                            "constructions."
                        ),
                        mandate=(
                            "Rewrite using SVO, explicit entities, and "
                            "cause→effect connectors."
                        ),
                        optimized=rewritten,
                        severity=Severity.HIGH,
                    )
                )

        score_delta = max(0, 80 - 10 * len(feedback))
        return AgentPassResult(
            feedback=feedback,
            optimized_blocks=optimized_blocks,
            score_delta=score_delta,
        )

    def _sentences(self, text: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]

    def _looks_passive(self, sentence: str) -> bool:
        return bool(
            re.search(
                r"\b(be|been|being|is|was|were)\b\s+\w+ed\b",
                sentence.lower(),
            )
        )

    def _needs_density_upgrade(self, text: str) -> bool:
        sentences = self._sentences(text)
        if not sentences:
            return False
        entity_mentions = re.findall(r"[A-Z][a-z]+", text)
        return len(sentences) < 3 or len(entity_mentions) < 2

    def _rewrite_dense(self, text: str) -> str:
        if not self.llm:
            return text
        prompt = (
            "Rephrase this paragraph using short SVO sentences and causal "
            "connectors. Add quantified comparisons and cite explicit "
            "entities."
        )
        response = self.llm.chat(
            messages=[
                ChatMessage(
                    role="system",
                    content="You polish text for NLP extraction.",
                ),
                ChatMessage(
                    role="user",
                    content=f"{prompt}\n\n{text}",
                ),
            ]
        )
        return response.content.strip()

    def _issue(
        self,
        element: str,
        issue: str,
        mandate: str,
        optimized: str,
        severity: Severity,
    ) -> OptimizationFeedback:
        return OptimizationFeedback(
            element_identified=element,
            current_issue=issue,
            improvement_mandate=mandate,
            optimized_version=optimized,
            severity=severity,
            impact_score=85 if severity == Severity.HIGH else 60,
        )
