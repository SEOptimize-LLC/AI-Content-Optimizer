"""Stage 4: Authority Builder agent scaffolding."""

from __future__ import annotations

import re
from typing import List, Optional

from modules.agent_base import (
    AgentContext,
    AgentPassResult,
    ContentBlock,
    ContentBlockType,
    OptimizationAgent,
    OptimizationFeedback,
    Severity,
)
from modules.config import AppConfig
from utils.llm_handler import ChatMessage, OpenRouterClient


class AuthorityBuilderAgent(OptimizationAgent):
    """Injects citations, dates, and experience/E-E-A-T signals."""

    stage_name = "Stage 4 · Authority Builder"

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
            if not self._has_citation(block.text):
                feedback.append(
                    self._issue(
                        element=f"Paragraph {block.block_id}",
                        issue="Missing inline attribution with a specific source.",
                        mandate="Reference a named source with a direct URL and publication year.",
                        optimized="Add citation",
                        severity=Severity.HIGH,
                    )
                )
            if not self._has_fresh_year(block.text):
                feedback.append(
                    self._issue(
                        element=f"Paragraph {block.block_id}",
                        issue="No publication year attached to the cited evidence.",
                        mandate="Add a year (2022+) to signal freshness.",
                        optimized="Add year",
                        severity=Severity.MEDIUM,
                    )

        score_delta = max(0, 85 - 10 * len(feedback))
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
            if self._needs_authority_upgrade(block.text):
                rewritten = self._rewrite_authority(block.text)
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
                        issue="Paragraph lacks explicit E-E-A-T markers.",
                        mandate="Add data-backed citation, date, and first-hand signal (e.g., 'In our audits...').",
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

    def _has_citation(self, text: str) -> bool:
        return bool(re.search(r"https?://", text))

    def _has_fresh_year(self, text: str) -> bool:
        return bool(re.search(r"20(2[0-9]|3[0-9])", text))

    def _needs_authority_upgrade(self, text: str) -> bool:
        return not (self._has_citation(text) and self._has_fresh_year(text))

    def _rewrite_authority(self, text: str) -> str:
        if not self.llm:
            return text
        prompt = (
            "Strengthen this paragraph with: 1) an explicit source + URL, 2) a "
            "publication year, and 3) a first-hand/experience statement. Keep "
            "tone factual and cite reputable domains (.gov/.edu/.org)."
        )
        response = self.llm.chat(
            messages=[
                ChatMessage(role="system", content="You add authoritative citations."),
                ChatMessage(role="user", content=f"{prompt}\n\n{text}"),
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
        impact = 95 if severity == Severity.HIGH else 70
        return OptimizationFeedback(
            element_identified=element,
            current_issue=issue,
            improvement_mandate=mandate,
            optimized_version=optimized,
            severity=severity,
            impact_score=impact,
        )
