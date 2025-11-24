"""Stage 1: Content Strategist agent scaffolding."""

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
from modules.config import AppConfig, RuleSet
from utils.llm_handler import ChatMessage, OpenRouterClient


class ContentStrategistAgent(OptimizationAgent):
    """Guarantees structural readiness before other agents run."""

    stage_name = "Stage 1 · Content Strategist"

    def __init__(
        self,
        config: AppConfig,
        llm_client: Optional[OpenRouterClient] = None,
    ) -> None:
        super().__init__(config)
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Structural layer
    # ------------------------------------------------------------------
    def structural_pass(self, context: AgentContext) -> AgentPassResult:
        feedback: List[OptimizationFeedback] = []
        optimized_blocks: List[ContentBlock] = []
        rules: RuleSet = self.config.rules

        blocks = context.document.blocks
        h1_blocks = [b for b in blocks if b.type == ContentBlockType.H1]
        h2_blocks = [b for b in blocks if b.type == ContentBlockType.H2]

        if len(h1_blocks) != 1:
            feedback.append(
                self._issue(
                    element="H1",
                    issue="Document should contain exactly one H1 heading.",
                    mandate=(
                        "Create a single H1 that states the primary question "
                        "this page answers for AI Overviews."
                    ),
                    optimized=self._suggest_core_question(h1_blocks[:1]),
                    severity=Severity.HIGH,
                )
            )

        if rules.require_answer_first_intro:
            intro = self._first_paragraph(blocks)
            if intro and not self._is_answer_first_intro(intro.text):
                optimized_blocks.append(
                    ContentBlock(
                        block_id=intro.block_id,
                        type=intro.type,
                        text=self._rewrite_intro(intro.text, context),
                        metadata=intro.metadata,
                    )
                )
                feedback.append(
                    self._issue(
                        element="Introduction",
                        issue=(
                            "Opening paragraph is not an answer-first "
                            "30-50 word summary."
                        ),
                        mandate=(
                            "Lead with the direct answer and preview the H2 "
                            "questions you will cover."
                        ),
                        optimized=optimized_blocks[-1].text,
                        severity=Severity.HIGH,
                    )
                )

        if rules.require_h2_questions:
            for block in h2_blocks:
                if not block.text.strip().endswith("?"):
                    optimized = self._questionize(block.text)
                    optimized_blocks.append(
                        ContentBlock(
                            block_id=block.block_id,
                            type=block.type,
                            text=optimized,
                            metadata=block.metadata,
                        )
                    )
                    feedback.append(
                        self._issue(
                            element=f"H2 · {block.text[:50]}",
                            issue=(
                                "H2 headings must be phrased as natural "
                                "questions aligned to intent."
                            ),
                            mandate=(
                                "Rewrite H2s into intent-based "
                                "questions so AI crawlers can "
                                "map them to clear needs."
                            ),
                            optimized=optimized,
                            severity=Severity.MEDIUM,
                        )
                    )

        if rules.require_faq:
            faq_blocks = [b for b in blocks if b.type == ContentBlockType.FAQ]
            if len(faq_blocks) < 3:
                feedback.append(
                    self._issue(
                        element="FAQ section",
                        issue=(
                            "Needs at least three long-tail follow-up "
                            "questions beyond the H2 coverage."
                        ),
                        mandate=(
                            "Add 3-5 FAQ entries covering adjacent intents so AI "
                            "overviews can reference them."
                        ),
                        optimized=(
                            "Add FAQ entries referencing the H2 coverage and "
                            "new long-tail intents."
                        ),
                        severity=Severity.MEDIUM,
                    )
                )

        score_delta = max(0, 100 - 20 * len(feedback))
        return AgentPassResult(
            feedback=feedback,
            optimized_blocks=optimized_blocks,
            score_delta=score_delta,
        )

    # ------------------------------------------------------------------
    # Copy layer (still structural but requires rewriting)
    # ------------------------------------------------------------------
    def copy_pass(
        self,
        context: AgentContext,
        structural_result: AgentPassResult,
    ) -> AgentPassResult:
        feedback: List[OptimizationFeedback] = []
        optimized_blocks: List[ContentBlock] = []

        intro_block = self._first_paragraph(context.document.blocks)
        intro_id = getattr(intro_block, "block_id", None)
        already_adjusted = any(
            block.block_id == intro_id
            for block in structural_result.optimized_blocks
        )

        if not already_adjusted and intro_block and self.llm:
            optimized_text = self._llm_answer_first_intro(intro_block.text)
            optimized_blocks.append(
                ContentBlock(
                    block_id=intro_block.block_id,
                    type=intro_block.type,
                    text=optimized_text,
                    metadata=intro_block.metadata,
                )
            )
            feedback.append(
                self._issue(
                    element="Introduction",
                    issue=(
                        "LLM rewrite enforced the answer-first specification "
                        "required for AI crawlers."
                    ),
                    mandate=(
                        "Adopt the Strategist rewrite so Stage 2 can chunk "
                        "around a clean answer-first intro."
                    ),
                    optimized=optimized_text,
                    severity=Severity.MEDIUM,
                )
            )

        score_delta = 10 if not feedback else 0
        return AgentPassResult(
            feedback=feedback,
            optimized_blocks=optimized_blocks,
            score_delta=score_delta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _first_paragraph(
        self,
        blocks: List[ContentBlock],
    ) -> Optional[ContentBlock]:
        for block in blocks:
            if block.type == ContentBlockType.PARAGRAPH:
                return block
        return None

    def _is_answer_first_intro(self, text: str) -> bool:
        word_count = len(text.split())
        has_preview = bool(
            re.search(r"(we'll|this guide|you'll)", text, flags=re.IGNORECASE)
        )
        return 30 <= word_count <= 60 and has_preview

    def _questionize(self, text: str) -> str:
        cleaned = text.strip().rstrip("?")
        pattern = r"^(how|what|why|when|where|who|should)"
        if not re.match(pattern, cleaned, flags=re.IGNORECASE):
            cleaned = f"How does {cleaned}".strip()
        return f"{cleaned}?"

    def _rewrite_intro(self, text: str, context: AgentContext) -> str:
        keyword = context.document.metadata.get(
            "primary_keyword",
            "your topic",
        )
        return (
            f"The fastest way to understand {keyword} is to answer the core "
            "question up front, then preview the H2 questions this guide "
            "covers."
        )

    def _suggest_core_question(self, h1_blocks: List[ContentBlock]) -> str:
        base = h1_blocks[0].text if h1_blocks else "the main topic"
        base = base.rstrip("?")
        return f"What is {base}?"

    def _issue(
        self,
        element: str,
        issue: str,
        mandate: str,
        optimized: str,
        severity: Severity,
    ) -> OptimizationFeedback:
        impact = 90 if severity in {Severity.CRITICAL, Severity.HIGH} else 60
        return OptimizationFeedback(
            element_identified=element,
            current_issue=issue,
            improvement_mandate=mandate,
            optimized_version=optimized,
            severity=severity,
            impact_score=impact,
        )

    def _llm_answer_first_intro(self, text: str) -> str:
        if not self.llm:
            return text
        prompt = (
            "Rewrite the introduction into a 35-45 word answer-first paragraph. "
            "State the direct answer in sentence one, then preview the H2 "
            "questions."
        )
        response = self.llm.chat(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are an SEO content strategist.",
                ),
                ChatMessage(
                    role="user",
                    content=f"{prompt}\n\n{text}",
                ),
            ]
        )
        return response.content.strip()
