"""Stage 5: Metadata & Schema optimizer scaffolding."""

from __future__ import annotations

from typing import Dict, List

from .agent_base import (
    AgentContext,
    AgentPassResult,
    ContentBlock,
    OptimizationAgent,
    OptimizationFeedback,
    Severity,
)
from .config import AppConfig

TITLE_LIMIT = 60
META_LIMIT_MIN = 140
META_LIMIT_MAX = 160


class MetadataOptimizerAgent(OptimizationAgent):
    """Aligns title, meta, and schema with the optimized body copy."""

    stage_name = "Stage 5 · Metadata & Schema"

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)

    def structural_pass(self, context: AgentContext) -> AgentPassResult:
        feedback: List[OptimizationFeedback] = []
        optimized_blocks: List[ContentBlock] = []
        meta = context.document.metadata or {}

        title = meta.get("title")
        meta_desc = meta.get("meta_description")
        schema = meta.get("schema", {})

        if not title or len(title) > TITLE_LIMIT:
            feedback.append(
                self._issue(
                    element="Title Tag",
                    issue="Missing or exceeds 60 characters.",
                    mandate=(
                        "Provide an answer-first title under 60 characters "
                        "that reflects the Stage 1 core question."
                    ),
                    optimized=title or "",
                    severity=Severity.HIGH,
                )
            )

        within_meta_range = meta_desc and (
            META_LIMIT_MIN <= len(meta_desc) <= META_LIMIT_MAX
        )
        if not within_meta_range:
            feedback.append(
                self._issue(
                    element="Meta Description",
                    issue=(
                        "Meta description missing or outside the "
                        "140-160 character window."
                    ),
                    mandate=(
                        "Summarize the core answer plus authority signal "
                        "within the 140-160 character band."
                    ),
                    optimized=meta_desc or "",
                    severity=Severity.MEDIUM,
                )
            )

        if not self._has_faq_schema(schema, context.document.blocks):
            feedback.append(
                self._issue(
                    element="FAQ Schema",
                    issue=(
                        "FAQPage schema missing or out of sync with "
                        "the on-page FAQ blocks."
                    ),
                    mandate=(
                        "Generate FAQPage JSON-LD that mirrors the Stage 1 "
                        "entries to keep SERP metadata aligned."
                    ),
                    optimized="Add FAQ schema",
                    severity=Severity.MEDIUM,
                )
            )

        score_delta = max(0, 90 - 15 * len(feedback))
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
        meta = dict(context.document.metadata or {})

        if not meta.get("title") or len(meta["title"]) > TITLE_LIMIT:
            meta["title"] = self._synthesize_title(context)
            feedback.append(
                self._issue(
                    element="Title Tag",
                    issue=(
                        "Generated optimized title ready for SERP and AI "
                        "overview display."
                    ),
                    mandate=(
                        "Adopt the synthesized answer-first title before "
                        "publishing."
                    ),
                    optimized=meta["title"],
                    severity=Severity.MEDIUM,
                )
            )

        meta_text = meta.get("meta_description", "")
        if not meta_text or not (
            META_LIMIT_MIN <= len(meta_text) <= META_LIMIT_MAX
        ):
            meta_text = self._synthesize_meta(context)
            meta["meta_description"] = meta_text
            feedback.append(
                self._issue(
                    element="Meta Description",
                    issue="Optimized meta generated from Stage 1-4 outputs.",
                    mandate=(
                        "Use the synthesized meta to keep SERP and AI "
                        "messaging fully aligned."
                    ),
                    optimized=meta_text,
                    severity=Severity.MEDIUM,
                )
            )

        context.document.metadata.update(meta)
        score_delta = 10 if feedback else 0
        return AgentPassResult(
            feedback=feedback,
            optimized_blocks=optimized_blocks,
            score_delta=score_delta,
        )

    def _has_faq_schema(
        self,
        schema: Dict[str, object],
        blocks: List[ContentBlock],
    ) -> bool:
        faq_blocks = [b for b in blocks if b.metadata.get("h2_label") == "FAQ"]
        faq_schema = schema.get("faq", []) if isinstance(schema, dict) else []
        return bool(faq_blocks and faq_schema)

    def _synthesize_title(self, context: AgentContext) -> str:
        keyword = context.document.metadata.get(
            "primary_keyword",
            "AI content optimization",
        )
        return f"{keyword.title()} – Answer, Evidence, Authority"

    def _synthesize_meta(self, context: AgentContext) -> str:
        keyword = context.document.metadata.get(
            "primary_keyword",
            "AI content optimization",
        )
        return (
            f"Learn how to optimize {keyword} with question-based structure, "
            "AEC chunks, NLP-friendly sentences, and cited authority."
        )

    def _issue(
        self,
        element: str,
        issue: str,
        mandate: str,
        optimized: str,
        severity: Severity,
    ) -> OptimizationFeedback:
        impact = 80 if severity in {Severity.CRITICAL, Severity.HIGH} else 55
        return OptimizationFeedback(
            element_identified=element,
            current_issue=issue,
            improvement_mandate=mandate,
            optimized_version=optimized,
            severity=severity,
            impact_score=impact,
        )
