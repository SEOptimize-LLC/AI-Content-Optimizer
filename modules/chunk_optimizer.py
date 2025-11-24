"""Stage 2: Chunk Optimizer agent scaffolding."""

from __future__ import annotations

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

MIN_CHUNK = 75
MAX_CHUNK = 250


class ChunkOptimizerAgent(OptimizationAgent):
    """Rebuilds body copy into AI-extractable semantic chunks."""

    stage_name = "Stage 2 · Chunk Optimizer"

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

        chunks = self._collect_chunks(context.document.blocks)
        for chunk in chunks:
            word_count = len(chunk.text.split())
            if word_count < MIN_CHUNK:
                feedback.append(
                    self._issue(
                        element=f"Chunk {chunk.metadata.get('h2_label', 'N/A')}",
                        issue="Chunk under 75 words cannot stand alone.",
                        mandate="Expand evidence/context so the chunk reaches 75+ words.",
                        optimized=chunk.text,
                        severity=Severity.MEDIUM,
                    )
                )
            elif word_count > MAX_CHUNK:
                feedback.append(
                    self._issue(
                        element=f"Chunk {chunk.metadata.get('h2_label', 'N/A')}",
                        issue="Chunk exceeds 250 words and mixes concepts.",
                        mandate="Split into multiple semantic chunks with clear focus.",
                        optimized=chunk.text[:200] + "...",
                        severity=Severity.MEDIUM,
                    )
                )
        score_delta = max(0, 100 - 15 * len(feedback))
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
        chunks = self._collect_chunks(context.document.blocks)

        # Limit the number of chunks processed to avoid timeouts
        # Process only the first 5 chunks that fail the AEC check
        processed_count = 0
        MAX_CHUNKS_TO_PROCESS = 5

        for chunk in chunks:
            if not self._passes_aec(chunk.text):
                if processed_count < MAX_CHUNKS_TO_PROCESS:
                    rewritten = self._rewrite_chunk(chunk.text)
                    processed_count += 1
                    
                    optimized_blocks.append(
                        ContentBlock(
                            block_id=chunk.block_id,
                            type=chunk.type,
                            text=rewritten,
                            metadata=chunk.metadata,
                        )
                    )
                    feedback.append(
                        self._issue(
                            element=f"Chunk {chunk.metadata.get('h2_label', 'N/A')}",
                            issue="Chunk does not follow Answer→Evidence→Context style.",
                            mandate="Rewrite chunk so first sentence answers, middle cites data, last sentence explains relevance.",
                            optimized=rewritten,
                            severity=Severity.HIGH,
                        )
                    )
                else:
                    # If we hit the limit, just keep the original chunk but maybe add a note
                    # For now, we just don't optimize it to save time
                    pass

        score_delta = max(0, 80 - 10 * len(feedback))
        return AgentPassResult(
            feedback=feedback,
            optimized_blocks=optimized_blocks,
            score_delta=score_delta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_chunks(self, blocks: List[ContentBlock]) -> List[ContentBlock]:
        chunks: List[ContentBlock] = []
        current_h2 = None
        for block in blocks:
            if block.type == ContentBlockType.H2:
                current_h2 = block.text
            if block.type == ContentBlockType.PARAGRAPH:
                metadata = dict(block.metadata)
                if current_h2:
                    metadata["h2_label"] = current_h2
                chunks.append(
                    ContentBlock(
                        block_id=block.block_id,
                        type=block.type,
                        text=block.text,
                        metadata=metadata,
                    )
                )
        return chunks

    def _passes_aec(self, text: str) -> bool:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 3:
            return False
        return True

    def _rewrite_chunk(self, text: str) -> str:
        if not self.llm:
            return text
        prompt = (
            "Rewrite the chunk using Answer→Evidence→Context. First sentence must "
            "answer the H2 question directly, next 2-3 sentences cite data or logic, "
            "final sentence explains why it matters."
        )
        response = self.llm.chat(
            messages=[
                ChatMessage(role="system", content="You optimize chunks for AI extraction."),
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
        impact = 85 if severity in {Severity.CRITICAL, Severity.HIGH} else 60
        return OptimizationFeedback(
            element_identified=element,
            current_issue=issue,
            improvement_mandate=mandate,
            optimized_version=optimized,
            severity=severity,
            impact_score=impact,
        )
