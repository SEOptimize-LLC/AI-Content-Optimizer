"""Streamlit entrypoint for the AI Content Optimizer pipeline."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from modules.agent_base import ContentBlock, ContentBlockType, DocumentPayload
from modules.config import AppConfig, ContentProfile, OptimizationMode
from modules.content_strategist import ContentStrategistAgent
from modules.chunk_optimizer import ChunkOptimizerAgent
from modules.nlp_stylist import NLPStylistAgent
from modules.authority_builder import AuthorityBuilderAgent
from modules.metadata_optimizer import MetadataOptimizerAgent
from utils.llm_handler import OpenRouterClient

st.set_page_config(
    page_title="AI Content Optimizer",
    layout="wide",
)


def parse_blocks(raw_text: str) -> List[ContentBlock]:
    blocks: List[ContentBlock] = []
    block_id = 0
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        block_type = ContentBlockType.PARAGRAPH
        metadata: Dict[str, str] = {}
        if stripped.startswith("### "):
            block_type = ContentBlockType.H3
            stripped = stripped[4:]
        elif stripped.startswith("## "):
            block_type = ContentBlockType.H2
            stripped = stripped[3:]
        elif stripped.startswith("# "):
            block_type = ContentBlockType.H1
            stripped = stripped[2:]
        block_id += 1
        blocks.append(
            ContentBlock(
                block_id=str(block_id),
                type=block_type,
                text=stripped,
                metadata=metadata,
            )
        )
    return blocks


def build_document(
    raw_text: str,
    profile: ContentProfile,
    metadata: Dict[str, str],
) -> DocumentPayload:
    blocks = parse_blocks(raw_text)
    return DocumentPayload(
        raw_text=raw_text,
        blocks=blocks,
        profile=profile,
        metadata=metadata,
    )


def run_pipeline(document: DocumentPayload, config: AppConfig):
    llm_client = OpenRouterClient(config)
    agents = [
        ContentStrategistAgent(config, llm_client),
        ChunkOptimizerAgent(config, llm_client),
        NLPStylistAgent(config, llm_client),
        AuthorityBuilderAgent(config, llm_client),
        MetadataOptimizerAgent(config),
    ]
    results = []
    current_doc = document
    for agent in agents:
        with st.spinner(f"Running {agent.stage_name}..."):
            result = agent.run(current_doc)
            current_doc = DocumentPayload(
                raw_text=current_doc.raw_text,
                blocks=result.optimized_blocks,
                profile=current_doc.profile,
                metadata=current_doc.metadata,
            )
            results.append(result)
    return results, current_doc


def render_feedback(results):
    for result in results:
        title = f"{result.stage_name} · Gate: {result.decision.value.upper()}"
        with st.expander(title, expanded=True):
            if not result.feedback:
                st.success("No issues detected.")
            for item in result.feedback:
                st.markdown(
                    f"**Element:** {item.element_identified}\n\n"
                    f"- Issue: {item.current_issue}\n"
                    f"- Mandate: {item.improvement_mandate}\n"
                    f"- Optimized: {item.optimized_version}"
                )


def main():
    st.title("AI Content Optimization Orchestrator")

    with st.sidebar:
        profile = st.selectbox(
            "Content Profile",
            options=list(ContentProfile),
            format_func=lambda p: p.value,
        )
        mode = st.selectbox(
            "Optimization Mode",
            options=list(OptimizationMode),
            format_func=lambda m: m.value,
        )
        selected_model = st.selectbox(
            "OpenRouter Model",
            options=[
                "openai/gpt-5.1",
                "openai/gpt-4.1-mini",
                "anthropic/claude-sonnet-4.5",
                "google/gemini-3-pro-preview",
                "google/gemini-2.5-flash-preview-09-2025",
                "x-ai/grok-4.1-fast",
                "qwen/qwen-turbo",
                "meta-llama/llama-4-maverick",
                "qwen/qwen3-vl-8b-thinking",
            ],
            index=3,
        )

    config = AppConfig(
        profile=profile,
        mode=mode,
    )
    config.selected_model = selected_model

    st.subheader("Content Inputs")
    primary_keyword = st.text_input(
        "Primary Keyword",
        value="ai content optimization",
    )
    
    uploaded_file = st.file_uploader(
        "Upload Content (Markdown)",
        type=["md", "txt"],
        help="Upload a markdown file containing your content. The app will extract metadata if present.",
    )
    
    raw_content_input = st.text_area(
        "Or Paste Content (Markdown)",
        height=300,
        placeholder="# H1 Example\n\n## How does this work?\nParagraph...",
    )

    if st.button("Run Optimization", type="primary"):
        raw_content = ""
        if uploaded_file is not None:
            raw_content = uploaded_file.read().decode("utf-8")
        elif raw_content_input.strip():
            raw_content = raw_content_input
        else:
            st.error("Please upload a file or paste content to optimize.")
            return

        # Basic metadata extraction from frontmatter could be added here if needed
        # For now, we rely on the primary keyword input and empty defaults for others
        # allowing the MetadataOptimizer to generate them.
        metadata = {
            "primary_keyword": primary_keyword,
            "title": "",  # Will be generated/optimized
            "meta_description": "",  # Will be generated/optimized
            "schema": {},
        }
        
        document = build_document(raw_content, profile, metadata)
        results, final_doc = run_pipeline(document, config)
        render_feedback(results)
        st.subheader("Optimized Blocks")
        for block in final_doc.blocks:
            st.markdown(f"**{block.type.value}:** {block.text}")


if __name__ == "__main__":
    main()
