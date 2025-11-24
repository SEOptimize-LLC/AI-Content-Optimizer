"""Configuration primitives for the AI Content Optimizer framework."""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field

# -- Content Profiles & Modes -------------------------------------------------


class ContentProfile(str, Enum):
    BLOG = "Blog Post"
    PRODUCT_PAGE = "Product Page"
    SERVICE_PAGE = "Service Page"
    THOUGHT_LEADERSHIP = "Thought Leadership"
    KNOWLEDGE_BASE = "Knowledge Base"


class OptimizationMode(str, Enum):
    STRICT = "Strict (Enforce all rules)"
    LITE = "Lite (Suggestions only)"


# -- OpenRouter Model Catalog -------------------------------------------------
# Users will provide an `OPENROUTER_API_KEY` via Streamlit Secrets and can pick
# any of the models below at runtime.
OPENROUTER_MODELS = [
    {
        "id": "openai/gpt-5.1",
        "label": "OpenAI GPT-5.1",
        "tier": "premium",
    },
    {
        "id": "openai/gpt-4.1-mini",
        "label": "OpenAI GPT-4.1 Mini",
        "tier": "balanced",
    },
    {
        "id": "anthropic/claude-sonnet-4.5",
        "label": "Anthropic Claude Sonnet 4.5",
        "tier": "premium",
    },
    {
        "id": "google/gemini-3-pro-preview",
        "label": "Google Gemini 3 Pro Preview",
        "tier": "balanced",
    },
    {
        "id": "google/gemini-2.5-flash-preview-09-2025",
        "label": "Google Gemini 2.5 Flash Preview (09/2025)",
        "tier": "fast",
    },
    {
        "id": "x-ai/grok-4.1-fast",
        "label": "xAI Grok 4.1 Fast",
        "tier": "fast",
    },
    {
        "id": "qwen/qwen-turbo",
        "label": "Qwen Turbo",
        "tier": "fast",
    },
    {
        "id": "meta-llama/llama-4-maverick",
        "label": "Meta Llama 4 Maverick",
        "tier": "balanced",
    },
    {
        "id": "qwen/qwen3-vl-8b-thinking",
        "label": "Qwen3 VL 8B Thinking",
        "tier": "vision",
    },
]


class OpenRouterSettings(BaseModel):
    api_base: str = "https://openrouter.ai/api/v1"
    default_model: str = "google/gemini-3-pro-preview"
    available_models: List[str] = Field(
        default_factory=lambda: [model["id"] for model in OPENROUTER_MODELS]
    )


# -- Rule Sets ----------------------------------------------------------------


class RuleSet(BaseModel):
    require_h2_questions: bool = True
    require_answer_first_intro: bool = True
    chunk_length_min: int = 75
    chunk_length_max: int = 250
    require_faq: bool = True
    evidence_density_check: bool = True
    svo_pattern_preference: str = "High"  # High, Medium, Low

    @classmethod
    def get_for_profile(cls, profile: ContentProfile) -> "RuleSet":
        rules = cls()

        if profile == ContentProfile.THOUGHT_LEADERSHIP:
            rules.require_h2_questions = False
            rules.chunk_length_max = 300

        elif profile == ContentProfile.PRODUCT_PAGE:
            rules.require_answer_first_intro = False
            rules.chunk_length_min = 50
            rules.require_faq = True

        elif profile == ContentProfile.KNOWLEDGE_BASE:
            rules.require_h2_questions = False
            rules.svo_pattern_preference = "Medium"

        return rules


class AppConfig(BaseModel):
    profile: ContentProfile = ContentProfile.BLOG
    mode: OptimizationMode = OptimizationMode.STRICT
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    selected_model: str = "google/gemini-3-pro-preview"

    @property
    def rules(self) -> RuleSet:
        return RuleSet.get_for_profile(self.profile)

    @property
    def model_options(self):
        return OPENROUTER_MODELS

    def validate_model(self, model_id: str) -> str:
        if model_id not in self.openrouter.available_models:
            raise ValueError(
                f"Model '{model_id}' is not enabled for this deployment."
            )
        return model_id
