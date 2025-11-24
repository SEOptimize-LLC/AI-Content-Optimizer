# AI Content Optimizer

A Streamlit-first workflow that transforms legacy content into AI-overview-ready assets through five agent gates: Content Strategist, Chunk Optimizer, NLP Stylist, Authority Builder, and Metadata Optimizer.

## Features
- **Configurable profiles and modes** – select Blog, Product, Service, Thought Leadership, etc., plus Strict or Lite enforcement (see [`modules/config.py`](modules/config.py)).
- **Universal OpenRouter client** – call any of the requested OpenRouter models (GPT-5.1, Claude Sonnet 4.5, Gemini 3 Pro, etc.) through [`utils/llm_handler.py`](utils/llm_handler.py).
- **Layered agent pipeline** – each stage enforces pass/fail gates and produces actionable feedback with optimized text snippets (see [`modules/`](modules)).
- **Metadata/schema finishing pass** – titles, meta descriptions, and FAQ schema stay in sync with the optimized copy.

## Prerequisites
- Python 3.11
- Streamlit Cloud (or local Streamlit install)
- OpenRouter API key

## Installation
```bash
cd "Marketing/Roger SEO/Scripts/AI_Content_Optimizer"
pip install -r requirements.txt
```

## Secrets (Streamlit Cloud or local `.streamlit/secrets.toml`)
```toml
OPENROUTER_API_KEY = "sk-or-..."
```

## Running Locally
```bash
streamlit run app.py
```

## Deployment Checklist
1. Push this folder to GitHub.
2. In Streamlit Cloud, point to `app.py` and set `OPENROUTER_API_KEY` in *Secrets*.
3. (Optional) expose additional metadata defaults via `st.secrets` if desired.

## Extending
- Improve Markdown parsing by enhancing `parse_blocks` in [`app.py`](app.py) to capture lists, FAQs, and metadata markers from the raw draft.
- Customize rule thresholds in [`modules/config.py`](modules/config.py) to match your templates.
- Swap or add OpenRouter models by editing the catalog in [`modules/config.py`](modules/config.py#L23).
