"""
app.py
Dharmic Intelligence Platform - Gradio application for HuggingFace Spaces.
This is the entry point for deployment. HuggingFace Spaces looks for app.py
and expects it to expose a Gradio interface bound to a public port.
Architecture:
  - Gradio 5.x for the UI (Python 3.13 compatible)
  - RAGPipeline from rag_pipeline.py for retrieval and generation
  - All state managed in module-level variables (Spaces is single-process)
"""

import json
import os
import time
from pathlib import Path

import gradio as gr

from rag_pipeline import RAGPipeline

# Module-level pipeline instance shared across all Gradio calls
pipeline = RAGPipeline(hf_token=os.environ.get("HF_TOKEN", ""))

# Custom CSS using the same palette as the original design
# No emojis. No em dashes. Clean, professional typography.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Outfit:wght@300;400;500;600&display=swap');
:root {
    --golden: #C9A84C;
    --golden-light: #E2C97E;
    --golden-dim: rgba(201, 168, 76, 0.15);
    --golden-border: rgba(201, 168, 76, 0.35);
    --dark: #1A0F07;
    --dark-mid: #2B1A0E;
    --dark-card: #231408;
    --warm-surface: #1E1108;
    --cream: #F5EDD6;
    --cream-dim: #C8BFAA;
    --accent: #D97B3A;
    --accent-dim: rgba(217, 123, 58, 0.2);
    --success: #6BAF7A;
    --error: #C06B6B;
    --radius: 10px;
    --radius-lg: 16px;
}
/* Base reset for Gradio container */
body, .gradio-container {
    background: var(--dark) !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--cream) !important;
}
/* Remove Gradio default padding artifacts */
.gradio-container > .main {
    padding: 0 !important;
}
/* Header block */
.dip-header {
    background: linear-gradient(160deg, #2B1A0E 0%, #1A0F07 100%);
    border-bottom: 1px solid var(--golden-border);
    padding: 2.5rem 3rem 2rem;
    margin-bottom: 0;
}
.dip-header h1 {
    font-family: 'EB Garamond', serif !important;
    font-size: 2.4rem;
    font-weight: 600;
    color: var(--golden-light);
    letter-spacing: 0.01em;
    line-height: 1.15;
    margin: 0 0 0.4rem;
}
.dip-header p {
    font-size: 0.95rem;
    color: var(--cream-dim);
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0;
}
/* Section divider */
.dip-divider {
    height: 1px;
    background: var(--golden-border);
    margin: 0;
}
/* Label overrides */
label, .block label span {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--cream-dim) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
/* Input and textarea styling */
textarea, input[type="text"], input[type="password"] {
    background: var(--dark-card) !important;
    border: 1px solid var(--golden-border) !important;
    color: var(--cream) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    border-radius: var(--radius) !important;
    padding: 0.85rem 1rem !important;
    transition: border-color 0.2s ease;
    resize: vertical !important;
}
textarea:focus, input[type="text"]:focus, input[type="password"]:focus {
    border-color: var(--golden) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(201, 168, 76, 0.08) !important;
}
/* Radio buttons as character selector tiles */
.character-radio .wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.6rem !important;
}
.character-radio label.svelte-1gfkn6j {
    background: var(--dark-card) !important;
    border: 1px solid var(--golden-border) !important;
    border-radius: var(--radius) !important;
    padding: 0.7rem 1.1rem !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    flex: 1 !important;
    min-width: 120px !important;
    text-align: center !important;
}
.character-radio label.svelte-1gfkn6j:hover {
    border-color: var(--golden) !important;
    background: var(--golden-dim) !important;
}
.character-radio input[type="radio"]:checked + label,
.character-radio label:has(input:checked) {
    border-color: var(--golden) !important;
    background: var(--golden-dim) !important;
    color: var(--golden-light) !important;
}
/* Buttons */
button.primary {
    background: linear-gradient(135deg, var(--golden) 0%, #B8912E 100%) !important;
    color: var(--dark) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.75rem 1.8rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(201, 168, 76, 0.25) !important;
}
button.secondary {
    background: transparent !important;
    border: 1px solid var(--golden-border) !important;
    color: var(--cream-dim) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.18s ease !important;
}
button.secondary:hover {
    border-color: var(--golden) !important;
    color: var(--cream) !important;
}
/* Output/response areas */
.output-box {
    background: var(--dark-card) !important;
    border: 1px solid var(--golden-border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
    color: var(--cream) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
    min-height: 200px !important;
}
/* Metric cards row */
.metric-row {
    display: flex !important;
    gap: 0.75rem !important;
    flex-wrap: wrap !important;
}
.metric-card {
    background: var(--dark-card) !important;
    border: 1px solid var(--golden-border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    flex: 1 !important;
    min-width: 140px !important;
    text-align: center !important;
}
.metric-value {
    font-family: 'EB Garamond', serif !important;
    font-size: 1.6rem !important;
    color: var(--golden-light) !important;
    font-weight: 600 !important;
    display: block !important;
}
.metric-label {
    font-size: 0.72rem !important;
    color: var(--cream-dim) !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}
/* Episode cards */
.episode-card {
    background: var(--dark-card) !important;
    border: 1px solid var(--golden-border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.4rem !important;
    margin-bottom: 0.6rem !important;
    transition: border-color 0.18s ease !important;
}
.episode-card:hover {
    border-color: var(--golden) !important;
}
.episode-title {
    font-family: 'EB Garamond', serif !important;
    font-size: 1.1rem !important;
    color: var(--golden-light) !important;
    margin-bottom: 0.3rem !important;
}
.episode-meta {
    font-size: 0.78rem !important;
    color: var(--cream-dim) !important;
    margin-bottom: 0.6rem !important;
}
.episode-body {
    font-size: 0.9rem !important;
    color: var(--cream-dim) !important;
    line-height: 1.65 !important;
}
.episode-principle {
    border-left: 2px solid var(--golden) !important;
    padding-left: 0.8rem !important;
    margin-top: 0.6rem !important;
    color: var(--cream) !important;
    font-style: italic !important;
    font-size: 0.88rem !important;
}
/* Accordion / collapse */
.gr-accordion {
    background: var(--dark-card) !important;
    border: 1px solid var(--golden-border) !important;
    border-radius: var(--radius) !important;
}
/* Tabs */
.tab-nav button {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--cream-dim) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.2rem !important;
    transition: all 0.18s ease !important;
}
.tab-nav button.selected {
    color: var(--golden-light) !important;
    border-bottom-color: var(--golden) !important;
}
/* Status log */
.status-log {
    font-family: 'Courier New', monospace !important;
    font-size: 0.82rem !important;
    color: var(--cream-dim) !important;
    background: #0F0804 !important;
    border: 1px solid var(--golden-border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    line-height: 1.7 !important;
}
/* Score bar */
.score-bar-track {
    background: var(--dark-mid) !important;
    border-radius: 4px !important;
    height: 6px !important;
    overflow: hidden !important;
}
.score-bar-fill {
    height: 100% !important;
    background: linear-gradient(90deg, var(--accent), var(--golden)) !important;
    border-radius: 4px !important;
    transition: width 0.4s ease !important;
}
/* Slimmer Gradio blocks */
.gap, .contain {
    gap: 1rem !important;
}
.block {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}
/* Hide Gradio footer branding */
footer {
    display: none !important;
}
"""


# ============================================================
# HTML Rendering Helpers
# ============================================================

def render_wisdom_response(result: dict) -> str:
    """Render the RAG result as styled HTML for the Gradio output."""
    if "error" in result:
        return f'<div class="output-box" style="color:#C06B6B;">{result["error"]}</div>'

    wisdom = result.get("wisdom", {})
    retrieved = result.get("retrieved_episodes", [])
    metrics = result.get("evaluation_metrics", {})
    character_id = result.get("character_id", "krishna")

    from rag_pipeline import RAGPipeline
    from pathlib import Path
    import json
    chars = json.loads((Path("data/characters.json")).read_text())
    char = chars.get(character_id, {})
    char_name = char.get("name", character_id.title())
    char_role = char.get("role", "")

    response_text = wisdom.get("response", "No response generated.")
    source = wisdom.get("source", "unknown")
    latency = wisdom.get("latency_ms", 0)
    primary_ep = wisdom.get("primary_episode", "")
    overall = metrics.get("overall_score", 0.0)
    context_rel = metrics.get("context_relevance", 0.0)
    answer_rel = metrics.get("answer_relevance", 0.0)
    faithfulness = metrics.get("faithfulness", 0.0)

    bar_pct = int(overall * 100)
    src_label = "Flan-T5-Large via HuggingFace" if source == "flan-t5-large" else "Template (API unavailable)"

    retrieved_html = ""
    for r in retrieved[:3]:
        ep = r["episode"]
        sim = r["similarity_score"]
        sim_pct = int(sim * 100)
        retrieved_html += f"""
        <div class="episode-card">
            <div class="episode-title">{ep['title']}</div>
            <div class="episode-meta">{ep['parva']} &nbsp;|&nbsp; Similarity: {sim_pct}%</div>
            <div class="episode-body">{ep['key_insight']}</div>
            <div class="episode-principle">{ep['dharmic_principle']}</div>
        </div>
        """

    html = f"""
    <div style="display:flex;flex-direction:column;gap:1.4rem;">
      <div style="border-bottom:1px solid rgba(201,168,76,0.2);padding-bottom:1rem;">
        <div style="font-size:0.72rem;letter-spacing:0.09em;text-transform:uppercase;
                    color:#C8BFAA;margin-bottom:0.3rem;">Speaking as</div>
        <div style="font-family:'EB Garamond',serif;font-size:1.25rem;color:#E2C97E;
                    font-weight:600;">{char_name}</div>
        <div style="font-size:0.8rem;color:#9E9080;font-style:italic;">{char_role}</div>
      </div>
      <div style="font-family:'EB Garamond',serif;font-size:1.05rem;line-height:1.85;
                  color:#F5EDD6;">
        {response_text}
      </div>
      <div style="background:#0F0804;border:1px solid rgba(201,168,76,0.2);
                  border-radius:10px;padding:1.2rem;">
        <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;
                    color:#C8BFAA;margin-bottom:0.9rem;">Response Quality Metrics</div>
        <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:0.8rem;">
          <div><span style="color:#E2C97E;font-size:1.1rem;font-weight:600;">{int(context_rel*100)}%</span>
               <div style="font-size:0.7rem;color:#9E9080;text-transform:uppercase;letter-spacing:0.06em;">
               Context Relevance</div></div>
          <div><span style="color:#E2C97E;font-size:1.1rem;font-weight:600;">{int(answer_rel*100)}%</span>
               <div style="font-size:0.7rem;color:#9E9080;text-transform:uppercase;letter-spacing:0.06em;">
               Answer Relevance</div></div>
          <div><span style="color:#E2C97E;font-size:1.1rem;font-weight:600;">{int(faithfulness*100)}%</span>
               <div style="font-size:0.7rem;color:#9E9080;text-transform:uppercase;letter-spacing:0.06em;">
               Faithfulness</div></div>
          <div><span style="color:#E2C97E;font-size:1.1rem;font-weight:600;">{bar_pct}%</span>
               <div style="font-size:0.7rem;color:#9E9080;text-transform:uppercase;letter-spacing:0.06em;">
               Overall Score</div></div>
        </div>
        <div style="background:#1A0F07;border-radius:4px;height:5px;overflow:hidden;">
          <div style="width:{bar_pct}%;height:100%;
                      background:linear-gradient(90deg,#D97B3A,#C9A84C);
                      border-radius:4px;"></div>
        </div>
      </div>
      <div>
        <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;
                    color:#C8BFAA;margin-bottom:0.7rem;">Retrieved Episodes</div>
        {retrieved_html}
      </div>
      <div style="font-size:0.72rem;color:#5E5040;display:flex;gap:1.5rem;">
        <span>Generation: {src_label}</span>
        <span>Latency: {latency}ms</span>
      </div>
    </div>
    """
    return html


def render_episodes_html(episodes: list) -> str:
    """Render all episodes as HTML cards."""
    if not episodes:
        return "<p style='color:#C8BFAA;'>Initialize the system to load episodes.</p>"

    cards = ""
    for ep in episodes:
        chars = ", ".join(ep.get("characters", []))
        tags = " ".join([
            f'<span style="background:rgba(201,168,76,0.12);color:#C9A84C;'
            f'padding:2px 8px;border-radius:20px;font-size:0.72rem;'
            f'letter-spacing:0.05em;">{t}</span>'
            for t in ep.get("tags", [])[:6]
        ])
        cards += f"""
        <div class="episode-card" style="margin-bottom:1rem;">
          <div class="episode-title">{ep['title']}</div>
          <div class="episode-meta">{ep['parva']} &nbsp;|&nbsp; {ep.get('complexity','').title()}</div>
          <div class="episode-body" style="margin-bottom:0.5rem;">
            <strong style="color:#C8BFAA;font-size:0.78rem;text-transform:uppercase;
                           letter-spacing:0.05em;">Characters:</strong>
            <span style="color:#F5EDD6;"> {chars}</span>
          </div>
          <div class="episode-body">{ep['narrative'][:240]}...</div>
          <div class="episode-principle">{ep['key_insight']}</div>
          <div style="margin-top:0.8rem;display:flex;flex-wrap:wrap;gap:0.3rem;">{tags}</div>
        </div>
        """
    return cards


def render_analytics_html(stats: dict) -> str:
    """Render system analytics as HTML."""
    if not stats.get("initialized", False):
        return "<p style='color:#C8BFAA;'>System not yet initialized.</p>"

    items = [
        ("Episodes Indexed", stats.get("episode_count", 0)),
        ("Characters Profiled", stats.get("character_count", 0)),
        ("Wisdom Queries", stats.get("query_count", 0)),
        ("Vector Store", stats.get("vector_store", "ChromaDB")),
        ("Embedding Model", stats.get("embedding_model", "").split("/")[-1]),
        ("Generation Model", stats.get("generation_model", "").split("/")[-1]),
    ]
    cards = ""
    for label, value in items:
        cards += f"""
        <div style="background:#231408;border:1px solid rgba(201,168,76,0.25);
                    border-radius:10px;padding:1rem 1.2rem;flex:1;min-width:160px;">
          <div style="font-family:'EB Garamond',serif;font-size:1.5rem;
                      color:#E2C97E;font-weight:600;">{value}</div>
          <div style="font-size:0.72rem;color:#9E9080;text-transform:uppercase;
                      letter-spacing:0.06em;margin-top:0.2rem;">{label}</div>
        </div>
        """
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.75rem;">{cards}</div>'


# ============================================================
# Gradio Event Handlers
# ============================================================

def initialize_system(hf_token_input: str) -> tuple[str, str, str, str]:
    """
    Initialize the RAG pipeline. Returns updates for:
      status_html, episodes_html, analytics_html, wisdom_placeholder
    """
    global pipeline
    if hf_token_input.strip():
        pipeline.hf_token = hf_token_input.strip()

    lines = ["Initializing Dharmic Intelligence Platform...\n"]

    def add(line):
        lines.append(line)
        return "\n".join(lines)

    status = pipeline.initialize()

    if status.get("success"):
        for step in status.get("steps", []):
            lines.append(f"  {step}")
        lines.append("\nSystem ready. Navigate to the Wisdom tab to begin.")
        status_html = f'<div class="status-log">' + "\n".join(lines) + "</div>"
        episodes_html = render_episodes_html(pipeline.get_all_episodes())
        analytics_html = render_analytics_html(pipeline.get_stats())
        placeholder = ""
    else:
        for w in status.get("warnings", []):
            lines.append(f"  [WARNING] {w}")
        lines.append("\nInitialization failed. Check warnings above.")
        status_html = f'<div class="status-log" style="color:#C06B6B;">' + "\n".join(lines) + "</div>"
        episodes_html = "<p style='color:#C06B6B;'>Initialization failed.</p>"
        analytics_html = "<p style='color:#C06B6B;'>Initialization failed.</p>"
        placeholder = ""

    return status_html, episodes_html, analytics_html, placeholder


def seek_wisdom(query: str, character: str) -> str:
    """Run the RAG pipeline and return rendered HTML."""
    if not pipeline.initialized:
        return (
            '<div class="output-box" style="color:#C06B6B;">'
            "System is not initialized. Go to the Setup tab and click Initialize."
            "</div>"
        )

    if not query.strip():
        return (
            '<div class="output-box" style="color:#9E9080;">'
            "Please describe your dilemma in the text area above."
            "</div>"
        )

    # Map display name back to character ID
    name_to_id = {
        "Krishna": "krishna",
        "Arjuna": "arjuna",
        "Yudhishthira": "yudhishthira",
        "Bhishma": "bhishma",
        "Draupadi": "draupadi",
        "Vidura": "vidura",
    }
    character_id = name_to_id.get(character, "krishna")

    try:
        result = pipeline.query(query, character_id, top_k=3)
        return render_wisdom_response(result)
    except Exception as e:
        return (
            f'<div class="output-box" style="color:#C06B6B;">'
            f"An error occurred: {str(e)}"
            "</div>"
        )


def refresh_analytics() -> str:
    return render_analytics_html(pipeline.get_stats())


# ============================================================
# Gradio Interface
# ============================================================

HEADER_HTML = """
<div class="dip-header">
  <h1>Dharmic Intelligence Platform</h1>
  <p>Ancient wisdom for contemporary moral questions, powered by retrieval-augmented generation</p>
</div>
<div class="dip-divider"></div>
"""

WISDOM_PLACEHOLDER_HTML = """
<div style="display:flex;align-items:center;justify-content:center;min-height:300px;
            border:1px solid rgba(201,168,76,0.2);border-radius:16px;
            background:#231408;flex-direction:column;gap:1rem;padding:2rem;">
  <div style="font-family:'EB Garamond',serif;font-size:2rem;color:rgba(201,168,76,0.3);">
    Wisdom awaits
  </div>
  <div style="font-size:0.85rem;color:#5E5040;text-align:center;max-width:340px;
              line-height:1.7;">
    Describe your dilemma, select a guide, and submit to receive
    counsel drawn from the Mahabharata.
  </div>
  <div style="font-family:'EB Garamond',serif;font-size:1rem;color:rgba(201,168,76,0.25);
              font-style:italic;margin-top:0.5rem;text-align:center;">
    Dharmo rakshati rakshitah
    <br>
    <span style="font-size:0.78rem;letter-spacing:0.04em;">
    Dharma protects those who protect it.
    </span>
  </div>
</div>
"""

CHARACTER_CHOICES = ["Krishna", "Arjuna", "Yudhishthira", "Bhishma", "Draupadi", "Vidura"]

CHARACTER_DESCRIPTIONS = {
    "Krishna": "Divine guide. Strategic, paradoxical, radically pragmatic. Speaks to action, detachment, and duty.",
    "Arjuna": "Conflicted warrior. Emotionally honest. Speaks from inside confusion and learned courage.",
    "Yudhishthira": "The just king who made grave errors. Authority earned through suffering and regret.",
    "Bhishma": "Bound by his own vows. Wisdom of a man who understood what was right and failed to do it.",
    "Draupadi": "Voice of righteous anger. Uncompromising about dignity. Exposes the cost of silence.",
    "Vidura": "Counselor who spoke truth to power and was ignored. Calm, analytical, permanently ethical.",
}


def build_interface():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Dharmic Intelligence Platform",
        theme=gr.themes.Base(
            font=gr.themes.GoogleFont("Outfit"),
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ---- Tab 1: Setup ----
            with gr.Tab("System Setup"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="padding:0.5rem 0 1rem;">
                          <div style="font-family:'EB Garamond',serif;font-size:1.4rem;
                                      color:#E2C97E;margin-bottom:0.4rem;">
                            Initialize the Knowledge Base
                          </div>
                          <div style="font-size:0.88rem;color:#C8BFAA;line-height:1.6;">
                            The system indexes 15 Mahabharata episodes into ChromaDB using
                            all-MiniLM-L6-v2 embeddings. Wisdom generation uses Flan-T5-Large
                            via the HuggingFace Inference API. Providing an HF token reduces
                            rate-limiting but is not required.
                          </div>
                        </div>
                        """)
                        hf_token_input = gr.Textbox(
                            label="HuggingFace API Token (optional)",
                            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxx",
                            type="password",
                            info="Speeds up inference. Leave blank to use the free unauthenticated tier.",
                        )
                        init_btn = gr.Button("Initialize System", variant="primary")

                    with gr.Column(scale=2):
                        setup_status_html = gr.HTML(
                            value='<div class="status-log">Ready to initialize. Click the button to begin.</div>',
                            label="",
                        )

            # ---- Tab 2: Seek Wisdom ----
            with gr.Tab("Seek Wisdom"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        dilemma_input = gr.Textbox(
                            label="Describe your moral dilemma",
                            placeholder=(
                                "Describe the situation as specifically as you can.\n\n"
                                "Examples:\n"
                                "- I know my manager is acting unethically and I need to decide whether to report it.\n"
                                "- A promise I made years ago is now causing harm to people I love.\n"
                                "- My loyalty to a friend requires me to stay silent about something I know is wrong."
                            ),
                            lines=8,
                        )

                        gr.HTML("""
                        <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;
                                    color:#C8BFAA;margin:0.6rem 0 0.4rem;">Select your guide</div>
                        """)

                        character_radio = gr.Radio(
                            choices=CHARACTER_CHOICES,
                            value="Krishna",
                            label="",
                            elem_classes=["character-radio"],
                        )

                        character_desc = gr.HTML(
                            value=f'<div style="font-size:0.84rem;color:#9E9080;'
                                  f'font-style:italic;padding:0.4rem 0 0.8rem;line-height:1.6;">'
                                  f'{CHARACTER_DESCRIPTIONS["Krishna"]}</div>'
                        )

                        wisdom_btn = gr.Button("Seek Wisdom", variant="primary")

                    with gr.Column(scale=2):
                        wisdom_output = gr.HTML(
                            value=WISDOM_PLACEHOLDER_HTML,
                            label="",
                        )

                # Update character description on radio change
                def update_char_desc(char):
                    desc = CHARACTER_DESCRIPTIONS.get(char, "")
                    return (
                        f'<div style="font-size:0.84rem;color:#9E9080;font-style:italic;'
                        f'padding:0.4rem 0 0.8rem;line-height:1.6;">{desc}</div>'
                    )

                character_radio.change(
                    fn=update_char_desc,
                    inputs=[character_radio],
                    outputs=[character_desc],
                )

                wisdom_btn.click(
                    fn=seek_wisdom,
                    inputs=[dilemma_input, character_radio],
                    outputs=[wisdom_output],
                )

            # ---- Tab 3: Episodes ----
            with gr.Tab("Episode Library"):
                gr.HTML("""
                <div style="padding:0.5rem 0 1.2rem;">
                  <div style="font-family:'EB Garamond',serif;font-size:1.4rem;
                              color:#E2C97E;margin-bottom:0.3rem;">Mahabharata Knowledge Base</div>
                  <div style="font-size:0.88rem;color:#C8BFAA;">
                    15 carefully structured episodes with moral conflicts, dharmic principles,
                    key insights, and contemporary parallels. Initialize the system to load.
                  </div>
                </div>
                """)
                episodes_html = gr.HTML(
                    value="<p style='color:#9E9080;'>Initialize the system to view episodes.</p>"
                )

            # ---- Tab 4: Analytics ----
            with gr.Tab("Analytics"):
                gr.HTML("""
                <div style="padding:0.5rem 0 1.2rem;">
                  <div style="font-family:'EB Garamond',serif;font-size:1.4rem;
                              color:#E2C97E;margin-bottom:0.3rem;">System Analytics</div>
                  <div style="font-size:0.88rem;color:#C8BFAA;">
                    Live statistics for the RAG pipeline including index size,
                    query count, and model configuration.
                  </div>
                </div>
                """)
                analytics_html = gr.HTML(
                    value="<p style='color:#9E9080;'>Initialize the system to view analytics.</p>"
                )
                refresh_btn = gr.Button("Refresh", variant="secondary")
                refresh_btn.click(fn=refresh_analytics, outputs=[analytics_html])

        # Initialization wires up all tabs
        init_btn.click(
            fn=initialize_system,
            inputs=[hf_token_input],
            outputs=[setup_status_html, episodes_html, analytics_html, wisdom_output],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )