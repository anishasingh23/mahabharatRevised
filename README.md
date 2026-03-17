---
title: Dharmic Intelligence Platform
emoji: 🏛️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.9.0
app_file: app.py
pinned: false
license: mit
short_description: Mahabharata-grounded RAG for contemporary moral dilemmas
---


# Dharmic Intelligence Platform

DEMO -> https://aniisha-mahabharat1.hf.space/

A retrieval-augmented generation system that answers contemporary moral dilemmas
by surfacing relevant episodes from the Mahabharata and synthesizing character-voiced
guidance using a locally evaluated response quality framework.

## Architecture

| Component | Technology |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB (in-memory) |
| Generation model | google/flan-t5-large via HF Inference API |
| UI framework | Gradio 4.x |
| Evaluation | RAGAS-inspired local metrics |

## What it does

1. User describes a moral dilemma in plain language.
2. The query is embedded and used to retrieve the top-3 most semantically
   relevant Mahabharata episodes from a ChromaDB vector store.
3. A structured prompt is constructed with the retrieved episode context
   and the selected character's personality profile.
4. Flan-T5-Large generates a response via the HuggingFace Inference API.
5. Four evaluation metrics are computed locally: context relevance,
   answer relevance, faithfulness, and retrieval precision.

## Evaluation Metrics

The system computes the following metrics on every query with no external API calls:

- **Context Relevance**: cosine similarity between the query and retrieved passages.
- **Answer Relevance**: cosine similarity between the response and the query.
- **Faithfulness**: lexical overlap between the response and the retrieved context,
  approximating groundedness without an NLI model.
- **Retrieval Precision**: proportion of retrieved episodes above a similarity threshold.
- **NDCG@k**: normalized discounted cumulative gain for the ranked retrieval list.
- **Overall Score**: weighted composite of the above.

Run `python evaluation.py` offline to produce a full batch evaluation report.

## Characters

| Guide | Role |
|---|---|
| Krishna | Divine strategist. Duty, detachment, action without attachment. |
| Arjuna | Conflicted warrior. Honest about fear and confusion. |
| Yudhishthira | Just king with regrets. Pride, truth, and its limits. |
| Bhishma | Bound by vows. The cost of inflexibility. |
| Draupadi | Voice of righteous anger. Institutional failure and dignity. |
| Vidura | Counselor who was ignored. Speaking truth to power. |

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

To run the standalone evaluator:
```bash
python evaluation.py
```

## Environment Variables

| Variable | Purpose |
|---|---|
| HF_TOKEN | HuggingFace API token for faster inference (optional) |
