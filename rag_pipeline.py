"""
rag_pipeline.py

Core retrieval-augmented generation pipeline for the Dharmic Intelligence Platform.

Components:
  - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (free, fast, runs on CPU)
  - Vector store: ChromaDB (in-memory for HuggingFace Spaces, persistent optional)
  - Generation model: google/flan-t5-large via HuggingFace Inference API (free tier)
  - Evaluation: custom RAGAS-inspired metrics (faithfulness, context relevance, answer relevance)

All models are free and open source. No paid API keys required.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent / "data"
EPISODES_PATH = DATA_DIR / "episodes.json"
CHARACTERS_PATH = DATA_DIR / "characters.json"

# Model config
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_ID = "google/flan-t5-large"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{GENERATION_MODEL_ID}"

# ChromaDB collection name
COLLECTION_NAME = "dharmic_episodes"


class RAGPipeline:
    """
    End-to-end retrieval-augmented generation pipeline.

    Retrieval: dense vector search over Mahabharata episodes using ChromaDB.
    Generation: Flan-T5-large via HuggingFace Inference API (no GPU needed).
    Evaluation: lightweight metrics computed locally without external dependencies.
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.collection = None
        self.episodes: list[dict] = []
        self.characters: dict[str, dict] = {}
        self.query_count = 0
        self.initialized = False

    def initialize(self) -> dict:
        """
        Load data, build embeddings, populate ChromaDB.
        Returns a status dict with counts and any warnings.
        """
        status = {"steps": [], "warnings": [], "success": False}

        # Step 1: load raw data
        try:
            with open(EPISODES_PATH, "r", encoding="utf-8") as f:
                self.episodes = json.load(f)
            with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
                self.characters = json.load(f)
            status["steps"].append(f"Loaded {len(self.episodes)} episodes and {len(self.characters)} characters.")
        except FileNotFoundError as e:
            status["warnings"].append(f"Data file missing: {e}")
            return status

        # Step 2: embedding model
        try:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            status["steps"].append(f"Embedding model ready: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            status["warnings"].append(f"Embedding model failed: {e}")
            return status

        # Step 3: ChromaDB in-memory client
        try:
            self.chroma_client = chromadb.Client()
            # Delete existing collection if it exists (for re-initialization)
            try:
                self.chroma_client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            status["steps"].append("ChromaDB in-memory vector store created.")
        except Exception as e:
            status["warnings"].append(f"ChromaDB setup failed: {e}")
            return status

        # Step 4: embed and index all episodes
        try:
            documents, metadatas, ids, embeddings = [], [], [], []
            for ep in self.episodes:
                # Combine the richest semantic content for indexing
                text = self._episode_to_index_text(ep)
                emb = self.embedding_model.encode(text, normalize_embeddings=True).tolist()
                documents.append(text)
                metadatas.append({"id": ep["id"], "title": ep["title"], "parva": ep["parva"]})
                ids.append(ep["id"])
                embeddings.append(emb)

            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            status["steps"].append(f"Indexed {len(self.episodes)} episodes into ChromaDB.")
        except Exception as e:
            status["warnings"].append(f"Indexing failed: {e}")
            return status

        # Step 5: verify HuggingFace API connectivity (soft check)
        if self.hf_token:
            status["steps"].append("HuggingFace token provided. Using authenticated inference.")
        else:
            status["steps"].append(
                "No HuggingFace token found. Using unauthenticated inference (rate-limited). "
                "Add HF_TOKEN for faster responses."
            )

        self.initialized = True
        status["success"] = True
        status["episode_count"] = len(self.episodes)
        status["character_count"] = len(self.characters)
        return status

    def _episode_to_index_text(self, ep: dict) -> str:
        """
        Create a single rich text string for embedding.
        This is what gets stored and searched in ChromaDB.
        """
        tags = " ".join(ep.get("tags", []))
        emotions = " ".join(ep.get("emotional_themes", []))
        return (
            f"{ep['title']}. "
            f"{ep['moral_conflict']}. "
            f"{ep['dharmic_principle']}. "
            f"{ep['key_insight']}. "
            f"{ep['modern_parallel']}. "
            f"Tags: {tags}. "
            f"Emotions: {emotions}."
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Dense retrieval from ChromaDB. Returns top_k episodes with similarity scores.
        """
        if not self.initialized or self.collection is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.episodes)),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results and results["ids"]:
            for i, ep_id in enumerate(results["ids"][0]):
                episode = next((ep for ep in self.episodes if ep["id"] == ep_id), None)
                if episode:
                    # ChromaDB cosine distance: 0 = identical, 2 = opposite
                    # Convert to similarity score 0-1
                    distance = results["distances"][0][i]
                    similarity = max(0.0, 1.0 - distance / 2.0)
                    retrieved.append({
                        "episode": episode,
                        "similarity_score": round(similarity, 4),
                        "rank": i + 1,
                    })
        return retrieved

    def generate_wisdom(
        self,
        user_query: str,
        character_id: str,
        retrieved_episodes: list[dict],
    ) -> dict:
        """
        Generate a wisdom response using Flan-T5 via HuggingFace Inference API.
        Falls back to a template-based response if the API is unavailable.
        """
        if not retrieved_episodes:
            return {
                "response": "No relevant episodes found for your query. Please try rephrasing.",
                "source": "fallback",
                "latency_ms": 0,
            }

        character = self.characters.get(character_id, self.characters["krishna"])
        best = retrieved_episodes[0]["episode"]

        prompt = self._build_prompt(user_query, character, best)
        start = time.time()
        generated_text, source = self._call_generation_api(prompt)
        latency_ms = int((time.time() - start) * 1000)

        self.query_count += 1

        return {
            "response": generated_text,
            "source": source,
            "prompt_used": prompt,
            "latency_ms": latency_ms,
            "character": character["name"],
            "primary_episode": best["title"],
        }

    def _build_prompt(self, query: str, character: dict, episode: dict) -> str:
        """
        Build a Flan-T5-compatible instruction prompt.
        Flan-T5 works best with explicit instruction framing and concise context.
        """
        prompt = (
            f"You are {character['name']} from the Mahabharata. "
            f"Your role is: {character['role']}. "
            f"Your teaching style: {character['speaking_style']} "
            f"Your moral focus: {character['moral_focus']}.\n\n"
            f"A person is facing this dilemma: {query}\n\n"
            f"Draw on this episode from the Mahabharata:\n"
            f"Episode: {episode['title']} ({episode['parva']})\n"
            f"What happened: {episode['narrative']}\n"
            f"The dharmic principle: {episode['dharmic_principle']}\n"
            f"The key insight: {episode['key_insight']}\n"
            f"Modern parallel: {episode['modern_parallel']}\n\n"
            f"Respond as {character['name']} speaking directly to this person. "
            f"Keep the response grounded in the episode. "
            f"Be specific, not generic. Do not quote Sanskrit unless translating it immediately. "
            f"Response (3-5 sentences, speak in first person as {character['name']}):"
        )
        return prompt

    def _call_generation_api(self, prompt: str) -> tuple[str, str]:
        """
        Call HuggingFace Inference API for Flan-T5-large.
        Returns (generated_text, source_label).
        """
        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.3,
                "length_penalty": 1.0,
            },
            "options": {
                "wait_for_model": True,
                "use_cache": False,
            },
        }

        try:
            response = requests.post(
                HF_INFERENCE_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "").strip()
                    # Flan-T5 sometimes echoes the prompt; strip it
                    text = self._clean_generated_text(text, prompt)
                    if text and len(text) > 20:
                        return text, "flan-t5-large"
            logger.warning("API returned status %d: %s", response.status_code, response.text[:200])
        except requests.exceptions.Timeout:
            logger.warning("HuggingFace API timed out. Using template fallback.")
        except Exception as e:
            logger.warning("HuggingFace API error: %s", e)

        # Fallback to template-based response
        return self._template_response(prompt), "template"

    def _clean_generated_text(self, text: str, prompt: str) -> str:
        """
        Remove prompt echoing from Flan-T5 output.
        """
        # If the output starts with the prompt, remove it
        if text.startswith(prompt[:50]):
            text = text[len(prompt):].strip()
        # Remove common artifacts
        text = re.sub(r"^(Response:|Answer:|Output:)\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _template_response(self, prompt: str) -> str:
        """
        Deterministic fallback when the API is unavailable.
        Produces character-specific responses from structured prompt content.
        """
        char_match = re.search(r"You are (\w+) from", prompt)
        char_name = char_match.group(1) if char_match else "the sage"

        ep_match = re.search(r"Episode: (.+?) \(", prompt)
        ep_title = ep_match.group(1) if ep_match else "this episode"

        insight_match = re.search(r"The key insight: (.+?)\n", prompt)
        insight = insight_match.group(1) if insight_match else ""

        principle_match = re.search(r"The dharmic principle: (.+?)\n", prompt)
        principle = principle_match.group(1) if principle_match else ""

        parallel_match = re.search(r"Modern parallel: (.+?)\n", prompt)
        parallel = parallel_match.group(1) if parallel_match else ""

        narrative_match = re.search(r"What happened: (.+?)\n", prompt)
        narrative = narrative_match.group(1) if narrative_match else ""

        character_openings = {
            "Krishna": f"Consider what the episode of {ep_title} reveals. {narrative} The principle at work here is: {principle} Act without attachment to outcome, but do not mistake inaction for wisdom. {insight}",
            "Arjuna": f"I know this weight you carry. I have stood in a similar place myself. The episode of {ep_title} taught me this: {insight} Your fear is real, but fear has never been a reason to abandon what you know is right. {principle}",
            "Yudhishthira": f"I speak from hard experience when I draw on {ep_title}. I have seen what happens when those who know the right thing stay silent. {principle} The cost of speaking may be high. The cost of silence compounds across time. {insight}",
            "Bhishma": f"From what I have lived and what I have lost, the episode of {ep_title} is directly relevant to you. {narrative} I stayed silent when I should have spoken. I kept vows when I should have broken them. Learn from that: {insight}",
            "Draupadi": f"Let me be direct with you. The episode of {ep_title} is not an abstraction. {principle} I asked a precise question in a court full of people who knew the answer. Not one spoke. That silence had consequences for everyone. {insight} Do not be the silence.",
            "Vidura": f"The situation you describe maps clearly onto {ep_title}. {principle} I told the king what would happen. He agreed with me. He did nothing. The obligation to speak is not contingent on whether you will be heard. {insight} That is the beginning and the end of it.",
        }

        response = character_openings.get(
            char_name,
            f"The episode of {ep_title} speaks to your situation. {principle} {insight}"
        )
        return response

    def query(self, user_input: str, character_id: str, top_k: int = 3) -> dict:
        """
        Full RAG pipeline: retrieve + generate + evaluate.
        Returns the complete result with metrics.
        """
        if not self.initialized:
            return {"error": "System not initialized. Please initialize first."}

        retrieved = self.retrieve(user_input, top_k=top_k)
        wisdom = self.generate_wisdom(user_input, character_id, retrieved)
        metrics = self.evaluate(user_input, wisdom["response"], retrieved)

        return {
            "query": user_input,
            "character_id": character_id,
            "retrieved_episodes": retrieved,
            "wisdom": wisdom,
            "evaluation_metrics": metrics,
        }

    def evaluate(self, query: str, response: str, retrieved_episodes: list[dict]) -> dict:
        """
        Compute lightweight RAG evaluation metrics locally (no external API needed).

        Metrics:
          - context_relevance: cosine similarity between query and retrieved context
          - answer_relevance: cosine similarity between response and query
          - faithfulness: lexical overlap between response and retrieved context
          - retrieval_precision: proportion of retrieved docs above similarity threshold
          - overall_score: weighted average of the above
        """
        if not self.initialized or not retrieved_episodes:
            return {}

        query_emb = self.embedding_model.encode(query, normalize_embeddings=True)
        response_emb = self.embedding_model.encode(response, normalize_embeddings=True)

        # Context relevance: how relevant are the retrieved episodes to the query
        context_texts = [r["episode"]["moral_conflict"] + " " + r["episode"]["key_insight"] for r in retrieved_episodes]
        context_embs = self.embedding_model.encode(context_texts, normalize_embeddings=True)
        context_relevance = float(np.mean([np.dot(query_emb, ce) for ce in context_embs]))
        context_relevance = max(0.0, min(1.0, context_relevance))

        # Answer relevance: does the response address the query
        answer_relevance = float(np.dot(response_emb, query_emb))
        answer_relevance = max(0.0, min(1.0, answer_relevance))

        # Faithfulness: lexical overlap between response tokens and all retrieved context
        response_tokens = set(response.lower().split())
        all_context = " ".join([
            r["episode"]["narrative"] + " " + r["episode"]["key_insight"] + " " + r["episode"]["dharmic_principle"]
            for r in retrieved_episodes
        ])
        context_tokens = set(all_context.lower().split())
        # Exclude stopwords for a cleaner signal
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                     "of", "with", "by", "is", "are", "was", "were", "be", "this", "that",
                     "it", "you", "i", "my", "your", "their", "his", "her", "we", "they"}
        response_content = response_tokens - stopwords
        context_content = context_tokens - stopwords
        if response_content:
            faithfulness = len(response_content & context_content) / len(response_content)
        else:
            faithfulness = 0.0
        faithfulness = max(0.0, min(1.0, faithfulness))

        # Retrieval precision: proportion of top results above similarity threshold 0.35
        threshold = 0.35
        above_threshold = sum(1 for r in retrieved_episodes if r["similarity_score"] >= threshold)
        retrieval_precision = above_threshold / max(len(retrieved_episodes), 1)

        # Overall: weighted composite
        overall = (
            0.30 * context_relevance
            + 0.30 * answer_relevance
            + 0.25 * faithfulness
            + 0.15 * retrieval_precision
        )

        return {
            "context_relevance": round(context_relevance, 3),
            "answer_relevance": round(answer_relevance, 3),
            "faithfulness": round(faithfulness, 3),
            "retrieval_precision": round(retrieval_precision, 3),
            "overall_score": round(overall, 3),
        }

    def get_all_episodes(self) -> list[dict]:
        return self.episodes

    def get_all_characters(self) -> dict:
        return self.characters

    def get_stats(self) -> dict:
        return {
            "initialized": self.initialized,
            "episode_count": len(self.episodes),
            "character_count": len(self.characters),
            "query_count": self.query_count,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "generation_model": GENERATION_MODEL_ID,
            "vector_store": "ChromaDB (in-memory)",
        }