"""
evaluation.py

Standalone evaluation module for the Dharmic Intelligence RAG pipeline.

Implements RAGAS-inspired metrics computed entirely locally with no external
API calls, making it suitable for CI pipelines and offline use.

Metrics implemented:
  1. context_relevance       : How relevant are retrieved passages to the query?
  2. answer_relevance        : Does the generated answer address the query?
  3. faithfulness            : Is the answer grounded in the retrieved context?
  4. retrieval_precision     : What proportion of retrieved docs are truly relevant?
  5. retrieval_recall_proxy  : Estimated coverage of relevant information in retrieval.
  6. ndcg_at_k               : Normalized discounted cumulative gain for ranked results.
  7. overall_score           : Weighted composite of the above.

Usage:
    from evaluation import DharmicEvaluator
    evaluator = DharmicEvaluator(embedding_model)
    metrics = evaluator.evaluate(query, response, retrieved_episodes, ground_truth_ids)
    report = evaluator.batch_evaluate(test_cases)
    evaluator.print_report(report)
"""

import json
import math
import statistics
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# Stopwords to exclude from lexical overlap calculations
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "it", "you", "i", "my", "your", "their", "his", "her",
    "we", "they", "from", "as", "has", "have", "had", "not", "so", "if",
    "its", "our", "what", "who", "which", "when", "then", "than", "also",
    "all", "any", "both", "each", "few", "more", "most", "no", "only",
    "same", "such", "too", "very", "just", "even", "can", "could", "would",
    "should", "will", "may", "might", "must", "do", "does", "did", "about",
})


class DharmicEvaluator:
    """
    Evaluator for the Dharmic Intelligence RAG pipeline.

    All semantic metrics use the same sentence-transformers model used during
    retrieval, so no additional models are needed.
    """

    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        # Similarity threshold above which a retrieved doc is considered relevant
        self.relevance_threshold = 0.35
        # Weights for composite scoring
        self.weights = {
            "context_relevance": 0.25,
            "answer_relevance": 0.25,
            "faithfulness": 0.25,
            "retrieval_precision": 0.15,
            "ndcg": 0.10,
        }

    def evaluate(
        self,
        query: str,
        response: str,
        retrieved_episodes: list[dict],
        ground_truth_ids: Optional[list[str]] = None,
    ) -> dict:
        """
        Compute all metrics for a single query-response pair.

        Args:
            query: The user's input dilemma.
            response: The generated wisdom text.
            retrieved_episodes: List of dicts from RAGPipeline.retrieve().
                Each dict has keys: episode (dict), similarity_score (float), rank (int).
            ground_truth_ids: Optional list of episode IDs that are known relevant.
                If not provided, precision-based metrics use the similarity threshold.

        Returns:
            Dict of metric names to float values, plus an overall_score.
        """
        if not retrieved_episodes:
            return self._empty_metrics()

        # Compute embeddings
        query_emb = self.embedding_model.encode(query, normalize_embeddings=True)
        response_emb = self.embedding_model.encode(response, normalize_embeddings=True)

        context_texts = [
            self._episode_to_context_text(r["episode"])
            for r in retrieved_episodes
        ]
        context_embs = self.embedding_model.encode(context_texts, normalize_embeddings=True)

        metrics = {}

        # 1. Context relevance
        metrics["context_relevance"] = self._context_relevance(query_emb, context_embs)

        # 2. Answer relevance
        metrics["answer_relevance"] = self._answer_relevance(query_emb, response_emb)

        # 3. Faithfulness
        all_context = self._build_full_context(retrieved_episodes)
        metrics["faithfulness"] = self._faithfulness(response, all_context)

        # 4. Retrieval precision
        metrics["retrieval_precision"] = self._retrieval_precision(
            retrieved_episodes, ground_truth_ids
        )

        # 5. Retrieval recall proxy (only meaningful with ground truth)
        if ground_truth_ids:
            metrics["retrieval_recall"] = self._retrieval_recall(
                retrieved_episodes, ground_truth_ids
            )
        else:
            metrics["retrieval_recall"] = None

        # 6. NDCG@k
        metrics["ndcg_at_k"] = self._ndcg_at_k(retrieved_episodes, ground_truth_ids)

        # 7. Overall score
        metrics["overall_score"] = self._compute_overall(metrics)

        # Round all float values
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

    def batch_evaluate(self, test_cases: list[dict]) -> dict:
        """
        Evaluate a list of test cases and aggregate results.

        Each test case dict must have:
            query (str), response (str), retrieved_episodes (list),
            ground_truth_ids (list, optional).

        Returns a summary dict with per-metric means and standard deviations.
        """
        all_metrics = []
        for tc in test_cases:
            m = self.evaluate(
                tc["query"],
                tc["response"],
                tc["retrieved_episodes"],
                tc.get("ground_truth_ids"),
            )
            all_metrics.append(m)

        return self._aggregate(all_metrics)

    def _context_relevance(
        self, query_emb: np.ndarray, context_embs: np.ndarray
    ) -> float:
        """
        Mean cosine similarity between the query and each retrieved context.
        Higher means retrieved passages are more related to the query.
        """
        scores = [float(np.dot(query_emb, ce)) for ce in context_embs]
        score = statistics.mean(scores) if scores else 0.0
        return max(0.0, min(1.0, score))

    def _answer_relevance(
        self, query_emb: np.ndarray, response_emb: np.ndarray
    ) -> float:
        """
        Cosine similarity between the query and the generated response.
        Higher means the response is more semantically aligned with what was asked.
        """
        score = float(np.dot(query_emb, response_emb))
        return max(0.0, min(1.0, score))

    def _faithfulness(self, response: str, all_context: str) -> float:
        """
        Lexical overlap between response content words and retrieved context.
        This approximates whether the response is grounded in retrieved material
        rather than hallucinated.

        Note: pure lexical overlap is a lower bound on faithfulness. With a more
        powerful NLI model this could be computed semantically. This implementation
        is deliberately lightweight for zero-cost inference.
        """
        response_tokens = self._content_tokens(response)
        context_tokens = self._content_tokens(all_context)

        if not response_tokens:
            return 0.0

        overlap = len(response_tokens & context_tokens)
        score = overlap / len(response_tokens)
        return max(0.0, min(1.0, score))

    def _retrieval_precision(
        self, retrieved_episodes: list[dict], ground_truth_ids: Optional[list[str]]
    ) -> float:
        """
        Proportion of retrieved episodes that are relevant.
        If ground truth is provided, uses exact ID matching.
        Otherwise falls back to similarity threshold.
        """
        if not retrieved_episodes:
            return 0.0

        if ground_truth_ids:
            relevant_count = sum(
                1 for r in retrieved_episodes
                if r["episode"]["id"] in ground_truth_ids
            )
        else:
            relevant_count = sum(
                1 for r in retrieved_episodes
                if r["similarity_score"] >= self.relevance_threshold
            )

        return relevant_count / len(retrieved_episodes)

    def _retrieval_recall(
        self, retrieved_episodes: list[dict], ground_truth_ids: list[str]
    ) -> float:
        """
        Proportion of known-relevant episodes that were actually retrieved.
        Requires ground truth IDs.
        """
        if not ground_truth_ids:
            return 0.0

        retrieved_ids = {r["episode"]["id"] for r in retrieved_episodes}
        found = len(retrieved_ids & set(ground_truth_ids))
        return found / len(ground_truth_ids)

    def _ndcg_at_k(
        self, retrieved_episodes: list[dict], ground_truth_ids: Optional[list[str]]
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K.

        If ground truth is provided: binary relevance (1 if in ground truth, 0 if not).
        Otherwise: relevance grade = similarity_score clipped to [0, 1].

        NDCG measures whether the most relevant results appear at the top of the ranking.
        """
        if not retrieved_episodes:
            return 0.0

        if ground_truth_ids:
            relevances = [
                1.0 if r["episode"]["id"] in ground_truth_ids else 0.0
                for r in retrieved_episodes
            ]
        else:
            relevances = [r["similarity_score"] for r in retrieved_episodes]

        dcg = self._dcg(relevances)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self._dcg(ideal_relevances)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def _dcg(self, relevances: list[float]) -> float:
        """Discounted Cumulative Gain."""
        return sum(
            rel / math.log2(rank + 2)
            for rank, rel in enumerate(relevances)
        )

    def _compute_overall(self, metrics: dict) -> float:
        """Weighted composite score from individual metrics."""
        score = (
            self.weights["context_relevance"] * metrics.get("context_relevance", 0.0)
            + self.weights["answer_relevance"] * metrics.get("answer_relevance", 0.0)
            + self.weights["faithfulness"] * metrics.get("faithfulness", 0.0)
            + self.weights["retrieval_precision"] * metrics.get("retrieval_precision", 0.0)
            + self.weights["ndcg"] * metrics.get("ndcg_at_k", 0.0)
        )
        return max(0.0, min(1.0, score))

    def _aggregate(self, all_metrics: list[dict]) -> dict:
        """Compute mean and std for each metric across a batch."""
        if not all_metrics:
            return {}

        keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        summary = {}
        for key in keys:
            values = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
            summary[key] = {
                "mean": round(statistics.mean(values), 4) if values else 0.0,
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4) if values else 0.0,
                "max": round(max(values), 4) if values else 0.0,
                "n": len(values),
            }
        return summary

    def print_report(self, report: dict) -> None:
        """Print a formatted evaluation report to stdout."""
        print("\n" + "=" * 60)
        print("DHARMIC INTELLIGENCE RAG EVALUATION REPORT")
        print("=" * 60)
        for metric, stats in report.items():
            print(f"\n{metric.upper().replace('_', ' ')}")
            print(f"  Mean  : {stats['mean']:.4f}")
            print(f"  Std   : {stats['std']:.4f}")
            print(f"  Min   : {stats['min']:.4f}")
            print(f"  Max   : {stats['max']:.4f}")
            print(f"  N     : {stats['n']}")
        print("\n" + "=" * 60)

    def _episode_to_context_text(self, episode: dict) -> str:
        """Convert an episode dict to the same text used for embedding at index time."""
        return (
            f"{episode['moral_conflict']}. "
            f"{episode['dharmic_principle']}. "
            f"{episode['key_insight']}. "
            f"{episode['modern_parallel']}."
        )

    def _build_full_context(self, retrieved_episodes: list[dict]) -> str:
        """Concatenate all retrieved episode text for faithfulness computation."""
        parts = []
        for r in retrieved_episodes:
            ep = r["episode"]
            parts.append(
                ep["narrative"] + " " +
                ep["dharmic_principle"] + " " +
                ep["key_insight"] + " " +
                ep.get("resolution", "")
            )
        return " ".join(parts)

    def _content_tokens(self, text: str) -> set:
        """Tokenize text and remove stopwords and short tokens."""
        tokens = text.lower().split()
        return {
            t.strip(".,;:!?\"'()[]")
            for t in tokens
            if len(t) > 3 and t not in STOPWORDS
        }

    def _empty_metrics(self) -> dict:
        return {
            "context_relevance": 0.0,
            "answer_relevance": 0.0,
            "faithfulness": 0.0,
            "retrieval_precision": 0.0,
            "retrieval_recall": None,
            "ndcg_at_k": 0.0,
            "overall_score": 0.0,
        }


# Test cases for offline validation
SAMPLE_TEST_CASES = [
    {
        "query": "I feel I must speak up at work about what my manager is doing but I know it will hurt my career",
        "ground_truth_ids": ["ep_003", "ep_012"],
        "description": "Whistleblowing and speaking truth to power",
    },
    {
        "query": "I made a promise to my family long ago and now keeping it is causing harm to people I love",
        "ground_truth_ids": ["ep_004", "ep_005"],
        "description": "Rigid vows and their unintended consequences",
    },
    {
        "query": "I know my friend is doing something wrong but they stood by me when no one else did",
        "ground_truth_ids": ["ep_005"],
        "description": "Loyalty to an unjust cause",
    },
    {
        "query": "I stayed silent when I should have spoken and now I regret it deeply",
        "ground_truth_ids": ["ep_003", "ep_002"],
        "description": "Bystander guilt and the cost of silence",
    },
    {
        "query": "I am considering a compromise that requires me to lie in order to save people from harm",
        "ground_truth_ids": ["ep_009"],
        "description": "Necessary deception in leadership",
    },
]


if __name__ == "__main__":
    # Standalone evaluation script for CI or offline testing
    import sys

    print("Loading embedding model for evaluation...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    evaluator = DharmicEvaluator(model)

    print("Loading RAG pipeline...")
    sys.path.insert(0, str(Path(__file__).parent))
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    status = pipeline.initialize()
    if not status["success"]:
        print("Pipeline initialization failed:", status["warnings"])
        sys.exit(1)

    print(f"\nRunning evaluation on {len(SAMPLE_TEST_CASES)} test cases...")
    batch_inputs = []
    for tc in SAMPLE_TEST_CASES:
        print(f"  Processing: {tc['description']}")
        result = pipeline.query(tc["query"], "krishna", top_k=3)
        batch_inputs.append({
            "query": tc["query"],
            "response": result["wisdom"]["response"],
            "retrieved_episodes": result["retrieved_episodes"],
            "ground_truth_ids": tc["ground_truth_ids"],
        })

    report = evaluator.batch_evaluate(batch_inputs)
    evaluator.print_report(report)

    # Save report to JSON
    output_path = Path(__file__).parent / "evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_path}")
