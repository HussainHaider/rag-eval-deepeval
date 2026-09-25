# eval_retriever_with_reranker.py
"""
The same retrieval eval as eval_retriever.py, pointed at the reranker.

Deliberately thin: it reuses run() so the golden set, the metrics, the judge and
the thresholds are identical on both sides. The retriever is the only thing that
changes, which is the only way a score delta can be attributed to reranking.
"""

from evals.eval_retriever import run
from evals.harness import print_summary
from src.reranker import CROSS_ENCODER, RerankingRetriever


def run_local():
    """Standalone convenience: build the reranking retriever, then run."""
    retriever = RerankingRetriever()
    return run(
        retriever,
        label=f"reranker_fetch{retriever.fetch_k}_top{retriever.top_k}",
        top_k=retriever.top_k,
        extra_hyperparameters={
            "fetch_k": retriever.fetch_k,
            "cross_encoder": CROSS_ENCODER,
        },
    )


if __name__ == "__main__":
    print_summary("retriever + reranker", run_local())
