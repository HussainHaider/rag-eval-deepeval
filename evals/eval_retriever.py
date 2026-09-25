# eval_retriever.py
"""
Retrieval eval: how good is the context we hand the generator?

`run()` is the shared body -- it takes the retriever as an argument, so the same
evaluation can be pointed at the base retriever (here) or at the reranker
(eval_retriever_with_reranker.py) with nothing else changed. That is what makes
the two runs comparable.
"""

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src import retriever as retriever_config
from src.retriever import build_retriever
from evals.harness import (
    JUDGE_MODEL,
    THRESHOLD,
    load_goldens,
    summarize_by_metric,
    print_summary,
)

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"


def run(retriever, label, top_k, extra_hyperparameters=None):
    """Evaluate `retriever` against the golden set. `label` and `top_k` describe
    the retriever under test; everything else is read from src/retriever.py so
    the report cannot claim a config the run did not use."""

    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                # neither metric below judges actual_output -- both score
                # retrieval_context against input/expected_output -- but the
                # field is required, so say plainly why it is a placeholder.
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": label,
            "embedding_model": retriever_config.EMBEDDING_MODEL,
            "chunk_size": retriever_config.CHUNK_SIZE,
            "chunk_overlap": retriever_config.CHUNK_OVERLAP,
            "top_k": top_k,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
            **(extra_hyperparameters or {}),
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the base retriever, then run."""
    return run(
        build_retriever(),
        label=f"base_k{retriever_config.TOP_K}",
        top_k=retriever_config.TOP_K,
    )


if __name__ == "__main__":
    print_summary("retriever (base)", run_local())
