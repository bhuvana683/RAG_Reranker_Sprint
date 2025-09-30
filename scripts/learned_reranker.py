# learned_reranker.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Callable, List, Dict

def train_learned_reranker(
    baseline_search_fn: Callable[[str, int], List[Dict]],
    keyword_search_fn: Callable[[str, int], List[Dict]]
) -> LogisticRegression:
    """
    Train a tiny logistic regression reranker using baseline and keyword scores.
    Pass the actual search functions from ask_api.py when calling this function.
    """

    questions = [
        "What are PPE safety requirements?",
        "How to safely operate a laser scanner?",
        "Define safety functions for machinery.",
        "Explain risk reduction steps for operators.",
        "What are type-C standards in ISO 13849-1?",
        "How to calculate performance level (PL) for a safety function?",
        "List hazards in industrial machinery.",
        "When should emergency stop be applied?"
    ]

    features = []
    labels = []

    for q in questions:
        baseline_results = baseline_search_fn(q, 5)
        keyword_results = keyword_search_fn(q, 5)

        combined = {}
        for r in baseline_results:
            key = r['text'][:100]
            combined[key] = {"pdf": r['pdf'], "text": r['text'], "vector_score": r['score'], "keyword_score": 0}

        for r in keyword_results:
            key = r['text'][:100]
            if key in combined:
                combined[key]['keyword_score'] = r['score']
            else:
                combined[key] = {"pdf": r['pdf'], "text": r['text'], "vector_score": 0, "keyword_score": r['score']}

        for v in combined.values():
            features.append([v['vector_score'], v['keyword_score']])
            # Placeholder label: consider chunks with vector_score >0.4 as relevant
            labels.append(1 if v['vector_score'] > 0.4 else 0)

    clf = LogisticRegression()
    clf.fit(np.array(features), np.array(labels))
    return clf
