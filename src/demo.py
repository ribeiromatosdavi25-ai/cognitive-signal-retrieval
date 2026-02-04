"""
Demo script comparing cognitive signal retrieval vs baseline.

Runs test cases and shows how cognitive signal scoring outperforms
pure semantic similarity for context retrieval.
"""

import json
from signal_scorer import CognitiveSignalScorer
from baseline import BaselineScorer
from typing import List, Dict, Tuple


def load_test_cases(filepath: str = '../data/test_conversations.json') -> List[Dict]:
    """Load test cases from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['test_cases']


def print_separator(char: str = '=', length: int = 80):
    """Print a separator line."""
    print(char * length)


def print_results(
    test_case: Dict,
    cognitive_results: List[Tuple[Dict, float]],
    baseline_results: List[Tuple[Dict, float]],
    top_k: int = 3
):
    """
    Print comparison of cognitive vs baseline results.
    """
    print_separator()
    print(f"TEST CASE: {test_case['name']}")
    print(f"Description: {test_case['description']}")
    print(f"Query: '{test_case['query']}'")
    print_separator('-')
    
    expected_indices = test_case['expected_top_indices']
    print(f"\nEXPECTED TOP RESULTS (indices): {expected_indices}")
    
    print(f"\nCOGNITIVE SIGNAL RETRIEVAL (Top {top_k}):")
    for i, (fragment, score) in enumerate(cognitive_results[:top_k], 1):
        marker = "CHECK" if i-1 in expected_indices else " "
        print(f"{marker} {i}. [{score:.3f}] {fragment['text']}")
        print(f"   Type: {fragment['type']}, Stability: {fragment['stability']}")
    
    print(f"\nBASELINE (Semantic Only, Top {top_k}):")
    for i, (fragment, score) in enumerate(baseline_results[:top_k], 1):
        marker = "CHECK" if i-1 in expected_indices else " "
        print(f"{marker} {i}. [{score:.3f}] {fragment['text']}")
    
    cognitive_correct = sum(
        1 for i, _ in enumerate(cognitive_results[:top_k])
        if i in expected_indices
    )
    baseline_correct = sum(
        1 for i, _ in enumerate(baseline_results[:top_k])
        if i in expected_indices
    )
    
    print(f"\nACCURACY (top {top_k}):")
    print(f"   Cognitive Signal: {cognitive_correct}/{len(expected_indices)} correct")
    print(f"   Baseline:         {baseline_correct}/{len(expected_indices)} correct")
    print()


def calculate_metrics(
    results: List[Tuple[Dict, float]],
    expected_indices: List[int],
    k: int = 3
) -> Dict[str, float]:
    """Calculate precision and recall metrics."""
    top_k_correct = sum(
        1 for i in range(min(k, len(results)))
        if i in expected_indices
    )
    
    precision = top_k_correct / k if k > 0 else 0
    recall = top_k_correct / len(expected_indices) if expected_indices else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    }


def run_demo():
    """Run complete demo comparing both approaches."""
    print("\n" + "="*80)
    print("COGNITIVE SIGNAL RETRIEVAL - DEMO")
    print("="*80)
    print("\nInitializing models...")
    
    cognitive_scorer = CognitiveSignalScorer()
    baseline_scorer = BaselineScorer()
    
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases\n")
    
    all_cognitive_metrics = []
    all_baseline_metrics = []
    
    for test_case in test_cases:
        fragments = test_case['fragments']
        query = test_case['query']
        expected = test_case['expected_top_indices']
        
        cognitive_results = cognitive_scorer.rank_fragments(fragments, query)
        baseline_results = baseline_scorer.rank_fragments(fragments, query)
        
        print_results(test_case, cognitive_results, baseline_results)
        
        cognitive_metrics = calculate_metrics(cognitive_results, expected)
        baseline_metrics = calculate_metrics(baseline_results, expected)
        
        all_cognitive_metrics.append(cognitive_metrics)
        all_baseline_metrics.append(baseline_metrics)
    
    print_separator('=')
    print("OVERALL SUMMARY")
    print_separator('=')
    
    avg_cognitive_precision = sum(m['precision'] for m in all_cognitive_metrics) / len(all_cognitive_metrics)
    avg_baseline_precision = sum(m['precision'] for m in all_baseline_metrics) / len(all_baseline_metrics)
    
    avg_cognitive_recall = sum(m['recall'] for m in all_cognitive_metrics) / len(all_cognitive_metrics)
    avg_baseline_recall = sum(m['recall'] for m in all_baseline_metrics) / len(all_baseline_metrics)
    
    print(f"\nAverage Precision@3:")
    print(f"  Cognitive Signal: {avg_cognitive_precision:.2%}")
    print(f"  Baseline:         {avg_baseline_precision:.2%}")
    print(f"  Improvement:      {(avg_cognitive_precision - avg_baseline_precision):.2%}")
    
    print(f"\nAverage Recall@3:")
    print(f"  Cognitive Signal: {avg_cognitive_recall:.2%}")
    print(f"  Baseline:         {avg_baseline_recall:.2%}")
    print(f"  Improvement:      {(avg_cognitive_recall - avg_baseline_recall):.2%}")
    
    print("\n" + "="*80)
    print("Demo complete.")
    print("="*80 + "\n")


if __name__ == '__main__':
    run_demo()
