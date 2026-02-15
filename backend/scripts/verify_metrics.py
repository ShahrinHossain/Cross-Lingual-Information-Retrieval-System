"""
Verify Evaluation Metrics Against Target Thresholds

Checks if metrics meet targets: Precision@10 >= 0.6, Recall@50 >= 0.5,
nDCG@10 >= 0.5, MRR >= 0.4
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from clir.evaluation import (
    Evaluator, 
    load_qrels_jsonl, 
    print_summary,
    normalize_url
)


# Target thresholds
TARGETS = {
    "Precision@10": 0.6,
    "Recall@50": 0.5,
    "nDCG@10": 0.5,
    "MRR": 0.4,
}


def verify_metrics(summary: dict) -> Tuple[bool, List[str]]:
    passed = True
    issues = []
    
    for metric_name, target_value in TARGETS.items():
        actual_value = summary.get(metric_name, 0.0)
        
        if actual_value < target_value:
            passed = False
            issues.append(
                f"❌ FAIL {metric_name}: {actual_value:.3f} < {target_value:.3f}"
            )
        else:
            issues.append(
                f"✅ PASS {metric_name}: {actual_value:.3f} >= {target_value:.3f}"
            )
    
    return passed, issues


def print_debug_url_matching(qrels_items: List[Dict[str, Any]], all_results: List[Any]) -> None:
    """Print detailed debug information about URL matching"""
    print("\n" + "="*90)
    print("🔍 DEBUG: URL MATCHING ANALYSIS")
    print("="*90)
    
    total_queries = len(qrels_items)
    queries_with_matches = 0
    total_matches = 0
    total_expected = 0
    
    for i, (qrel_item, result) in enumerate(zip(qrels_items, all_results), 1):
        query = qrel_item.get("query", "")
        relevant_urls_raw = qrel_item.get("relevant_urls", [])
        relevant_urls = {normalize_url(str(u).strip()) for u in relevant_urls_raw if str(u).strip()}
        ranked_urls = [normalize_url(doc.url) for doc in result.ranked_documents if doc.url]
        
        print(f"\n📝 Query {i}/{total_queries}: {query}")
        print(f"   Expected {len(relevant_urls)} relevant URLs")
        
        # Show expected URLs
        if relevant_urls:
            print(f"   📌 Expected relevant URLs:")
            for j, url in enumerate(list(relevant_urls)[:5], 1):  # Show first 5
                print(f"      {j}. {url}")
            if len(relevant_urls) > 5:
                print(f"      ... and {len(relevant_urls) - 5} more")
        else:
            print(f"   ⚠️  WARNING: No relevant URLs in QRELS for this query!")
        
        # Show retrieved URLs
        print(f"   🔎 Retrieved {len(ranked_urls)} URLs (showing top 10):")
        matches_found = []
        for j, url in enumerate(ranked_urls[:10], 1):
            if url in relevant_urls:
                print(f"      {j}. ✅ MATCH - {url}")
                matches_found.append(url)
            else:
                print(f"      {j}. ❌ NO MATCH - {url}")
        
        if len(ranked_urls) > 10:
            remaining_matches = sum(1 for url in ranked_urls[10:] if url in relevant_urls)
            if remaining_matches > 0:
                print(f"      ... and {remaining_matches} more matches in remaining results")
                matches_found.extend([url for url in ranked_urls[10:] if url in relevant_urls])
        
        # Summary for this query
        match_count = len(matches_found)
        if match_count > 0:
            queries_with_matches += 1
        total_matches += match_count
        total_expected += len(relevant_urls)
        
        print(f"   📊 Matches: {match_count}/{len(ranked_urls[:10])} in top 10")
        print(f"   📊 P@10: {result.precision_at_10:.3f}, R@50: {result.recall_at_50:.3f}, nDCG@10: {result.ndcg_at_10:.3f}, MRR: {result.mrr:.3f}")
        
        # Show why there might be no matches
        if match_count == 0 and relevant_urls and ranked_urls:
            print(f"   ⚠️  NO MATCHES FOUND - Possible reasons:")
            print(f"      1. URLs in dataset don't match URLs in QRELS")
            print(f"      2. Retrieval model isn't finding the right documents")
            print(f"      3. QRELS might contain URLs not in your dataset")
            
            # Compare first URLs to help diagnose
            if len(relevant_urls) > 0 and len(ranked_urls) > 0:
                sample_expected = list(relevant_urls)[0]
                sample_retrieved = ranked_urls[0]
                print(f"      📋 Example comparison:")
                print(f"         Expected:  {sample_expected}")
                print(f"         Retrieved: {sample_retrieved}")
    
    # Overall summary
    print("\n" + "="*90)
    print("📊 OVERALL URL MATCHING SUMMARY")
    print("="*90)
    print(f"Total queries: {total_queries}")
    print(f"Queries with at least 1 match: {queries_with_matches} ({100*queries_with_matches/total_queries if total_queries > 0 else 0:.1f}%)")
    print(f"Total matches found: {total_matches}")
    print(f"Total relevant URLs expected: {total_expected}")
    print(f"Match rate: {100*total_matches/total_expected if total_expected > 0 else 0:.1f}%")
    
    if queries_with_matches == 0:
        print("\n⚠️  CRITICAL: NO MATCHES FOUND IN ANY QUERY!")
        print("This explains why all metrics are 0.0")
        print("\nTroubleshooting steps:")
        print("1. Check if URLs in QRELS match URLs in your dataset files (bn.jsonl, en.jsonl)")
        print("2. Verify your dataset files contain the expected documents")
        print("3. Check if retrieval models are working (try --demo_query first)")
        print("4. Ensure QRELS file is properly formatted")
    
    print("="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify CLIR evaluation metrics against target thresholds"
    )
    parser.add_argument(
        "--qrels",
        type=str,
        required=True,
        help="Path to QRELS JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["bm25", "tfidf", "fuzzy", "semantic", "hybrid"],
        help="Retrieval model to evaluate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K documents for ranking (default: 10)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug output showing URL matching",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*90}")
    print(f"🎯 CLIR Metrics Verification - Model: {args.model}")
    print(f"{'='*90}")
    print(f"QRELS: {args.qrels}")
    print("\n📋 Target Thresholds:")
    for metric, target in TARGETS.items():
        print(f"  {metric}: >= {target:.2f}")
    print()
    
    # Load QRELS
    try:
        qrels_items = load_qrels_jsonl(args.qrels)
        print(f"✅ Loaded {len(qrels_items)} queries from QRELS file")
        
        if len(qrels_items) == 0:
            print("❌ ERROR: QRELS file is empty!")
            return 1
            
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print(f"\nMake sure the QRELS file exists at: {args.qrels}")
        return 1
    except Exception as e:
        print(f"❌ ERROR loading QRELS: {e}")
        return 1
    
    # Create evaluator
    print(f"\n⚙️  Initializing evaluator with model: {args.model}")
    evaluator = Evaluator(
        model_name=args.model,
        top_k_for_ranking=args.top_k,
    )
    
    # Run evaluation
    print(f"🔄 Running evaluation on {len(qrels_items)} queries...")
    print("   (This may take a while for semantic/hybrid models...)\n")
    
    try:
        all_results, summary = evaluator.evaluate_queries(qrels_items)
    except Exception as e:
        print(f"❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print summary
    print_summary(summary)
    
    # Debug mode - show detailed URL matching
    if args.debug:
        print_debug_url_matching(qrels_items, all_results)
    
    # Verify against targets
    passed, issues = verify_metrics(summary)
    
    print("\n" + "="*90)
    print("📊 VERIFICATION RESULTS")
    print("="*90)
    for issue in issues:
        print(issue)
    
    print("\n" + "="*90)
    if passed:
        print("✅ SUCCESS: ALL TARGETS MET!")
        print("="*90)
        return 0
    else:
        print("❌ FAILURE: SOME TARGETS NOT MET")
        print("="*90)
        
        # Check if all metrics are exactly 0
        all_zero = all(summary.get(metric, 0.0) == 0.0 for metric in TARGETS.keys())
        
        if all_zero:
            print("\n⚠️  ALL METRICS ARE 0.0 - This indicates a fundamental problem:")
            print("\n🔧 TROUBLESHOOTING STEPS:")
            print("   1. Run with --debug flag to see URL matching details:")
            print(f"      python verify_metrics.py --qrels {args.qrels} --model {args.model} --debug")
            print("\n   2. Check if your QRELS URLs match your dataset URLs:")
            print("      - Open your QRELS file and note a few URLs")
            print("      - Search for those URLs in bn.jsonl and en.jsonl")
            print("      - URLs must match exactly (after normalization)")
            print("\n   3. Test retrieval is working:")
            print(f"      cd backend")
            print(f"      python -m clir.evaluation --demo_query \"your test query\" --model {args.model}")
            print("\n   4. Verify dataset files exist and contain data:")
            print("      backend/data/processed/bn.jsonl")
            print("      backend/data/processed/en.jsonl")
        else:
            print("\n💡 RECOMMENDATIONS:")
            print("   1. Review query processing and translation quality")
            print("   2. Improve retrieval models (especially semantic matching)")
            print("   3. Tune hybrid model weights")
            print("   4. Expand and improve the dataset")
            print("   5. Review QRELS labels for accuracy")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())