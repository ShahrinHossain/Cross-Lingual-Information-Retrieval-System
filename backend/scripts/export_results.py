import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from clir.evaluation import Evaluator, load_qrels_jsonl, normalize_url

# Export detailed query-by-query results to CSV
def export_query_results_csv(qrels_items: List[Dict], all_results: List[Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'query',
            'rank',
            'doc_title',
            'doc_url',
            'doc_language',
            'score',
            'confidence',
            'is_relevant',
            'matched_keywords'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for qrel_item, result in zip(qrels_items, all_results):
            query = qrel_item.get('query', '')
            relevant_urls = {normalize_url(str(u).strip()) 
                           for u in qrel_item.get('relevant_urls', []) 
                           if str(u).strip()}
            
            for rank, doc in enumerate(result.ranked_documents, 1):
                is_relevant = normalize_url(doc.url) in relevant_urls
                matched_keywords = ', '.join(doc.matched_keywords[:5]) if doc.matched_keywords else ''
                
                writer.writerow({
                    'query': query,
                    'rank': rank,
                    'doc_title': doc.title,
                    'doc_url': doc.url,
                    'doc_language': doc.language,
                    'score': f"{doc.raw_score:.4f}",
                    'confidence': f"{doc.matching_confidence:.4f}",
                    'is_relevant': 'YES' if is_relevant else 'NO',
                    'matched_keywords': matched_keywords
                })
    
    print(f"✓ Query-by-query results exported to: {output_path}")

# Export summary metrics to CSV
def export_summary_csv(summary: Dict[str, float], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for metric, value in summary.items():
            writer.writerow({
                'Metric': metric,
                'Value': f"{value:.4f}"
            })
    
    print(f"✓ Summary metrics exported to: {output_path}")

# Export per-query metrics to CSV
def export_per_query_metrics_csv(qrels_items: List[Dict], all_results: List[Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'query',
            'num_relevant',
            'num_retrieved',
            'precision@10',
            'recall@50',
            'ndcg@10',
            'mrr',
            'ap'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for qrel_item, result in zip(qrels_items, all_results):
            query = qrel_item.get('query', '')
            relevant_urls = qrel_item.get('relevant_urls', [])
            
            writer.writerow({
                'query': query,
                'num_relevant': len(relevant_urls),
                'num_retrieved': len(result.ranked_documents),
                'precision@10': f"{result.precision_at_10:.4f}",
                'recall@50': f"{result.recall_at_50:.4f}",
                'ndcg@10': f"{result.ndcg_at_10:.4f}",
                'mrr': f"{result.mrr:.4f}",
                'ap': f"{getattr(result, 'average_precision', 0.0):.4f}"
            })
    
    print(f"✓ Per-query metrics exported to: {output_path}")

# Export all results to JSON
def export_json(qrels_items: List[Dict], all_results: List[Any], summary: Dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    json_output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'num_queries': len(qrels_items),
        },
        'summary': summary,
        'queries': []
    }
    
    for qrel_item, result in zip(qrels_items, all_results):
        query = qrel_item.get('query', '')
        relevant_urls = {normalize_url(str(u).strip()) 
                        for u in qrel_item.get('relevant_urls', []) 
                        if str(u).strip()}
        
        query_data = {
            'query': query,
            'num_relevant': len(relevant_urls),
            'metrics': {
                'precision@10': result.precision_at_10,
                'recall@50': result.recall_at_50,
                'ndcg@10': result.ndcg_at_10,
                'mrr': result.mrr,
                'average_precision': result.average_precision
            },
            'top_k_results': []
        }
        
        for rank, doc in enumerate(result.ranked_documents[:10], 1):
            is_relevant = normalize_url(doc.url) in relevant_urls
            query_data['top_k_results'].append({
                'rank': rank,
                'title': doc.title,
                'url': doc.url,
                'language': doc.language,
                'score': doc.raw_score,
                'confidence': doc.matching_confidence,
                'is_relevant': is_relevant,
                'matched_keywords': doc.matched_keywords[:5] if doc.matched_keywords else []
            })
        
        json_output['queries'].append(query_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON results exported to: {output_path}")

# Main entry point for exporting evaluation results
def main():
    parser = argparse.ArgumentParser(
        description="Export CLIR evaluation results to CSV and JSON formats"
    )
    parser.add_argument(
        "--qrels",
        type=str,
        required=True,
        help="Path to QRELS JSONL file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["bm25", "tfidf", "fuzzy", "semantic", "hybrid"],
        help="Retrieval model to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval/results",
        help="Output directory for results (default: data/eval/results)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K documents for ranking (default: 50)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "both"],
        default="both",
        help="Export format (default: both)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"CLIR Evaluation Results Exporter")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"QRELS: {args.qrels}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Format: {args.format}")
    print()
    
    # Load QRELS
    try:
        qrels_items = load_qrels_jsonl(args.qrels)
        print(f"✓ Loaded {len(qrels_items)} queries from QRELS")
    except Exception as e:
        print(f"✗ Error loading QRELS: {e}")
        return 1
    
    # Initialize evaluator
    print(f"\n⏳ Running evaluation on {len(qrels_items)} queries...")
    evaluator = Evaluator(
        model_name=args.model,
        top_k_for_ranking=args.top_k
    )
    
    try:
        all_results, summary = evaluator.evaluate_queries(qrels_items)
        print(f"✓ Evaluation complete!")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("Summary Metrics:")
    print(f"{'='*80}")
    for metric, value in summary.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Export results
    print(f"\n{'='*80}")
    print("Exporting Results...")
    print(f"{'='*80}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model
    
    if args.format in ["csv", "both"]:
        # Export query-by-query results
        query_results_path = os.path.join(
            args.output_dir, 
            f"{model_name}_query_results_{timestamp}.csv"
        )
        export_query_results_csv(qrels_items, all_results, query_results_path)
        
        # Export per-query metrics
        per_query_path = os.path.join(
            args.output_dir,
            f"{model_name}_per_query_metrics_{timestamp}.csv"
        )
        export_per_query_metrics_csv(qrels_items, all_results, per_query_path)
        
        # Export summary
        summary_path = os.path.join(
            args.output_dir,
            f"{model_name}_summary_{timestamp}.csv"
        )
        export_summary_csv(summary, summary_path)
    
    if args.format in ["json", "both"]:
        # Export JSON
        json_path = os.path.join(
            args.output_dir,
            f"{model_name}_results_{timestamp}.json"
        )
        export_json(qrels_items, all_results, summary, json_path)
    
    print(f"\n{'='*80}")
    print("✓ Export Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())