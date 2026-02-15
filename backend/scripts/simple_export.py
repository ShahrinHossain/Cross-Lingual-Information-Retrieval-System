import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from clir.evaluation import Evaluator, load_qrels_jsonl

def main():
    parser = argparse.ArgumentParser(description="Simple CSV/JSON Export for CLIR Results")
    parser.add_argument("--qrels", type=str, required=True, help="Path to QRELS file")
    parser.add_argument("--model", type=str, default="hybrid", 
                       choices=["bm25", "tfidf", "fuzzy", "semantic", "hybrid"])
    parser.add_argument("--output", type=str, default="data/eval/results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Exporting Results - Model: {args.model}")
    print(f"{'='*80}\n")
    
    # Load QRELS
    qrels_items = load_qrels_jsonl(args.qrels)
    print(f"✓ Loaded {len(qrels_items)} queries\n")
    
    # Run evaluation
    print("⏳ Running evaluation...")
    evaluator = Evaluator(model_name=args.model, top_k_for_ranking=50)
    all_results, summary = evaluator.evaluate_queries(qrels_items)
    print("✓ Evaluation complete!\n")
    
    # Print summary
    print("Summary Metrics:")
    for metric, value in summary.items():
        print(f"  {metric}: {value:.4f}")
    
    # Export to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"{args.output}/{args.model}_results_{timestamp}.csv"
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Query', 'Rank', 'Title', 'URL', 'Language', 'Score', 'Confidence'])
        
        # Data
        for qrel, result in zip(qrels_items, all_results):
            query = qrel['query']
            for rank, doc in enumerate(result.ranked_documents[:10], 1):
                writer.writerow([
                    query,
                    rank,
                    doc.title,
                    doc.url,
                    doc.language,
                    f"{doc.raw_score:.4f}",
                    f"{doc.matching_confidence:.4f}"
                ])
    
    print(f"\n✓ Results exported to: {csv_file}")
    
    # Export summary to separate CSV
    summary_file = f"{args.output}/{args.model}_summary_{timestamp}.csv"
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for metric, value in summary.items():
            writer.writerow([metric, f"{value:.4f}"])
    
    print(f"✓ Summary exported to: {summary_file}")
    
    # Export to JSON
    json_file = f"{args.output}/{args.model}_results_{timestamp}.json"
    json_data = {
        'model': args.model,
        'timestamp': timestamp,
        'summary': summary,
        'queries': []
    }
    
    for qrel, result in zip(qrels_items, all_results):
        query_data = {
            'query': qrel['query'],
            'precision@10': result.precision_at_10,
            'recall@50': result.recall_at_50,
            'ndcg@10': result.ndcg_at_10,
            'mrr': result.mrr,
            'results': [
                {
                    'rank': rank,
                    'title': doc.title,
                    'url': doc.url,
                    'language': doc.language,
                    'score': doc.raw_score,
                    'confidence': doc.matching_confidence
                }
                for rank, doc in enumerate(result.ranked_documents[:10], 1)
            ]
        }
        json_data['queries'].append(query_data)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON exported to: {json_file}")
    print(f"\n{'='*80}")
    print("✓ Export Complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()