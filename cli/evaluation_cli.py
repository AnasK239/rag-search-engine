import argparse , json , os , sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.search_utils import  load_movies
from lib.hybrid_search import HybridSearch
from llm import cross_encoder_rerank_results

GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT , "data" , "golden_dataset.json")

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    movies = load_movies()
    with open(GOLDEN_DATASET_PATH , "r") as f:
        golden_dataset = json.load(f)

    tests = golden_dataset['test_cases']
    hybrid_search = HybridSearch(movies)

    metrics = []
    for test in tests:
        results = hybrid_search.rrf_search(query= test['query'] , k=60 , limit=limit*35)
        total = len(results)

        relevant = 0
        retrieved = []
        for res in results:
            retrieved.append(res['title'])
            if res['title'] in test['relevant_docs']:
                relevant += 1

        percision = relevant / float(total)
        recall = relevant / float(len(test['relevant_docs']))
        f1 = 2 * (percision * recall) / (percision + recall)
        
        metrics.append({
            "query" : test['query'],
            "percision" : percision,
            "recall" : recall,
            "f1" : f1,
            "retrieved_docs" : retrieved,
            "relevant_docs" : test['relevant_docs']
        })

    for document in metrics:
        print_metric(document , limit)

def print_metric(doc: dict, limit: int):
    print(f"""- Query: {doc['query']}
   - Precision@{limit}: {doc['percision']:.4f}
   - Recall@{limit}: {doc['recall']:.4f}
   - F1 Score: {doc['f1']:.4f}
   - Retrieved: {", ".join(doc['retrieved_docs'])}
   - Relevant: {", ".join(doc['relevant_docs'])}""")



if __name__ == "__main__":
    main()