import argparse , os , sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.hybrid_search import HybridSearch, minmax_normalize , DEFAULT_LIMIT ,DEFAULT_WEIGHTED_ALPHA , DEFAULT_K
from lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize" , help="Normalize a list of scores")
    normalize_parser.add_argument("scores" , type = float ,nargs="*")

    weighted_search_parser = subparsers.add_parser("weighted-search" , help="Hybrid weighted search by keyword score , semantic score with tunable alpha")
    weighted_search_parser.add_argument("query" , type = str , help = "search query")
    weighted_search_parser.add_argument("--alpha" , type= float , default = DEFAULT_WEIGHTED_ALPHA, help="How much keyword vs semantic scoring matters")
    weighted_search_parser.add_argument("--limit" , type = int , default = DEFAULT_LIMIT)

    rrf_search_parser = subparsers.add_parser("rrf-search" , help="Use reciprocal rank fusion for searching and ranking")
    rrf_search_parser.add_argument("query" , type=str)
    rrf_search_parser.add_argument("-k" , type=int , default= DEFAULT_K,help="RRF K Parameter (ranking control)")
    rrf_search_parser.add_argument("--limit" , type=int , default= DEFAULT_LIMIT)
    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = minmax_normalize(args.scores)
            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.weighted_search(args.query , args.alpha , args.limit)

            for i , res in enumerate(results,1):
                print(f"""{i}. {res['title']}\n 
                Hybrid Score:{res['hybrid_score']}\n
                BM25:{res['bm25_score']}, Semantic:{res['semantic_score']}\n
                {res['description']}\n""")

        case "rrf-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.rrf_search(args.query , args.k ,args.limit)

            for i , res in enumerate(results,1):
                print(f"""{i}. {res['title']}\n 
                RRF Score:{res['rrf_score']}\n
                BM25 Rank:{res['bm25_rank']}, Semantic Rank:{res['semantic_rank']}\n
                {res['description']}\n""")
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()