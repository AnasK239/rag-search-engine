import argparse , os , sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.hybrid_search import HybridSearch, minmax_normalize , DEFAULT_LIMIT ,DEFAULT_WEIGHTED_ALPHA , DEFAULT_K
from lib.search_utils import load_movies
from cli.llm import client , batch_rerank_results , cross_encoder_rerank_results ,enhance_query , rerank_results
from textwrap import dedent

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
    rrf_search_parser.add_argument("--enhance" , type=str , choices=["spell","rewrite","expand"], help="Query enhancement using external LLM")
    rrf_search_parser.add_argument("--rerank-method" , type=str , choices=["individual" , "batch" , "cross_encoder"])
    
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

            original_limit = args.limit
            new_limit = original_limit
            rerank_method = args.rerank_method

            if rerank_method is not None and rerank_method.strip() != "":
                new_limit = original_limit*5

            ##LLM Query-enhancement
            query = enhance_query(args.query , args.enhance)
            results = hybrid_search.rrf_search(query , args.k , new_limit)


            if rerank_method is not None and rerank_method.strip() != "":
                print(f"""\
                Re-ranking top {original_limit} results using {rerank_method} method...
                Reciprocal Rank Fusion Results for '{query}' (k={args.k}):""")       

                match rerank_method:
                    case "individual":
                        results = rerank_results(query , results , rerank_method , original_limit)
                        print_rerank_results(results)
                        return

                    case "batch":
                        results = batch_rerank_results(query , results , rerank_method , original_limit)
                        print_batch_rerank_results(results)
                        return

                    case "cross_encoder":
                        results = cross_encoder_rerank_results(query , results , original_limit)
                        print_cross_rerank_results(results)
                        return
            
            print_rrf(results)
                
        case _:
            parser.print_help()

def print_rerank_results(results:list):
    for i , res in enumerate(results,1):
        print(f"""{i}. {res['title']}
   Re-rank Score: {res['rerank_score']}/10
   RRF Score: {res['rrf_score']}
   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}
   {res['description'][:100]}...\n""")

def print_batch_rerank_results(results:list):
    for i , res in enumerate(results,1):
        print(f"""{i}. {res['title']}
   Re-rank Rank: {res['rerank_rank']}
   RRF Score: {res['rrf_score']}
   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}
   {res['description'][:100]}...\n""")

def print_cross_rerank_results(results:list):
    for i , res in enumerate(results,1):
        print(f"""{i}. {res['title']}
   Cross Encoder Score: {res['cross_encoder_score']}
   RRF Score: {res['rrf_score']}
   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}
   {res['description'][:100]}...\n""")
              
def print_rrf(results:list):
    for i , res in enumerate(results,1):
        print(f"""{i}. {res['title']}
   RRF Score: {res['rrf_score']}
   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}
   {res['description'][:100]}...\n""")
        
if __name__ == "__main__":
    main()