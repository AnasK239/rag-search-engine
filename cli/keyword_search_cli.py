import argparse , sys , os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.inverted_index import BM25_B, BM25_K1
from lib.keyword_search import bm25_idf_command, bm25_search_command, bm25_tf_command, tfidf_command, build_command, idf_command, search_command , STOPWORDS, tf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    index_parser = subparsers.add_parser("build", help="Build movies index")

    tf_parser = subparsers.add_parser("tf" , help="Find term frequency")
    tf_parser.add_argument("doc_id", type = int , help = "Document id")
    tf_parser.add_argument("term", type=str, help="Requested Term")

    idf_parser = subparsers.add_parser("idf" , help = "Find inverse doc frequency")
    idf_parser.add_argument("term" , type = str , help ="Search term")

    tf_idf_parser = subparsers.add_parser("tfidf" , help = "Find TF-IDF Score for a given term")
    tf_idf_parser.add_argument("doc_id" , type = int , help ="Document id")
    tf_idf_parser.add_argument("term" , type = str , help ="Search term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")


    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default = BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default = BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Limit of search results")

    args = parser.parse_args()


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_results = search_command(args.query)

            for i , res in enumerate(search_results , 1):
                print(f"{i}. {res['title']}")
            return 

        case "build":
            index = build_command()
            #docs = index.get_documents("merida")
            #print(f"First document for token 'merida' = {docs[0]}")
            return

        case "idf":
            idf = idf_command(args.term)

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            
        case "tf":
            res = tf_command(args.doc_id , args.term)
            print(res)

        case "tfidf":
            tf_idf = tfidf_command(args.doc_id , args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id , args.term , args.k1 , args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            results_list = bm25_search_command(args.query , args.limit)
            ## results is in tuple format [(DOCUMENT , SCORE) , (DOCUMENT , SCORE)]

            for i , item in enumerate(results_list , 1):
                print(f"{i}. ({item[0]['id']}) {item[0]['title']} - Score: {item[1]:.2f}")
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()