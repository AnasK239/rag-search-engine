import argparse , os , sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.semantic_search import verify_model , embed_text , verify_embeddings , embed_query_text , SemanticSearch
from lib.search_utils import DEFAULT_SEARCH_LIMIT , load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    search_parser = subparsers.add_parser("verify", help="")

    
    embedding_parser = subparsers.add_parser("embed_text", help="Embed text")
    embedding_parser.add_argument("text", type=str, help="Text to be embedded")  

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings" , help="Load or create embeddings for documents")

    query_parser = subparsers.add_parser("embed_query" , help="Create embeddings for search query")
    query_parser.add_argument("query" , type=str , help="query")

    semantic_search_parser = subparsers.add_parser("search" , help = "Preform semantic search using query")
    semantic_search_parser.add_argument("query" , type=str , help="Search query")
    semantic_search_parser.add_argument("--limit" , type=int , default= DEFAULT_SEARCH_LIMIT, help="Search query")


    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            sc = SemanticSearch()
            movies = load_movies()

            sc.load_or_create_embeddings(movies)
            results = sc.search(args.query , args.limit)

            for i , result in enumerate(results , 1):
                print(f"{i}. {result['title']} ({result['score']})\n{result['description']}\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()