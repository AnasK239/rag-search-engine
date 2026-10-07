import argparse , sys , os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.keyword_search import search_command , STOPWORDS

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_results = search_command(args.query)

            for i , res in enumerate(search_results , 1):
                print(f"{i}. {res['title']}")

            return 
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()