import argparse , os , sys , string

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.semantic_search import ChunkedSemanticSearch, verify_model , embed_text , verify_embeddings , embed_query_text , SemanticSearch , fixed_chunk_text , semantic_chunk_text
from lib.search_utils import DEFAULT_SEARCH_LIMIT , load_movies

DEFAULT_CHUNK_SIZE = 200
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_OVERLAP = 0

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

    chunk_parser = subparsers.add_parser("chunk" , help = "Fixed size chunking")
    chunk_parser.add_argument("text" , type=str , help="Fixed size chunking text")
    chunk_parser.add_argument("--chunk-size" , type=int , default= DEFAULT_CHUNK_SIZE, help="Fixed size chunking")
    chunk_parser.add_argument("--overlap" , type=int , default= DEFAULT_OVERLAP)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk" , help="Semantic chunking of input text")
    semantic_chunk_parser.add_argument("text" , type = str , help="TEXT")
    semantic_chunk_parser.add_argument("--max-chunk-size" , type = int , default= DEFAULT_SEMANTIC_CHUNK_SIZE)
    semantic_chunk_parser.add_argument("--overlap" , type = int , default= DEFAULT_OVERLAP)

    semantic_chunk_embedder_parser = subparsers.add_parser("embed_chunks" , help="Created chunked semantic embeddings from the documents")

    semantic_search_chunked_parser = subparsers.add_parser("search_chunked" , help="Preform semantic search using chunked embeddings")
    semantic_search_chunked_parser.add_argument("text" , type = str)
    semantic_search_chunked_parser.add_argument("--limit" , type = int , default = 5)

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

        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            text = args.text
            chunk_size = args.chunk_size
            overlap = args.overlap

            chunks = fixed_chunk_text(text , chunk_size , overlap)

            for i , chunk in enumerate(chunks,1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            text = args.text
            chunk_size = args.max_chunk_size
            overlap = args.overlap

            chunks = semantic_chunk_text(text , chunk_size , overlap)
            for i , chunk in enumerate(chunks,1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            movies = load_movies()
            chunked_sc = ChunkedSemanticSearch("all-MiniLM-L6-v2")

            embeddings = chunked_sc.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            movies = load_movies()
            chunked_sc = ChunkedSemanticSearch("all-MiniLM-L6-v2")

            chunked_sc.load_or_create_chunk_embeddings(movies)
            
            results = chunked_sc.search_chunks(args.text , args.limit)
            for i , res in enumerate(results,1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()