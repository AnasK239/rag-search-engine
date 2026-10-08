import string , math

from lib.inverted_index import InvertedIndex

from .search_utils import tokenize_term, tokenize_text , load_movies, preprocess_text ,  STOPWORDS , DEFAULT_SEARCH_LIMIT

index = InvertedIndex()

try:
    index.load()
except FileNotFoundError:
    index.build()
    index.save()

def search_command(query: str , limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movie_ids = set()

    query_tokens = tokenize_text(query)

    for tok in query_tokens:
        movie_ids.update(index.get_documents(tok))

    res = []
    
    for id in sorted(movie_ids)[:limit]:
        res.append(index.docmap[id])
            
    return res

def build_command() -> InvertedIndex:
    index.build()
    index.save()
    return index

def tf_command(doc_id , term) -> int:
    term = tokenize_term(term)

    return index.get_tf(doc_id , term)

def idf_command(term: str) -> float:
    term = tokenize_term(term)

    total_doc_count = len(index.docmap)
    term_match_doc_count = len(index.get_documents(term))
    
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    return idf

def tfidf_command(doc_id:int , term:str) -> float:
    return tf_command(doc_id , term) * idf_command(term)