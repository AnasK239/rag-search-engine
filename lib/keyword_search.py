import string , math

from lib.inverted_index import BM25_B, BM25_K1, InvertedIndex

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

def bm25_idf_command(term: str) -> float:
    term = tokenize_term(term)
    return index.get_bm25_idf(term)

def bm25_tf_command(doc_id:int , term:str, k1:float = BM25_K1 , b:float = BM25_B) -> float:
    term = tokenize_term(term)
    return index.get_bm25_tf(doc_id , term , k1 , b)

def bm25_search_command(query:str , limit:int = DEFAULT_SEARCH_LIMIT) -> list:
    results = index.bm25_search(query , limit)
    ## results = [ (doc_id , score) , ... ]
    readable_results = []
    for item in results:
        res = []
        res.append(index.docmap[item[0]])
        res.append(item[1])
        readable_results.append(res)
    return readable_results
    