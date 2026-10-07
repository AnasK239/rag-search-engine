import string

from .search_utils import load_movies, preprocess_text ,  STOPWORDS , DEFAULT_SEARCH_LIMIT
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def search_command(query: str , limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:

    movies = load_movies()
    results = []
    preprocessed_query = preprocess_text(query)
    query_tokens = tokenize(preprocessed_query)
    query_tokens = stem_and_remove_stopwords(query_tokens)
    
    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        title_tokens = tokenize(preprocessed_title)
        title_tokens = stem_and_remove_stopwords(title_tokens)

        found = False
        for token in query_tokens:
            for title_token in title_tokens:

                if token in title_token:
                    results.append(movie)
                    found = True
                    break

            if found:
                break

        if len(results) >= limit:
            break
            
    return results


def stem_and_remove_stopwords(tokens: list[str]) -> list:
    res = []
    for item in tokens:
        if item not in STOPWORDS:
            item = stemmer.stem(item)
            res.append(item)
        
    return res 


def tokenize(text: str) -> list:
    res = text.split()
    return res
