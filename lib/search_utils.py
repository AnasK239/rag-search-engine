import json
import os , string


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
DEFAULT_SEARCH_LIMIT = 5

from typing import Any , TypedDict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

class Movie (TypedDict):
    id: int
    title: str
    description: str
    
def load_movies() -> list[Movie]:
    with open(DATA_PATH , "r") as f:
        data = json.load(f)

    return data["movies"]

## Normalize text and remove punctuation 
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("" , "" , string.punctuation))
    return text

def stem_and_remove_stopwords(tokens: list[str]) -> list:
    res = []
    for item in tokens:
        if item not in STOPWORDS:
            item = stemmer.stem(item)
            res.append(item)
        
    return res 

def tokenize_text(text: str) -> list[str]:
    tokens = preprocess_text(text).split()
    return stem_and_remove_stopwords(tokens)

def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)

    if len(tokens) != 1:
        raise ValueError("Expected exactly one token")

    return tokens[0]
    
def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH , "r") as f:
        data = f.read()

    data = preprocess_text(data)
    
    return data.splitlines()

STOPWORDS = load_stopwords()


