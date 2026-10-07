import json
import os , string


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
DEFAULT_SEARCH_LIMIT = 5

from typing import Any , TypedDict

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

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH , "r") as f:
        data = f.read()

    data = preprocess_text(data)
    
    return data.splitlines()

STOPWORDS = load_stopwords()

