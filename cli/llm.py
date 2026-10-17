import os , json
from dotenv import load_dotenv
from openai import OpenAI
from textwrap import dedent
from time import sleep
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

def query_model(prompt:str) -> str:
    messages = [{"role" : "user","content" : prompt}]
    response = client.chat.completions.create(model="openrouter/free" , messages=messages)
    return response.choices[0].message.content

def enhance_query(query:str , mode:str) -> str:
    if mode is None or mode == "":
        return query

    prompt = get_mode_prompt(query , mode)

    enhanced_query = query_model(prompt)
    print(f"Enhanced query ({mode}): '{query}' -> '{enhanced_query}'\n")
    
    return enhanced_query

def rerank_results(query:str , results:list[dict] , mode:str , limit:int) -> list:
    """Point-wise LLM Reranking"""
    reranked_results = []

    for res in results:
        prompt = get_rerank_prompt(query , res , mode)
        response = query_model(prompt)
        try:
            score = float(response.strip())
        except ValueError:
            score = 0.0
        res['rerank_score'] = round(score , 3)
        reranked_results.append(res)
        sleep(3)


    sorted_results = sorted(reranked_results , key = lambda item:item['rerank_score'], reverse = True)

    return sorted_results[:limit]

    
def batch_rerank_results(query:str , results:list[dict] , mode:str , limit:int) -> list:
    """Batch LLM Reranking"""
    lines = []
    mapping = {}
    for res in results:
        mapping[res['id']] = res
        lines.append(f"ID:{res['id']} | Title:{res['title']} | Description:{res['description']}")
    
    doc_list_str = "\n".join(lines)
    
    prompt = get_rerank_prompt(query , doc_list_str , mode)
    response = query_model(prompt) 

    start = response.find("[")
    end = response.rfind("]")
    candidate = response[start:end + 1]
    
    try:
        ranked_ids = json.loads(candidate)
    except json.JSONDecodeError:
        return results[:limit]

    reranked_results = []
    for i , id in enumerate(ranked_ids,1):
        if id not in mapping:
            continue
        doc = mapping[id]
        doc['rerank_rank'] = i
        reranked_results.append(doc)
    
    print(len(reranked_results))

    return reranked_results[:limit]

def cross_encoder_rerank_results(query:str , results:list[dict] , limit:int) -> list:
    """Cross Encoder reranking , No LLM"""
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    pairs = []
    for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])

    scores = cross_encoder.predict(pairs)
    for i,doc in enumerate(results,0):
        doc_score = scores[i]
        doc['cross_encoder_score'] = round(float(doc_score),3)
    
    sorted_results = sorted(results , key=lambda item: item['cross_encoder_score'] , reverse=True)
    return sorted_results[:limit]

def get_mode_prompt(query:str , mode:str) -> str:
    prompt = ""
    match mode:
        case "spell":
            prompt = dedent(f"""\
                Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}" """).strip()

        case "expand":
            prompt = dedent(f"""\
                Expand the user-provided movie search query below with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User query: "{query}" """).strip()
            
        case "rewrite":
            prompt = dedent(f"""\
                Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}" """).strip()
    
    return prompt

def get_rerank_prompt(query:str , doc  , mode:str) -> str:
    prompt = ""
    match mode:
        case "individual":
            prompt = dedent(f"""\
                Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("description", "")}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation or puncutuation.

                Score:""").strip()

        case "batch":
            prompt = dedent(f"""\
                Rank the movies listed below by relevance to the following search query.

                Query: "{query}"

                Movies:
                {doc}

                Return the movie IDs in order of relevance, best match first.

                Your response must be a raw JSON array of integers.
                Do not wrap the JSON in Markdown. Do not use a ```json code block.
                Do not include any explanatory text.

                For example:
                [75, 12, 34, 2, 1]

                Ranking:""").strip()
            
    return prompt

def llm_judge(results:list , query:str) -> list:
    formatted = []
    for i, item in enumerate(results,1):
        formatted.append(f"{i}. {item['title']}: {item['description'][:150]}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    result = query_model(prompt)
    try:
        result = json.loads(result)
    except:
        print(f"Incorrect Json parsed")
        return []

    return result
    
# messages = [
#     {
#         "role": "user",
#         "content": "Hey is this working ?",
#     }
# ]

# response = client.chat.completions.create(model="openrouter/free" , messages=messages)

# print(f"{response.choices[0].message.content}")
# print(f"""Prompt tokens: {response.usage.prompt_tokens}, 
#       Response tokens: {response.usage.completion_tokens}
#       """)