import base64
import json
import os
import time
import requests
import yaml

import arxiv
import ast
import concurrent
from csv import writer
from IPython.display import display, Markdown, Latex
import openai
import pandas as pd
from PyPDF2 import PdfReader
import requests
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from tqdm import tqdm
from termcolor import colored


GPT_MODEL = "gpt-3.5-turbo-0613"
EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_base = "https://dev.chatwithoracle.com/api/v1"

# Set a directory to store downloaded papers
doc_root = 'data'

data_dir = os.path.join(doc_root, 'papers')
paper_dir_filepath = f"{doc_root}/arxiv_library.csv"
conversation_dir_filepath = f"{doc_root}/conversation_history.json"

# Generate a blank dataframe where we can store downloaded files
df = pd.DataFrame(list())
if os.path.exists(paper_dir_filepath):
    df = pd.read_csv(paper_dir_filepath)
else :
    df.to_csv(paper_dir_filepath)

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def embedding_request(text):
    response = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
    return response

def build_paper_head(paper_head, prefix = None) :
    # f"Title: {strings[0]['title']}\nAuthor: {strings[0]['authors']}\nPublished: {strings[0]['published']}\n" 
    head = ''
    if prefix is not None: 
        head = f"#{prefix}. "
    
    head += f"**[{paper_head['title']}]({paper_head['article_url']})**\n- **Author(s):**"

    authors = paper_head['authors']
    if isinstance(authors, str) :
        authors = eval(authors)

    for idx, author in enumerate(authors) :
        head += f"{author.name}" if idx == 0 else f", {author.name}"
    
    if 'summary' in paper_head :
        head += f"\n- **Summary**: {paper_head['summary']}"
    head += f"\n- **Publish Date**: {paper_head['published']}"
    return head

def search_article_list(query, library=paper_dir_filepath, top_k=5, recent_days=None):
    """
    This function gets the top_k articles based on a user's query, sorted by relevance and recency. 
    Return the title, summary, authors and published date for each article
    """
    search = arxiv.Search(
        query=query, max_results=top_k, sort_by=arxiv.SortCriterion.Relevance
    )

    result_list = []
    for result in search.results():
        result_dict = {}
        result_dict.update({"title": result.title})
        result_dict.update({"summary": result.summary})
        result_dict.update({"authors": result.authors})
        result_dict.update({"published": result.published})

        # Taking the first url provided
        result_dict.update({"article_url": [x.href for x in result.links][0]})
        result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_list.append(result_dict)
    
        # Store references in library file
        response = embedding_request(text=result.title)
        print(f"download PDF to {data_dir}")
        file_reference = [
            result.title,
            result.authors,
            result.published,
            result_dict['article_url'],
            result.download_pdf(data_dir),
            response["data"][0]["embedding"],
        ]

        # Write to file
        with open(library, "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(file_reference)
            f_object.close()
            
    # compose result manually
    if len(result_list) == 0:
        return "No results found"
    result_buffer = f"Search results for '**{query}**': \n\n"
    for idx, result in enumerate(result_list):
        result_buffer += build_paper_head(result, idx) + "\n\n"

    return result_buffer

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
    rel_th : float = 0.95) -> list[str]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = embedding_request(query)
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        ({"filepath" : row["filepath"], "title": row["title"], 
          "authors" : row["authors"], "published" : row["published"], 
          "article_url" : row["article_url"]}, 
         relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    if relatednesses[top_n-1] > rel_th :
        print(f"Find paper with relatedness {relatednesses[top_n-1]} > {rel_th}!\n")
        return strings[:top_n]
    return None

def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    """Returns successive n-sized chunks from provided text."""
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def extract_chunk(content, template_prompt):
    """This function applies a prompt to some input content. In this case it returns a summarize chunk of text"""
    prompt = template_prompt + content
    response = openai.ChatCompletion.create(
        model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response["choices"][0]["message"]["content"]

def summarize_text(query):
    """This function does the following:
    - Reads in the arxiv_library.csv file in including the embeddings
    - Finds the closest file to the user's query
    - Scrapes the text out of the file and chunks it
    - Summarizes each chunk in parallel
    - Does one final summary and returns this to the user"""

    # A prompt to dictate how the recursive summarizations should approach the input paper
    summary_prompt = """Summarize this text from an academic paper. Extract any key points with reasoning.\n\nContent:"""

    # If the library is empty (no searches have been performed yet), we perform one and download the results
    library_df = pd.read_csv(paper_dir_filepath).reset_index()
    if len(library_df) == 0:
        print("No papers searched yet, downloading first.")
        result_list = get_articles(query)
        print("Papers downloaded, continuing")
        library_df = pd.read_csv(paper_dir_filepath).reset_index()

    # rank by relatedness
    library_df.columns = ["title", "authors", "published", "article_url", "filepath", "embedding"]
    library_df["embedding"] = library_df["embedding"].apply(ast.literal_eval)
    strings = strings_ranked_by_relatedness(query, library_df, top_n=1, rel_th=0.95)

    if strings is None : # no-related papers in local repository, search arXiv again
        result_list = get_articles(query)
        library_df = pd.read_csv(paper_dir_filepath).reset_index()
        library_df.columns = ["title", "authors", "published", "article_url", "filepath", "embedding"]
        library_df["embedding"] = library_df["embedding"].apply(ast.literal_eval)
        strings = strings_ranked_by_relatedness(query, library_df, top_n=1)
    
    assert strings is not None, "No papers found"
    
    print("Chunking text from paper")
    pdf_text = read_pdf(strings[0]['filepath'])

    # Initialise tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results = ""

    # Chunk up the document into 1500 token chunks
    chunks = create_chunks(pdf_text, 1500, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    print("Summarizing each chunk of text")

    # Parallel process the summaries
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(text_chunks)
    ) as executor:
        futures = [
            executor.submit(extract_chunk, chunk, summary_prompt)
            for chunk in text_chunks
        ]
        with tqdm(total=len(text_chunks)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            data = future.result()
            if len(results) < 3500 : # limit by 4097 context-size
                results += data
            else :
                break

    # Final summary
    print("Summarizing into overall summary")
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Write a summary collated from this collection of key points extracted from an academic paper.
                        The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
                        User query: {query}
                        The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                        Key points:\n{results}\nSummary:\n""",
            }
        ],
        temperature=0,
    )
    
    # attach title, author & published
    paper_head = build_paper_head(strings[0])
    response["choices"][0]["message"]["content"] = paper_head + "\n\n" + response["choices"][0]["message"]["content"]
    return response

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        base = openai.api_base or "https://api.openai.com"
        base = base.replace("/v1", "")
        response = requests.post(
            f"{base}/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        print(f"response from {base}/v1/chat/completions: '{response}' for request '{messages}'")
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

class Conversation:
    def __init__(self, conversation_dir_filepath):
        self.conversation_history = []
        self.conversation_dir_filepath = conversation_dir_filepath

        if os.path.exists(self.conversation_dir_filepath):
            self.load_history()
            self.display_conversation()
        self.system_setted = False

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        if role == "system" :
            self.system_setted = True
        
        self.conversation_history.append(message)
        with open(self.conversation_dir_filepath, "w") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False)

    def load_history(self) :
        with open(self.conversation_dir_filepath, "r") as f:
            self.conversation_history = json.load(f)
            for message in self.conversation_history :
                if message["role"] == "system" :
                    self.system_setted = True
                    break
        pass
    
    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )

# Initiate our get_articles and read_article_and_summarize functions
arxiv_functions = [
    {
        "name": "get_articles",
        "description": """Use this function to download an academic paper from arXiv to answer user questions.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            User query in JSON. Responses should be summarized and should include the article URL reference
                            """,
                }
            },
            "required": ["query"],
        }
    },

    {
        "name": "search_and_return_articles", 
        "description": """Use this function to search arXiv database for a list of articles related to user query and return, 
        argument 'top_k' should be an integer for the number of articles to return,
        argument 'recency_days' should be an integer for the number of days since the article was published""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""Subject of the article in plain text"""
                },
                "top_k" : {
                    "type": "integer",
                    "description": f"""Number of articles to return"""
                }, 
                "recent_days" : {
                    "type": "integer",
                    "description": f"""Number of days since the article was published"""
                }
            }, 
            "required": ["query"],
        },
    },
    
    {
        "name": "read_article_and_summarize",
        "description": """Use this function to read whole papers and provide a summary for users.
        You should NEVER call this function before search_and_return_articles has been called in the conversation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            Description of the article in plain text based on the user's query
                            """,
                }
            },
            "required": ["query"],
        },
    }
]

def chat_completion_with_function_execution(messages, functions=[None]):
    """This function makes a ChatCompletion API call with the option of adding functions"""
    response = chat_completion_request(messages, functions)
    full_message = response.json()["choices"][0]
    if full_message["finish_reason"] == "function_call":
        print(f"Function generation requested, calling function {full_message}")
        return call_arxiv_function(messages, full_message)
    else:
        print(f"Function not required, responding to user")
        return response.json()

def get_articles(query):
    """Function to download an academic paper from arXiv to answer user questions."""
    print(f"Query: {query}")
    articles = arxiv.query(query=query, max_results=10)
    print(f"Number of articles found: {len(articles)}")
    return articles

def call_arxiv_function(messages, full_message):
    """Function calling function which executes function calls when the model believes it is necessary.
    Currently extended by adding clauses to this if statement."""
    func_name = full_message["message"]["function_call"]["name"]
    print(f'Get function name: {func_name}')
    if func_name == "get_articles":
        try:
            parsed_output = json.loads(
                full_message["message"]["function_call"]["arguments"]
            )
            print("Getting search results")
            results = get_articles(parsed_output["query"])
        except Exception as e:
            print(parsed_output)
            print(f"Function execution failed")
            print(f"Error message: {e}")
        messages.append(
            {
                "role": "function",
                "name": full_message["message"]["function_call"]["name"],
                "content": str(results),
            }
        )
        try:
            print("Got search results, summarizing content")
            response = chat_completion_request(messages)
            return response.json()
        except Exception as e:
            print(type(e))
            raise Exception("Function chat request failed")

    elif func_name == "read_article_and_summarize":
        parsed_output = json.loads(
            full_message["message"]["function_call"]["arguments"]
        )
        print("Finding and reading paper")
        summary = summarize_text(parsed_output["query"])
        return summary
    elif func_name == "search_and_return_articles" :
        arguments = eval(full_message["message"]["function_call"]["arguments"])
        article_lst = search_article_list(arguments['query'], top_k=arguments['top_k'])
        return article_lst
    else:
        raise Exception("Function does not exist and cannot be called")

#################################################
# Define a conversation for the paper assistant #
#################################################

paper_system_message = """You are arXivGPT, a helpful assistant pulls academic papers to answer user questions. 
You search for a list of papers to answer user questions. You read and summarize the paper clearly according to user provided topics.
Please summarize in clear and concise format. Begin!"""

paper_conversation = Conversation(conversation_dir_filepath)
if not paper_conversation.system_setted :
    paper_conversation.add_message("system", paper_system_message)

def custom_generate_reply(question, original_question, seed, state, eos_token, stopping_strings, is_chat = True):

    # Add a user message
    print(f"custom_generate(question): {question}\n")
    print(f"custom_generate(original_question): {original_question}\n")
    print(f"custom_generate(seed): {seed}\n")
    print(f"custom_generate(state): {state}\n")
    print(f"custom_generate(eos_token): {eos_token}\n")
    print(f"custom_generate(stopping_strings): {stopping_strings}\n")

    # extract the last question
    current_question = ":".join(question.split("\n")[-2].split(":")[1:]).lstrip()
    paper_conversation.add_message("user", f"{current_question}")
    chat_response = chat_completion_with_function_execution(
        paper_conversation.conversation_history, functions=arxiv_functions
    )

    if isinstance(chat_response, str):
        assistant_message = chat_response
    else :
        assistant_message = chat_response["choices"][0]["message"]["content"]
    
    paper_conversation.add_message("assistant", assistant_message)
    paper_conversation.display_conversation()

    yield assistant_message