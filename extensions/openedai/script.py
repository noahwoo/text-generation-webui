import json
import os
import concurrent
from csv import writer
from IPython.display import display, Markdown, Latex
import openai
import pandas as pd
from tqdm import tqdm
from termcolor import colored
from extensions.openedai.search import ScholarSearch, PaperRepository, VectorStore
from extensions.openedai.utils import Utils, OpenAIUtils, GPT_MODEL
from extensions.openedai.conversation import Conversation

# Set a directory to store downloaded papers
doc_root = 'data'

paper_dir = os.path.join(doc_root, 'papers')
paper_dir_filepath = f"{doc_root}/arxiv_library.csv"
conversation_dir_filepath = f"{doc_root}/conversation_history.json"

# wrap df with PaperRepository
paper_repo = PaperRepository()
if os.path.exists(paper_dir_filepath):
    paper_repo.load(paper_dir_filepath)
else :
    paper_repo.create(paper_dir_filepath)

# vector store for paper trunks
paper_trunks = VectorStore()

"""Function to download an academic paper from arXiv to answer user questions."""
def get_articles(query, library=paper_dir_filepath):
    
    print(f"Query: {query}")
    search_results = ScholarSearch().search(query, site='arxiv.org', num = 1)
    
    assert len(search_results) > 0, f"No papers found for query {query}"
    result = search_results[0]
    # Store references in library file
    response = OpenAIUtils.embedding_request(text=result.title)
    print(f"download [{result.title}] to {paper_dir}")
    file_reference = [
        result.title,
        result.authors,
        result.published,
        result.article_url,
        result.pdf_url,
        result.download_pdf(paper_dir),
        response["data"][0]["embedding"],
    ]

    # Write to file
    with open(library, "a") as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(file_reference)
        f_object.close()

    return result

"""functions: support the paper search request from users"""
def search_article_list(query, library=paper_dir_filepath, top_k=5, recent_days=None, year_from=None):
    """
    This function gets the top_k articles based on a user's query, sorted by relevance and recency. 
    Return the title, summary, authors and published date for each article
    """
    # search = arxiv.Search(
    #     query=query, max_results=top_k, sort_by=arxiv.SortCriterion.Relevance
    # )

    search_results = ScholarSearch().search(query, site='arxiv.org', num = top_k, year_from = year_from)

    result_list = []
    # for result in search.results():
    for result in search_results:
        result_dict = {}
        result_dict.update({"title": result.title})
        result_dict.update({"summary": result.summary})
        result_dict.update({"authors": result.authors if result.authors is not None else 'NA'})
        result_dict.update({"published": result.published})

        # Taking the first url provided
        # result_dict.update({"article_url": [x.href for x in result.links][0]})
        # result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_dict.update({"article_url": result.article_url})
        result_list.append(result_dict)
    
        # Store references in library file
        response = OpenAIUtils.embedding_request(text=result.title)
        file_reference = [
            result.title,
            result.authors,
            result.published,
            result.article_url,
            result.pdf_url,
            result.local_pdf_path(paper_dir),
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
        result_buffer += Utils.build_paper_head(result, idx) + "\n\n"

    return result_buffer

"""functions: support the paper content query from user"""
def chat_with_paper(query, pdf_url) :
    print(f"chat_with_paper: query:{query}")
    print(f"chat_with_paper: pdf_url:{pdf_url}")

    pdf_file_path = paper_repo.get_filepath(pdf_url)
    if not os.path.exists(pdf_file_path):
        print(f"Download paper from [{pdf_url}] to [{pdf_file_path}]")
        Utils.download_pdf_to_path(pdf_url, pdf_file_path)
    
    if not paper_trunks.indexed(pdf_url) :
        paper_trunks.add_paper(pdf_url, pdf_file_path, paper_repo.get_meta(pdf_url))

    trunks = paper_trunks.query(query, n_result=10)
    background = ';'.join(trunks)

    # prompt and summarize 
    print("Summarizing into overall summary")
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Given the Background Content, summarize an answer for the user query.\n\n
                    Background Content:{background}\nUser Query:{query}\nSummary:""",
            }
        ],
        temperature=0,
    )
    
    return response

"""functions: support whole paper summarization request from user"""
def summarize_text(query):
    """This function does the following:
    - Reads in the arxiv_library.csv file in including the embeddings
    - Finds the closest file to the user's query
    - Scrapes the text out of the file and chunks it
    - Summarizes each chunk in parallel
    - Does one final summary and returns this to the user"""

    # A prompt to dictate how the recursive summarizations should approach the input paper
    summary_prompt = """Summarize this text from an academic paper. Extract any key points with reasoning.\n\nContent:"""
    # load paper repository
    paper_repo.load(paper_dir_filepath)
    # rank by relatedness
    strings = paper_repo.strings_ranked_by_relatedness(query, top_n=1, rel_th=0.9)

    if strings is None : # no-related papers in local repository, search arXiv again
        result = get_articles(query)
        paper_repo.load(paper_dir_filepath)
        strings = paper_repo.strings_ranked_by_relatedness(query, top_n=1)
    
    assert strings is not None and len(strings) > 0, f"No papers found for query [{query}]"

    print("Chunking text from paper")
    if not os.path.exists(strings[0]['filepath']) :
        print(f"Download paper from [{strings[0]['pdf_url']}] to [{strings[0]['filepath']}]")
        Utils.download_pdf_to_path(strings[0]['pdf_url'], strings[0]['filepath'])
    
    assert os.path.exists(strings[0]['filepath']), f"File not found at {strings[0]['filepath']}"

    pdf_text = Utils.read_pdf(strings[0]['filepath'])

    # Chunk up the document into 1500 token chunks
    text_chunks = Utils.chrunk_sentence(pdf_text, max_tokens=1500)
    print("Summarizing each chunk of text")

    results = ""
    # Parallel process the summaries
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(text_chunks)
    ) as executor:
        futures = [
            executor.submit(Utils.extract_chunk, chunk, summary_prompt)
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
    paper_head = Utils.build_paper_head(strings[0])
    response["choices"][0]["message"]["content"] = paper_head + "\n\n" + response["choices"][0]["message"]["content"]
    return response

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
                    "description": f"""User query is the subject of the article in plain text. 
                            Responses should be summarized and should include the article URL reference""",
                }
            },
            "required": ["query"],
        }
    },

    {
        "name": "search_and_return_articles", 
        "description": """Use this function to search arXiv database for a list of articles related to user query and return, 
        argument 'top_k' should be an integer for the number of articles to return,
        argument 'recency_days' should be an integer for the number of days since the article was published
        argument 'year_from' should be an integer for the year from which you want the results to be included""",
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
                },
                "year_from" : {
                    "type": "integer",
                    "description": f"""The year from which you want the results to be included"""
                },
            }, 
            "required": ["query"],
        },
    },
    
    {
        "name": "read_article_and_summarize",
        "description": """Use this function to read whole paper and provide a summary for users.
        You should NEVER call this function before search_and_return_articles has been called in the conversation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""Description of the article in plain text based on the user's query""",
                }
            },
            "required": ["query"],
        },
    },

    {
        "name" : "paper_content_explorer",
        "description" : """This function is used to explore a part of the paper content, such as abstract, introduction,
                        method & algorithm, experimental results, conclusion or limitations etc., to answer users questions. 
                        It is called when user asks questions about the paper content.""",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "query" : {
                    "type" : "string",
                    "description" : """The question from user to query the content of paper in plain text"""
                }, 
                "pdf_url" : {
                    "type" : "string",
                    "description" : """URL of the pdf file of the paper, value of this argument should be extracted 
                        from the response of search_and_return_articles function"""
                }
            }, 
             "required": ["query", "pdf_url"],
        }
    }
]
    
def call_arxiv_function(messages, full_message):
    """Function calling function which executes function calls when the model believes it is necessary.
    Currently extended by adding clauses to this if statement."""
    func_name = full_message["message"]["function_call"]["name"]
    print(f'####Get function name: {func_name}')
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
            response = OpenAIUtils.chat_completion_request(messages)
            return response.json()
        except Exception as e:
            print(type(e))
            raise Exception("Function chat request failed")

    elif func_name == "read_article_and_summarize":
        parsed_output = json.loads(
            full_message["message"]["function_call"]["arguments"]
        )
        print("Finding and reading paper")
        query = parsed_output["query"]
        summary = summarize_text(query)
        # comment out function call record, cause leads to error for following function generation
        # paper_conversation.add_message('function', f'{func_name}("{query}")')
        return summary
    elif func_name == "search_and_return_articles" :
        arguments = eval(full_message["message"]["function_call"]["arguments"])
        article_lst = search_article_list(arguments['query'], 
                                          top_k=arguments['top_k'] if 'top_k' in arguments else None,
                                          year_from=arguments['year_from'] if 'year_from' in arguments else None,
                                          recent_days=arguments['recent_days'] if 'recent_days' in arguments else None)
        # comment out function call record, cause leads to error for following function generation
        # paper_conversation.add_message('function', f'{func_name}({arguments})')
        return article_lst
    elif func_name == "paper_content_explorer" :
        arguments = eval(full_message["message"]["function_call"]["arguments"])
        print("Finding and reading paper")
        query   = arguments["query"]
        pdf_url = arguments["pdf_url"]
        result  = chat_with_paper(query, pdf_url)
        # comment out function call record, cause leads to error for following function generation
        # paper_conversation.add_message('function', f'{func_name}({arguments})')
        return result
    else:
        raise Exception("Function does not exist and cannot be called")


def chat_completion_with_function_execution(messages, functions=[None]):
    """This function makes a ChatCompletion API call with the option of adding functions"""
    response = OpenAIUtils.chat_completion_request(messages, functions)
    full_message = response.json()["choices"][0]
    if full_message["finish_reason"] == "function_call":
        print(f"Function generation requested, calling function {full_message}")
        return call_arxiv_function(messages, full_message)
    else:
        print(f"Function not required, responding to user")
        return response.json()


"""
# Define a conversation for the paper assistant, reply to user input from WebUI
"""

paper_system_message = """You are arXivGPT, a helpful assistant pulls academic papers to answer user questions. 
You search for a list of papers to answer user questions. You read and summarize the paper clearly according to user provided topics.
Please summarize in clear and concise format. Begin!"""

paper_conversation = Conversation(conversation_dir_filepath)
if not paper_conversation.system_setted :
    paper_conversation.add_message("system", paper_system_message)

def custom_generate_reply(question, original_question, seed, state, eos_token, stopping_strings, is_chat = True):
    # extract the last question
    print(f"custom_generate_reply called with question: [{question}]")
    paper_conversation.check_and_reset(question)
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