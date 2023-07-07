import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from PyPDF2 import PdfReader
import tiktoken
import openai
import math
from chromadb.utils import embedding_functions

GPT_MODEL = "gpt-3.5-turbo-0613"
EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_base = "https://dev.chatwithoracle.com/api/v1"

"""General utils"""
class Utils :
    def __init__(self) :
        pass
    
    @staticmethod
    def download_pdf_to_path(url, path) :
        with open(path, "wb") as f: 
            response = requests.get(url)
            f.write(response.content)
        return path

    @staticmethod
    def read_pdf(path, filter_ref = True) :
        """Takes a filepath to a PDF and returns a string of the PDF's contents"""
        # creating a pdf reader object
        reader = PdfReader(path)
        pdf_text = ""
        page_number = 0
        for page in reader.pages:
            page_number += 1
            page_text = page.extract_text()
            pdf_text += page_text + f"\nPage Number: {page_number}"
        return pdf_text
    
    @staticmethod
    def create_sentence_chunks(text, n, tokenizer) :
        """Returns successive n-sized chunks from provided text."""
        tokens = tokenizer.encode(text)
        i = 0
        while i < len(tokens):
            # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
            j = min(i + int(1.5 * n), len(tokens))
            while j > i + int(0.5 * n):
                # Decode the tokens and check for full stop or newline
                chunk = tokenizer.decode(tokens[i:j])
                if chunk.endswith(".") or chunk.endswith("\n") or chunk.endswith("?") or chunk.endswith("!"):
                    break
                j -= 1
            # If no end of sentence found, use n tokens as the chunk size
            if j == i + int(0.5 * n):
                j = min(i + n, len(tokens))
            yield tokens[i:j]
            i = j
    
    @staticmethod
    def chunk_sentence(text, max_tokens_each_trunk = 1500, max_chunks = None) :
        # Initialise tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")
        # Chunk up the document into 1500 token chunks
        chunks = Utils.create_sentence_chunks(text, max_tokens_each_trunk, tokenizer)
        text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
        
        if max_chunks is None or len(text_chunks) <= max_chunks :
            return text_chunks
        
        # merge chunks if max_chunks not None
        merged_chunks = []
        n = math.ceil(len(text_chunks)/float(max_chunks))

        merged_chunks = ["".join(text_chunks[i:i+n if i+n < len(text_chunks) else len(text_chunks)]) for i in range(0, len(text_chunks), n)]
        return merged_chunks

    @staticmethod
    def build_paper_head(paper_head, prefix = None) :
        # f"Title: {strings[0]['title']}\nAuthor: {strings[0]['authors']}\nPublished: {strings[0]['published']}\n" 
        head = ''
        if prefix is not None: 
            head = f"#{prefix}. "
        
        head += f"**[{paper_head['title']}]({paper_head['article_url']})**\n- **Author(s):**"

        authors = paper_head['authors']
        head += authors
        
        if 'summary' in paper_head :
            head += f"\n- **Summary**: {paper_head['summary']}"
        head += f"\n- **Publish Date**: {paper_head['published']}"
        head += f"\n- **URL**: {paper_head['article_url']}"
        return head
    
"""utils for openai API"""
class OpenAIUtils :
    def __init__(self) :
        pass
    
    @staticmethod
    def extract_chunk(content, template_prompt):
        """This function applies a prompt to some input content. In this case it returns a summarize chunk of text"""
        prompt = template_prompt + content
        response = openai.ChatCompletion.create(
            model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return response["choices"][0]["message"]["content"]

    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def embedding_request(text):
        response = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
        return response
    
    @staticmethod
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
            print(f"response from {base}/v1/chat/completions: '{response.json()}' for request '{messages}'")
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e
        
class EmbeddingService : 
    def __init__(self, embedding_vendor = 'OpenAI') :
        self.embedding_vendor = embedding_vendor

    def embedding_texts(self, texts : list) :
        # TODO: concurrent implementation
        if self.embedding_vendor == 'OpenAI' :
            embedding_func = embedding_functions.OpenAIEmbeddingFunction()
        elif self.embedding_vendor == 'Default' :
            embedding_func = embedding_functions.DefaultEmbeddingFunction()
        
        return embedding_func(texts)
