#from transformers import BigBirdModel, BigBirdTokenizer
import requests
#from sentence_transformers import SentenceTransformer
from os import environ as env
from dotenv import find_dotenv, load_dotenv
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine
import httpx
from langchain_community.embeddings import OllamaEmbeddings
import ollama 
from sentence_transformers import SentenceTransformer
from transformers import BigBirdModel, BigBirdTokenizer
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import AutoTokenizer
from transformers import AutoConfig
import asyncio
import time
from requests.exceptions import RequestException
from typing import Tuple, List, Optional
from services.logging_config import get_logger
import os
import certifi
from transformers import AutoTokenizer, AutoModel

# Create a logger instance for this module
logger = get_logger(__name__)  # Will use 'models.embedding_model' as the logger name


import  torch
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Retrieve environment variables
hugging_face_api_url_sentence_bert = env.get("HUGGING_FACE_API_URL_SENTENCE_BERT")
hugging_face_api_url_big_bird = env.get("HUGGING_FACE_API_URL_BIG_BIRD")
hugging_face_access_token = env.get("HUGGING_FACE_ACCESS_TOKEN")

class EmbeddingModel:
    # Set certificate paths at class level before any HuggingFace operations
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi.where()
    logger.info(f"Set certificate paths to: {certifi.where()}")

    _sentence_model_instance = None
    _bigbird_model_instance = None
    _bigbird_tokenizer_instance = None
    _sentence_bert_tokenizer = None
    _sentence_bert_config = None
    _bigbird_tokenizer = None
    _bigbird_config = None

    def __init__(self):
        if env.get("LOCALPROCESSING"):
            logger.info("Applying local processing  ")  
            if  EmbeddingModel._bigbird_model_instance is None:     
                EmbeddingModel._bigbird_model_instance = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
                self.bigbird_model_instance =  EmbeddingModel._bigbird_model_instance
                EmbeddingModel._bigbird_tokenizer_instance = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
                self.bigbird_tokenizer_instance = EmbeddingModel._bigbird_tokenizer_instance
            if  EmbeddingModel._sentence_model_instance is None:
                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                EmbeddingModel._sentence_model_instance = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device = "cpu")
                self.sentence_model_instance = EmbeddingModel._sentence_model_instance

            #self.client = Ollama(model=model)
            #self.embedding_model =env.get("EMBEDDING_MODEL_LOCAL") if env.get("EMBEDDING_MODEL_LOCAL") else "text-embedding-3-small"
           

        #if EmbeddingModel._bigbird_model_instance is None:
        #    EmbeddingModel._bigbird_model_instance = BigBirdModel.from_pretrained(
        #        bigbird_model_name
        #    )
        #    EmbeddingModel._bigbird_tokenizer_instance = BigBirdTokenizer.from_pretrained(bigbird_model_name)
        #self.bigbird_model = EmbeddingModel._bigbird_model_instance
        #self.bigbird_tokenizer = EmbeddingModel._bigbird_tokenizer_instance
        #pass

        # Initialize tokenizers if not already initialized
        if EmbeddingModel._sentence_bert_tokenizer is None:
            logger.info("Initializing BERT tokenizer...")
            # Set certificate path for HuggingFace
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            os.environ['SSL_CERT_FILE'] = certifi.where()
            
            hugging_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            EmbeddingModel._sentence_bert_tokenizer = AutoTokenizer.from_pretrained(hugging_model_name)
            EmbeddingModel._sentence_bert_config = AutoConfig.from_pretrained(hugging_model_name)
            logger.info("BERT tokenizer initialized")

            # Initialize text splitter with the tokenizer
            num_special_tokens = EmbeddingModel._sentence_bert_tokenizer.num_special_tokens_to_add()
            max_token_limit = EmbeddingModel._sentence_bert_config.max_position_embeddings - num_special_tokens - 30
            EmbeddingModel._sentence_bert_splitter = TokenTextSplitter(
                chunk_size=max_token_limit, 
                chunk_overlap=0
            )
            logger.info("BERT tokenizer and text splitter initialized")

        if EmbeddingModel._bigbird_tokenizer is None:
            logger.info("Initializing BigBird tokenizer...")
            # Set certificate path for HuggingFace
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            os.environ['SSL_CERT_FILE'] = certifi.where()
            
            model_bigbird = "google/bigbird-roberta-base"
            EmbeddingModel._bigbird_tokenizer = AutoTokenizer.from_pretrained(model_bigbird)
            EmbeddingModel._bigbird_config = AutoConfig.from_pretrained(model_bigbird)
            logger.info("BigBird tokenizer initialized")

            # Initialize BigBird text splitter
            num_special_tokens = EmbeddingModel._bigbird_tokenizer.num_special_tokens_to_add()
            max_token_limit = EmbeddingModel._bigbird_config.max_position_embeddings - num_special_tokens - 30
            EmbeddingModel._bigbird_splitter = TokenTextSplitter(
                chunk_size=max_token_limit, 
                chunk_overlap=0
            )
            logger.info("BigBird tokenizer and text splitter initialized")


    #def get_sentence_embedding(self, text):
    #    embedding = self.sentence_model.encode([text])
    #    return embedding[0]
    def call_huggingface_api(input_ids):
    # Replace 'your-api-url' and 'your-api-token' with your actual API URL and token
        api_url = hugging_face_api_url_sentence_bert
        headers = {"Authorization": f"Bearer {hugging_face_access_token}"}
        payload = {"inputs": input_ids}

        response = requests.post(api_url, headers=headers, json=payload)
        return response.json()
    # Replace the local model call with a Hugging Face API call


    async def get_sentence_embedding(self, text: str) -> Tuple[List[List[float]], List[str]]:
        """
        Get sentence embeddings for the given text asynchronously.
        
        Args:
            text (str): The text to embed
            
        Returns:
            Tuple[List[List[float]], List[str]]: A tuple containing:
                - List of embeddings where each embedding is a list of floats
                - List of text chunks
        """
        start_time = time.time()

        if env.get("LOCALPROCESSING"):
            logger.info("Local processing is enabled for sentence embedding")
            loop = asyncio.get_event_loop()
            try:
                embeddinglocal = await loop.run_in_executor(
                    None,
                    lambda: EmbeddingModel._sentence_model_instance.encode(text)
                )
                return embeddinglocal.tolist(), [text]
            except Exception as e:
                logger.error(f"Error in local sentence embedding: {str(e)}")
                raise

        # Initialize tokenizers if not already initialized
        if EmbeddingModel._sentence_bert_tokenizer is None:
            logger.info("Initializing BERT tokenizer...")
            # Set certificate path for HuggingFace
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            os.environ['SSL_CERT_FILE'] = certifi.where()
            
            hugging_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            EmbeddingModel._sentence_bert_tokenizer = AutoTokenizer.from_pretrained(hugging_model_name)
            EmbeddingModel._sentence_bert_config = AutoConfig.from_pretrained(hugging_model_name)
            
            # Initialize text splitter with the tokenizer
            num_special_tokens = EmbeddingModel._sentence_bert_tokenizer.num_special_tokens_to_add()
            max_token_limit = EmbeddingModel._sentence_bert_config.max_position_embeddings - num_special_tokens - 30
            EmbeddingModel._sentence_bert_splitter = TokenTextSplitter(
                chunk_size=max_token_limit, 
                chunk_overlap=0
            )
            logger.info("BERT tokenizer and text splitter initialized")

        # Use pre-initialized splitter
        chunks = EmbeddingModel._sentence_bert_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")

        async def process_chunk(chunk: str, i: int) -> Optional[List[float]]:
            # Use pre-initialized tokenizer and config
            tokenized_chunk = EmbeddingModel._sentence_bert_tokenizer(
                chunk,
                truncation=True,
                max_length=EmbeddingModel._sentence_bert_config.max_position_embeddings,
                return_tensors="pt"
            )

            if len(tokenized_chunk['input_ids'][0]) > EmbeddingModel._sentence_bert_config.max_position_embeddings:
                logger.warning(f"Chunk {i} exceeds max token size: {len(tokenized_chunk['input_ids'][0])} tokens")
                return None

            payload = {
                "inputs": chunk,
                "parameters": {"truncation": True}
            }
            headers = {
                "Authorization": f"Bearer {hugging_face_access_token}",
                "Content-Type": "application/json"
            }

            try:
                logger.info(f"Sending async request to Hugging Face API sentence_bert with payload size: {len(chunk)} characters")
                async with httpx.AsyncClient(verify=False) as client:
                    response = await client.post(
                        hugging_face_api_url_sentence_bert,
                        headers=headers,
                        json=payload,
                        timeout=30.0
                    )

                    if response.status_code == 400:
                        error_text = response.text
                        logger.error(f"Bad Request: {error_text}")
                        return None

                    response.raise_for_status()
                    response_json = response.json()

                    if isinstance(response_json, list) and isinstance(response_json[0], list):
                        embedding = response_json[0]
                        logger.info(f"Chunk {i} embedding dimensions: {len(embedding)}")
                        return embedding
                    else:
                        logger.error(f"Unexpected API response format: {response_json}")
                        return None

            except httpx.HTTPError as e:
                logger.error(f"HTTP error in async API request: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error in async API request: {str(e)}")
                return None

        # Process all chunks concurrently
        chunk_embeddings = await asyncio.gather(
            *[process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        )

        # Filter out None values
        embeddings = [emb for emb in chunk_embeddings if emb is not None]

        if not embeddings:
            logger.warning("No embeddings were generated. Returning empty list.")
            return [], chunks

        total_time = time.time() - start_time
        logger.info(f"Total async BERT embedding generation took: {total_time:.4f} seconds")

        return embeddings, chunks
    
    async def get_bigbird_embedding(self, text: str) -> Tuple[List[List[float]], List[str]]:
        """
        Get BigBird embeddings for the given text asynchronously.
        
        Args:
            text (str): The text to embed
            
        Returns:
            Tuple[List[List[float]], List[str]]: A tuple containing:
                - List of embeddings where each embedding is a list of floats
                - List of text chunks
        """
        if env.get("LOCALPROCESSING"):
            logger.info("Big bird Local processing is enabled")
            loop = asyncio.get_event_loop()
            
            def process_locally():
                inputs = EmbeddingModel._bigbird_tokenizer_instance(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = EmbeddingModel._bigbird_model_instance(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                return embeddings

            return await loop.run_in_executor(None, process_locally)

        # Use pre-initialized tokenizer and config
        if not EmbeddingModel._bigbird_tokenizer or not EmbeddingModel._bigbird_config:
            logger.error("BigBird tokenizer or config not initialized")
            raise ValueError("BigBird tokenizer or config not initialized")

        # Use the singleton BigBird text splitter
        chunks = EmbeddingModel._bigbird_splitter.split_text(text)
        embeddings: List[List[float]] = []

        async def process_chunk(chunk: str, i: int) -> Optional[List[float]]:
             # Use pre-initialized tokenizer
            tokenized_chunk = EmbeddingModel._bigbird_tokenizer(
                chunk, 
                truncation=True, 
                max_length=EmbeddingModel._bigbird_config.max_position_embeddings,
                return_tensors="pt"
            )
            
            if len(tokenized_chunk['input_ids'][0]) > EmbeddingModel._bigbird_config.max_position_embeddings:
                logger.warning(f"Chunk {i} exceeds max token size: {len(tokenized_chunk['input_ids'][0])} tokens")
                return None

            payload = {
                "inputs": chunk,
                "parameters": {"truncation": True}
            }
            headers = {
                "Authorization": f"Bearer {hugging_face_access_token}",
                "Content-Type": "application/json"
            }

            try:
                logger.info(f"Sending async request to Hugging Face API bigbird with payload size: {len(chunk)} characters")
                async with httpx.AsyncClient(verify=False) as client:
                    response = await client.post(
                        hugging_face_api_url_big_bird,
                        headers=headers,
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code == 400:
                        error_text = response.text
                        logger.error(f"Bad Request: {error_text}")
                        return None

                    response.raise_for_status()
                    response_json = response.json()

                    if isinstance(response_json, list) and all(isinstance(elem, list) for elem in response_json) and all(isinstance(sub_elem, list) for elem in response_json for sub_elem in elem):
                        flattened_embeddings = [item for sublist in response_json for item in sublist]
                        return np.mean(np.array(flattened_embeddings), axis=0).tolist()
                    else:
                        logger.error("Unexpected API response format.")
                        return None

            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred: {str(e)}")
                return None

        # Process all chunks concurrently
        chunk_embeddings = await asyncio.gather(
            *[process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        )
        
        # Filter out None values and empty lists
        embeddings = [emb for emb in chunk_embeddings if emb is not None]

        if not embeddings:
            logger.warning("No embeddings were generated. Returning empty list.")
            return [], chunks

        return embeddings, chunks

    def get_bigbird_embedding_hugging_face(self, text):
        
        if env.get("LOCALPROCESSING"):
            logger.info("Big bird Local processing is enabled")
            inputs = EmbeddingModel._bigbird_tokenizer_instance(text, return_tensors="pt", padding=True, truncation=True)
            # Get the embeddings from BigBird model
            with torch.no_grad():
                outputs = EmbeddingModel._bigbird_model_instance(**inputs)
                # Extract the last layer hidden states (embeddings)
                last_hidden_states = outputs.last_hidden_state.numpy()
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                flattened_embeddings = last_hidden_states.reshape(-1, last_hidden_states.shape[-2], last_hidden_states.shape[-1])
                #embeddingsflat = last_hidden_states.tolist()
                #flattened_embeddings = last_hidden_states.view(last_hidden_states.size(0), -1)
                #embeddings = np.mean(np.array(flattened_embeddings), axis=0)
         
            #return last_hidden_states[0]
            return embeddings
        
        headers = {
        "Authorization": f"Bearer {hugging_face_access_token}"
        }
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True},  # Ensure the request waits for the model if it's currently loading
            # Add any other parameters here, such as truncation or padding
        }
        
        response = requests.post(hugging_face_api_url_big_bird, headers=headers, json=payload)

        if response.status_code == 200:
            response_json = response.json()

            # Log the raw response for debugging
            #logger.info(f"BigBird hugging_face raw response: {response_json}")

            # Assuming the API returns an array of token embeddings, we average them
            # Verify this assumption based on the actual structure of response_json
            if isinstance(response_json, list) and all(isinstance(elem, list) for elem in response_json) and all(isinstance(sub_elem, list) for elem in response_json for sub_elem in elem):
                # Flatten the list of lists of lists to a list of lists
                flattened_embeddings = [item for sublist in response_json for item in sublist]
                # Calculate the mean across the first dimension (tokens) to get a single embedding vector
                embeddings = np.mean(np.array(flattened_embeddings), axis=0)
                logger.info(f"BigBird hugging_face processed embedding dimensions: {embeddings.shape}")
                return embeddings
            else:
                logger.error("Unexpected API response format.")
                raise ValueError("Unexpected API response format.")

            
        else:
            logger.error("Failed to get embeddings from Hugging Face API")
            response.raise_for_status()
    
    def get_embeddinglocal(self,text):
        response = ollama.embeddings(model=self.embedding_model, prompt=text)            
            #response = self.embeddingclient.embed_documents(text)
        return response['embedding']