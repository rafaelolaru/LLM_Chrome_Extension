import torch
from transformers import pipeline, AutoModelForCausalLM, \
    AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain, StuffDocumentsChain, ReduceDocumentsChain, MapReduceDocumentsChain

# from langchain_community.retrievers import RetrievalQ
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
# from langchain.schema.document import Document
from typing import Optional
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core import Document as llama_Document
from llama_index.core import StorageContext
from llama_index.core import Settings
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import (wikipedia, wikidata)
# from llama_index import ServiceContext
# import promptlayer
import tempfile
import shutil
import os
from langchain.chains.summarize import load_summarize_chain
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
from langchain_core.prompts.prompt import PromptTemplate
import urllib.request
import tiktoken
import pandas as pd
import json
from pathlib import Path as p
import os
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
class LanguageModel:
    def __init__(self, model_id):
        self.quantization_config = BitsAndBytesConfig(
            # llm_int8_threshold=6.0,
            # load_in_8bit=True,
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       use_fast=False,
                                                       )
        

        # self.device_map = infer_auto_device_map(self.model, no_split_module_classes=["OPTDecoderLayer"], dtype="float16")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                          quantization_config=self.quantization_config,
                                                          device_map={"":0},
                                                          trust_remote_code=True,
                                                          )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            # max_length=1024,
            max_new_tokens=512,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        Settings.llm = self.llm
        self.device="cuda"

    
    def _add_context_if_needed(self, user_prompt, context):
        # Define keywords or conditions to determine if context should be used
        keywords = ["article", "page", "information about", "details on", "what does it say about"]
        if any(keyword in user_prompt.lower() for keyword in keywords) and context:
            return f"Considering the following context: {context}, {user_prompt}"
        return user_prompt
    
    def _extract_relevant_response(self, full_response):
        # Find the last occurrence of the [/INST] marker and return the text after it
        inst_marker = "[/INST]"
        inst_index = full_response.rfind(inst_marker)
        if inst_index != -1:
            return full_response[inst_index + len(inst_marker):]
        return full_response

    def ask_llm(self, user_prompt, context="", conversation_history=None):
        if conversation_history is None:
            conversation_history = []

        # Modify the user prompt to include context if needed
        modified_user_prompt = self._add_context_if_needed(user_prompt, context)

        # Add the new user message to the conversation history
        conversation_history.append({"role": "user", "content": modified_user_prompt})

        # Print the conversation history for debugging
        print(f"Conversation history: {conversation_history}")

        # Use apply_chat_template to prepare the inputs for the model
        encodeds = self.tokenizer.apply_chat_template(conversation_history, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        
        # Generate the response
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract the relevant part of the response
        response = self._extract_relevant_response(decoded[0])
        
        return response.strip()


    def run_my_rag(self, query, article_context="", conversation_history=""):
        # TODO: implement a sliding window mechanism
        if not article_context:
            return self.ask_llm(user_prompt=query,
                                context=article_context, 
                                conversation_history=conversation_history)
        article_context = [Document(page_content=article_context)] # metadata={"source": "local"}
        # print(context)
        # Splitting context into manageable pieces
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        print(text_splitter)
        # all_splits = text_splitter.split_documents(documents)
        documents = text_splitter.split_documents(article_context)
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        vectordb = FAISS.from_documents(documents, embeddings)
        retriever = vectordb.as_retriever()

        
        # Setting up and running the RetrievalQA process
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="map_reduce", 
            retriever=retriever,
            verbose=True
        )
        
        response = qa.invoke(query)
        print(response)
        return response['result']
    
    def run_my_reduced_rag(self, query, context="", conversation_history=""):
        # TODO: implement a sliding window mechanism
        if not context:
            return self.ask_llm(user_prompt=query,
                                context=context, 
                                conversation_history=conversation_history)
        
        context = [Document(page_content=context)] # metadata={"source": "local"}
        # print(context)
        # Splitting context into manageable pieces
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=784, chunk_overlap=50)
        # all_splits = text_splitter.split_documents(documents)
        documents = text_splitter.split_documents(context)

        refine_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            return_intermediate_steps=False,
        )

        refine_outputs = refine_chain({"input_documents": documents}, return_only_outputs=True)
        print(json.dumps(refine_outputs, indent=1))
        return refine_outputs["output_text"]


# class ChatMemory:
#     def __init__(self):
#         self.history = []

#     def add_message(self, role, content):
#         """Add a new message to the history."""
#         self.history.append({"role": role, "content": content})

#     def get_conversation(self):
#         """Return the entire conversation as a single string."""
#         return "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in self.history])

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
lm = LanguageModel(model_id=model_id)
# chat_memory = ChatMemory()
