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
            
            # load_in_4bit=True,
            llm_int8_threshold=6.0,
            load_in_8bit=True,
            # llm_int8_enable_fp32_cpu_offload = True,
            #load_in_4bit_fp32_cpu_offload=True,
            
            #bnb_4bit_compute_dtype=torch.float16,
            #bnb_4bit_quant_type="nf4",
            #bnb_4bit_use_double_quant=True,
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


    def num_tokens_from_string(self, string: str) -> int:
        encoding = self.encoding
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def generate_prompt(self, user_prompt, context="", conversation_history=""):
        system_prompt = (
            "You are an AI assistant, designed to provide helpful responses in a straightforward "
            "and sincere manner. You only answer in english, the user does not know any other language."
            "Your primary goal is to assist users by answering their questions "
            "directly and succinctly. Focus solely on providing clear and concise answers to the user's "
            "queries. Your responses should directly address the user's request, providing accurate, "
            "concise, and useful information as efficiently as possible. Answer strictly to the query"
            "after <|im_start|>user. Everything before that just use as context. Do not force context"
            "into query response. Integrate context to the answer only if the query demands it.\n\n"
        )
        # f"context\n"
        # f"{context}"
        prompt = \
        f"<s>{conversation_history}" 
        f"[INST] {user_prompt} [/INST]"

        return prompt

    def ask_llm(self, user_prompt, context="",conversation_history=""):
        runtime_flag = "cuda:0"  
        
        full_prompt = self.generate_prompt(user_prompt, context, conversation_history)
        
        assistant_inst = "<assistant_response>"
        # print(f"context:{context}")
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(runtime_flag)
        outputs = self.model.generate(**inputs, max_new_tokens=500)[0]
        full_response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_start_idx = full_response.find(assistant_inst) + len(assistant_inst)
        return full_response[response_start_idx:].strip()


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


class ChatMemory:
    def __init__(self):
        self.history = []

        self.history = []

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
lm = LanguageModel(model_id=model_id)
chat_memory = ChatMemory()
