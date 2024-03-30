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
import promptlayer
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

os.environ["GOOGLE_CSE_ID"] = "006924ad473b24a37"
os.environ["GOOGLE_API_KEY"] = "AIzaSyA7CYjzkYWmocIiJdhKCl1hZtUunMcoyRc"
os.environ["TAVILY_API_KEY"] = "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"

class LanguageModel:
    def __init__(self, model_id):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                          quantization_config=self.quantization_config, 
                                                          device_map={"": 0},
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
            top_k=6,
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
            "and sincere manner. Your primary goal is to assist users by answering their questions "
            "directly and succinctly. Focus solely on providing clear and concise answers to the user's "
            "queries. Your responses should directly address the user's request, providing accurate, "
            "concise, and useful information as efficiently as possible. Answer strictly to the query"
            "after <|im_start|>user. Everything before that just use as context. Do not force context"
            "into query response. Integrate context to the answer only if the query demands it.\n\n"
        )
        assistant_inst = "<assistant_response>"
        prompt = f"{system_prompt}{context}{conversation_history}<|im_start|>user\n{user_prompt.strip()}\n{assistant_inst}\n"

        return prompt

    def ask_llm(self, user_prompt, context="",conversation_history=""):
        runtime_flag = "cuda:0"  
        if self.should_use_context(user_prompt):
            full_prompt = self.generate_prompt(user_prompt, context, conversation_history)
        else:
            full_prompt = self.generate_prompt(user_prompt, "", conversation_history)
        assistant_inst = "<assistant_response>"
        # print(f"context:{context}")
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(runtime_flag)
        outputs = self.model.generate(**inputs, max_new_tokens=500)[0]
        full_response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_start_idx = full_response.find(assistant_inst) + len(assistant_inst)
        return full_response[response_start_idx:].strip()

    def should_use_context(self, user_prompt:str) -> bool:
        prompt = (
        f"Consider the user query '{user_prompt}'. Analyze whether the task described by the user "
            "is likely to require detailed external information or context for an accurate and complete response. "
            "Tasks that typically require external context include summarizing specific articles, "
            "writing content based on detailed prompts, or providing insights based on specific data. "
            "Does the query describe such a task? Please answer 'yes' or 'no'."
        )
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        outputs = self.model.generate(**inputs, max_new_tokens=500)[0]
        full_response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        print(f"Query: '{user_prompt}' | Context Needed: {'yes' in full_response.lower()}")  # Debugging aid
        
        return "yes" in full_response.lower()

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
    
    def run_my_refined_rag(self, query, context="", conversation_history=""):
        # TODO: implement a sliding window mechanism
        if not context:
            return self.ask_llm(user_prompt=query,
                                context=context, 
                                conversation_history=conversation_history)
        question_prompt_template = """
                  Please provide a summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """

        context = [Document(page_content=context)] # metadata={"source": "local"}
        # print(context)
        # Splitting context into manageable pieces
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=784, chunk_overlap=50)
        # all_splits = text_splitter.split_documents(documents)
        documents = text_splitter.split_documents(context)

        question_prompt = PromptTemplate(
            template=question_prompt_template, input_variables=["text"]
        )

        refine_prompt_template = """
                    Write a concise summary of the following text delimited by triple backquotes.
                    Return your response in bullet points which covers the key points of the text.
                    ```{text}```
                    BULLET POINT SUMMARY:
                    """

        refine_prompt = PromptTemplate(
            template=refine_prompt_template, input_variables=["text"]
        )
        refine_chain = load_summarize_chain(
            self.llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
        )

        refine_outputs = refine_chain({"input_documents": documents}, return_only_outputs=True)
        print(json.dumps(refine_outputs, indent=1))
        return refine_outputs["output_text"]

    def run_my_new_rag(self, query, article_context="", conversation_history=""):
        # documents = [Document(page_content=context)] # metadata={"source": "local"}
        Settings.embed_model = 'local'
        
        text_list = [article_context]
        documents = [llama_Document(text=t) for t in text_list]
        print(f"context: {article_context}")
        print(f"documents: {documents}")

        pg_nodes = Settings.node_parser.get_nodes_from_documents(documents=documents)
        pg_storage_context = StorageContext.from_defaults()
        pg_storage_context.docstore.add_documents(pg_nodes)
    
        

        pg_summary_index = SummaryIndex.from_documents(documents)
        pg_vector_index = VectorStoreIndex.from_documents(documents)

        
        summary_query_engine = pg_summary_index.as_query_engine(response_mode = 'tree_summarize')
        vector_query_engine = pg_vector_index.as_query_engine(response_mode = 'refine')
        summary_tool = QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name="Context_Summary",
                    description="Summarizes the context"
                    )
                )
        vector_tool = QueryEngineTool(
                query_engine=vector_query_engine,
                metadata = ToolMetadata(
                    name="Context_QA",
                    description="Retrieves answers for questions from your context"
                    )
                )
        
        wikipedia.metadata = ToolMetadata(
                    name="Wikipedia",
                    description="Searches on Wikipedia for exact information"
                    )
        search = GoogleSearchAPIWrapper()

        
        def top5_results(query):
            return search.results(query, 5)
        TavilyAnswer_tool = TavilyAnswer(max_results=1, func= top5_results)
        TavilyAnswer_tool.metadata = ToolMetadata(
                    name="Intermediate Answer",
                    description="Search Tavily for recent internet results."
                    )
                                # metadata=ToolMetadata(
                                #     name="Intermediate Answer",
                                #     description="Search Tavily for recent results."
                                #     ),
                                # description="Search Tavily for recent results.",
                                # func=TavilyAnswer.run
    
        pg_tools = [TavilyAnswer_tool, summary_tool, vector_tool, wikipedia]
        for tool in pg_tools:
            print(tool.metadata.name)
        
        agent = ReActAgent.from_tools(
            tools= pg_tools,
            verbose=True,
            agent=AgentType.OPENAI_MULTI_FUNCTIONS,
            context="""
            You are an  agent capable of using a variety of tools to answer a question. You must always get your facts
            right and have factual data to sustain your answer. Never answer from memory, if unsure, use Tavilyanswer_tool.
            If the answer is not directly in the context, search on the internet with TavilyAnswer_tool.

            Here are the tools:
            -summary_tool
            -vector_tool
            -wikipedia
            -TavilyAnswer_tool
            

            To use these tools you must always respond in JSON format containing `"tool_name"` and `"input"` key-value pairs! For example, ...

            ```json
            {
                "tool_name": "sql_get_similar_examples",
                "input": "How many machines are there?"
            }
            ```

            Use the following format:

            User: the input question you must answer 
            Thought: you should always think about what to do 
            Action: the action to take in the JSON format listed above, whereas "tool_name" should be one of [...] and "input" should be the input of the tool
            Observation: the result of the action 
            ... (this Thought/Action/Observation can repeat N times) 
            Thought: I now know the final answer 
            Final Answer: the final answer to the original input question by using the Final Answer tool in JSON format   

            """
            f"You also get an article that you can use to get answers from, if you deem necessary: {article_context}"
        )
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
        response = agent.query(query)
        print (response)
        return str(response)

class ChatMemory:
    def __init__(self):
        self.history = []

    def add_conversation(self, user_prompt, model_response):
        if user_prompt.strip() and model_response.strip():
            self.history.append((user_prompt.strip(), model_response.strip()))

    def get_history(self):
        conversation_history = ""
        for user_prompt, model_response in self.history[-10:]:  # Only consider the last 10 conversations
            # Format each conversation pair as: "User: {user_prompt} Assistant: {model_response}"
            conversation_history += f"User: {user_prompt} \nAssistant: {model_response}\n"
        return conversation_history

    def clear_history(self):
        self.history = []

model_id = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser"
lm = LanguageModel(model_id=model_id)
chat_memory = ChatMemory()
