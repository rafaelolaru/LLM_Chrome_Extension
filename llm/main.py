# from llama_index.prompts import PromptTemplate
# from langchain_community.document_loaders import TextLoader

import torch
from transformers import pipeline, AutoModelForCausalLM, \
    AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from typing import Optional
import tempfile
import shutil
import os
import urllib.request

model_id = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser"

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
            max_length=1024,
            max_new_tokens=250,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

    def generate_prompt(self, user_prompt, context="", conversation_history=""):
        """Constructs the prompt for the model based on user input, context, and conversation history."""
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
        
        # Return True if "yes" is in the response, indicating context is needed
        return "yes" in full_response.lower()

    def run_my_rag(self, query, context="", conversation_history=""):
        # TODO: implement a sliding window mechanism
        if not context:
            return self.ask_llm(user_prompt=query,
                                context=context, 
                                conversation_history=conversation_history)
        documents = [Document(page_content=context, metadata={"source": "local"})]
        
        # Splitting context into manageable pieces
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        all_splits = text_splitter.split_documents(documents)
        
        # Embedding documents
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings)
        retriever = vectordb.as_retriever()
        
        # Setting up and running the RetrievalQA process
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="map_reduce",  # Adjust this as needed
            retriever=retriever,
            verbose=True
        )
        
        result = qa.run(query)
        return result

class ChatMemory:
    def __init__(self):
        self.history = []

    def add_conversation(self, user_prompt, model_response):
        if user_prompt.strip() and model_response.strip():
            self.history.append((user_prompt.strip(), model_response.strip()))

    # def get_history(self):
    #     conversation_history = ""
    #     for pair in self.history[-10:]:  # Only consider the last 10 conversations
    #         conversation_history += f"<s>[INST] {pair[0]} [/INST] {pair[1]}</s>"
    #     return conversation_history
            
    def get_history(self):
        conversation_history = ""
        for user_prompt, model_response in self.history[-10:]:  # Only consider the last 10 conversations
            # Format each conversation pair as: "User: {user_prompt} Assistant: {model_response}"
            conversation_history += f"User: {user_prompt} \nAssistant: {model_response}\n"
        return conversation_history

    def clear_history(self):
        self.history = []


def get_paper(url: str, main_file: Optional[str] = None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "_paper.tmp")
        if main_file is not None:
            # we are dealing with an archive
            file_path += ".tar"

        urllib.request.urlretrieve(url, file_path)

        if main_file is not None:
            # we are dealing with an archive
            shutil.unpack_archive(file_path, tmp_dir)
            main_file_path = os.path.join(tmp_dir, main_file)
        else:
            main_file_path = file_path

        with open(main_file_path, "r") as f:
            data = f.read()

    return data


@torch.no_grad()
def load_to_memory(model, tokenizer, text: str):
    tokenized_data = tokenizer(text, return_tensors="pt")
    input_ids = tokenized_data.input_ids
    input_ids = input_ids.to(model.device)
    torch.manual_seed(0)
    output = model(input_ids=input_ids)
    memory = output.past_key_values
    return memory


PROMPT_PREFIX = "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can.\n"
# Initialize the language model with your specific model ID
model_id = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser"
lm = LanguageModel(model_id=model_id)
fot_paper = get_paper(url="https://raw.githubusercontent.com/CStanKonrad/long_llama/main/assets/fot_paper.tar", main_file="fot_paper.tex")
fot_memory = load_to_memory(lm.model, lm.tokenizer, PROMPT_PREFIX + fot_paper)
chat_memory = ChatMemory()
