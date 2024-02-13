import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from huggingface_hub import InferenceClient


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,device_map={"": 0})
system_prompt = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
query_wrapper_prompt = "<|USER|>{query_str}<|ASSISTANT|>"
def ask_llm(user_prompt):
    runtimeFlag = "cuda:0"
    B_INST, E_INST = "### Instruction:\n", "### Response:\n"

    prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n\n{E_INST}"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Despite returning the usual output, the streamer will also print the generated text to stdout.
    outputs =  model.generate(**inputs, streamer=streamer, max_new_tokens=500)[0]
    full_response = tokenizer.decode(outputs, skip_special_tokens=True)

    # Extract the response part after "### Response:\n"
    response_start_idx = full_response.find(E_INST) + len(E_INST)
    actual_response = full_response[response_start_idx:].strip()

    print(f"response:\"{actual_response}\"")
    return actual_response
pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=500,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)
llm = HuggingFacePipeline(pipeline=pipeline)

def run_my_rag(query, context):
    print(f"Query: {query}\n")
    print(f"Context: {context}\n")
    if context:
        print(f"Context: {context}\n")
    else:
        context = "no context"
    documents = [Document(page_content=context, metadata={"source": "local"})]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    result = qa.run(query)
    return result
