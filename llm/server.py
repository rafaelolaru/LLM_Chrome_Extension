from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import *
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def requires_context (query):
    context_keywords = ['summarize', 'explain the term', 'based on the text']
    return any(keyword in query.lower() for keyword in context_keywords)



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/manipulate")
async def manipulate_text(data: dict):
    print(data)
    query = data.get("chat", "")
    context = data.get("content","")
    #result = run_my_rag(query,context)
    if requires_context(query):
        print("The question requires context!!!")
        result = run_my_rag(query,context)
    else:
        result = ask_llm(query)
    # result =  ask_llm(query)
    print(result)
    return {"result": result}