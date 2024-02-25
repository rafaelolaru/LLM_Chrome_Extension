from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import lm, chat_memory
from pydantic import BaseModel
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    context: str = ""


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/manipulate")
async def manipulate_text(data: ChatRequest):
    context = data.context if data.context else ""
    conversation_history=chat_memory.get_history()
    if "deep context" in data.query:
        print("run_my_rag")
        result = lm.run_my_rag(query=data.query, 
                               context=data.context, 
                               conversation_history=conversation_history)
    else:
        print("ask_llm")
        result = lm.ask_llm(user_prompt=data.query, 
                            context=context, 
                            conversation_history=conversation_history)
    chat_memory.add_conversation(data.query, result)
    return {"result": result}

@app.delete("/clear_memory")
def clear_memory():
    chat_memory.clear_history()
    return {"status": "Chat memory cleared"}
