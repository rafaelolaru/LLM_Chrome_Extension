from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import run_my_rag
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/manipulate")
async def manipulate_text(data: dict):
    print(data)
    query = data.get("chat", "")
    context = data.get("content","")
    result = run_my_rag(query,context)
    return {"result": result}