from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from main import lm, chat_memory
from pydantic import BaseModel, EmailStr
import redis
import jwt
from jose import jwt,  JWTError
from datetime import datetime, timedelta
import os
from conversation_manager import ConversationHistoryManager

SECRET_KEY = f"{os.getenv('JWT_SECRET_KEY')}"
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserAuth(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    query: str
    context: str = ""
app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="authenticate")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

users_redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

conversation_redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

conversation_manager = ConversationHistoryManager(conversation_redis_client)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_username_from_token(token: str = Security(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")  # 'sub' is typically used to store the username
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_403_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependency to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=HTTPException, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = users_redis_client.get(username)
    if user is None:
        raise credentials_exception
    return {"username": username}


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

@app.post("/verify_token")
async def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Check if token is expired
        expire = payload.get("exp")
        if datetime.now().timestamp() > expire:
            return {"valid": False, "message": "Token is expired"}
        return {"valid": True, "message": "Token is valid"}
    except JWTError:
        return {"valid": False, "message": "Invalid token"}

@app.post("/register")
async def register(user: UserRegister):
    if users_redis_client.exists(user.username):
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = pwd_context.hash(user.password)
    users_redis_client.set(user.username, hashed_password)

    return {"message": "User registered successfully"}

@app.post("/authenticate")
async def authenticate(user: UserAuth):
    stored_password = users_redis_client.get(user.username)
    if not stored_password or not pwd_context.verify(user.password, stored_password):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Example of a protected route
@app.get("/users/me")
async def read_users_me(current_user: UserRegister = Depends(get_current_user)):
    return current_user

@app.post("/manipulate")
async def manipulate_text(data: ChatRequest, token: str = Depends(oauth2_scheme)):
    context = data.context if data.context else ""
    print(f'hello?')
    username = get_username_from_token(token=token)
    print(f'username: {username}')

    if "??" in data.query:
        print("run_my_rag")
        result = lm.run_my_rag(query=data.query, 
                               article_context=data.context, 
                               conversation_history=conversation_manager.get_conversation_history(username=username))
    elif "?" in data.query:
        print("ask_llm")
        result = lm.ask_llm(user_prompt=data.query, 
                            context=context, 
                            conversation_history=conversation_manager.get_conversation_history(username=username))
    else:
        print('run_my_refined_rag')
        result = lm.run_my_refined_rag(query=data.query,
                               context=data.context, 
                               conversation_history=conversation_manager.get_conversation_history(username=username))
    conversation_manager.add_message(username, data.query, result)
    print(conversation_manager.get_conversation_history(username=username))
    return {"result": result}

@app.delete("/clear_memory")
def clear_memory():
    chat_memory.clear_history()
    return {"status": "Chat memory cleared"}

if __name__ == "__main__":
    os.system("uvicorn server:app --port 5000 --reload")
