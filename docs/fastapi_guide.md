# FastAPI — Complete Guide

## What is FastAPI?

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python, based on standard Python type hints. Key features include:

- **Fast**: Very high performance, on par with NodeJS and Go.
- **Fast to code**: Increase development speed by 200–300%.
- **Fewer bugs**: Automatic data validation via Pydantic.
- **Intuitive**: Great editor support with auto-completion.
- **Easy**: Designed to be easy to use and learn.
- **Short**: Minimises code duplication.
- **Robust**: Get production-ready code with automatic interactive documentation.
- **Standards-based**: Based on OpenAPI and JSON Schema.

---

## Installation

```bash
pip install fastapi uvicorn[standard]
```

---

## First Steps

The simplest FastAPI application:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run the server:

```bash
uvicorn main:app --reload
```

- `main` — the module name (file `main.py`)
- `app` — the FastAPI object
- `--reload` — restart server on code changes (development only)

---

## Path Parameters

You can declare path parameters with the same syntax used by Python format strings:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

FastAPI automatically:
- Validates that `item_id` is an integer.
- Returns a 422 Unprocessable Entity error if validation fails.
- Shows the correct type in the OpenAPI docs.

### Path parameter with type

```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    return {"model_name": model_name, "message": "Have some residuals"}
```

---

## Query Parameters

Parameters not part of the path are automatically treated as query parameters:

```python
fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}]

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

URL: `/items/?skip=0&limit=10`

### Optional query parameters

```python
from typing import Optional

@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}
```

---

## Request Body

Declare request bodies using Pydantic models:

```python
from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

---

## Dependencies

FastAPI has a powerful Dependency Injection system:

```python
from fastapi import Depends

async def common_parameters(q: Optional[str] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

### Class-based dependencies

```python
class CommonQueryParams:
    def __init__(self, q: Optional[str] = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    return response
```

---

## Middleware

Add middleware to execute code before/after every request:

```python
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### CORS Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## HTTP Exceptions

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: str, items: dict):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}
```

---

## Background Tasks

Run code after returning a response:

```python
from fastapi import BackgroundTasks

def write_notification(email: str, message: str = ""):
    with open("log.txt", mode="w") as f:
        content = f"notification for {email}: {message}"
        f.write(content)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

---

## Response Models

Control the response shape with `response_model`:

```python
class UserIn(BaseModel):
    username: str
    password: str
    email: str

class UserOut(BaseModel):
    username: str
    email: str

@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn):
    return user  # password is automatically excluded
```

---

## File Uploads

```python
from fastapi import File, UploadFile

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}
```

---

## Lifespan Events

Run startup and shutdown code:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load ML model, connect to DB, etc.
    print("Starting up")
    yield
    # Shutdown: close connections
    print("Shutting down")

app = FastAPI(lifespan=lifespan)
```

---

## Status Codes

```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(name: str):
    return {"name": name}
```

---

## Security

### OAuth2 with Password (and Bearer)

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    return {"token": token}
```
