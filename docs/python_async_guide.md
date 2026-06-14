# Python Async/Await — Complete Guide

## What is Async/Await?

Python's `async`/`await` syntax (introduced in Python 3.5 via PEP 492) enables writing asynchronous, non-blocking code that looks like synchronous code. It is built on top of the `asyncio` event loop.

**Why use async?**
- I/O-bound tasks (network requests, file I/O, database queries) spend most of their time waiting.
- With async, the event loop can run other tasks while waiting instead of blocking the thread.
- For CPU-bound tasks, use multiprocessing instead.

---

## Basics

### Coroutines

An `async def` function returns a **coroutine object** when called. To execute it, you must `await` it (from another coroutine) or run it with `asyncio.run()`.

```python
import asyncio

async def say_hello():
    print("Hello")
    await asyncio.sleep(1)  # non-blocking sleep
    print("World")

# Run from synchronous code
asyncio.run(say_hello())
```

### Awaiting a coroutine

```python
async def fetch_data():
    await asyncio.sleep(0.5)
    return {"data": "result"}

async def main():
    result = await fetch_data()  # suspends here until fetch_data finishes
    print(result)

asyncio.run(main())
```

---

## Running Tasks Concurrently

### asyncio.gather — run multiple coroutines concurrently

```python
import asyncio

async def task(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name} done"

async def main():
    # Both tasks run concurrently — total time ~2s, not 3s
    results = await asyncio.gather(
        task("A", 1.0),
        task("B", 2.0),
    )
    print(results)  # ['A done', 'B done']

asyncio.run(main())
```

### asyncio.create_task — fire and forget

```python
async def main():
    task_a = asyncio.create_task(task("A", 1.0))
    task_b = asyncio.create_task(task("B", 2.0))
    # Both tasks start immediately; await them when needed
    result_a = await task_a
    result_b = await task_b
```

### asyncio.wait_for — with timeout

```python
async def main():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=5.0)
    except asyncio.TimeoutError:
        print("Operation timed out")
```

---

## Async Context Managers

```python
class AsyncDB:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

async def main():
    async with AsyncDB() as db:
        result = await db.query("SELECT ...")
```

---

## Async Iterators and Generators

```python
class AsyncCounter:
    def __init__(self, stop: int):
        self.stop = stop

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current >= self.stop:
            raise StopAsyncIteration
        self.current += 1
        await asyncio.sleep(0.1)
        return self.current

# Async generator (simpler syntax)
async def count_up(stop: int):
    for i in range(stop):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for number in count_up(5):
        print(number)
```

---

## Async HTTP Requests with httpx

```python
import httpx

async def fetch_url(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

async def fetch_multiple(urls: list[str]) -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]
```

---

## FastAPI and Async

FastAPI natively supports both sync and async route handlers:

```python
from fastapi import FastAPI

app = FastAPI()

# Async route — non-blocking I/O
@app.get("/async")
async def async_route():
    await asyncio.sleep(0.1)  # simulate DB query
    return {"message": "async response"}

# Sync route — FastAPI runs it in a thread pool automatically
@app.get("/sync")
def sync_route():
    import time
    time.sleep(0.1)
    return {"message": "sync response"}
```

**Best practice**: Use `async def` for routes that do I/O (database, HTTP calls). Use `def` for CPU-bound operations.

---

## asyncio.Queue — Producer/Consumer Pattern

```python
import asyncio

async def producer(queue: asyncio.Queue):
    for i in range(10):
        await queue.put(i)
        await asyncio.sleep(0.1)
    await queue.put(None)  # sentinel

async def consumer(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        queue.task_done()

async def main():
    queue: asyncio.Queue = asyncio.Queue()
    await asyncio.gather(producer(queue), consumer(queue))
```

---

## Error Handling in Async Code

```python
async def risky_operation() -> str:
    raise ValueError("something went wrong")

async def main():
    try:
        result = await risky_operation()
    except ValueError as e:
        print(f"Caught: {e}")

# With gather — by default, one failure cancels others
results = await asyncio.gather(task_a(), task_b(), return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        print(f"Task failed: {result}")
    else:
        print(f"Task succeeded: {result}")
```

---

## Synchronous vs Asynchronous

| Scenario | Use |
|----------|-----|
| Network requests (many concurrent) | `async def` + `httpx.AsyncClient` |
| Database queries | `async def` + async ORM (SQLAlchemy async, Tortoise) |
| File I/O (large files) | `async def` + `aiofiles` |
| CPU computation (image processing, ML inference) | `concurrent.futures.ProcessPoolExecutor` |
| Mixing sync library with async | `loop.run_in_executor()` |

### Running sync code from async context

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def blocking_function(x: int) -> int:
    import time
    time.sleep(1)
    return x * 2

async def main():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, blocking_function, 42)
    print(result)  # 84
```

---

## Common Pitfalls

### 1. Forgetting `await`

```python
# Wrong — creates coroutine object but doesn't run it
result = fetch_data()  # returns <coroutine object>

# Correct
result = await fetch_data()
```

### 2. Using blocking I/O in async functions

```python
# Wrong — blocks the event loop
async def bad():
    import time
    time.sleep(5)      # blocks ALL other coroutines

# Correct
async def good():
    await asyncio.sleep(5)  # yields control to event loop
```

### 3. Calling asyncio.run() inside a running event loop

```python
# Wrong — raises RuntimeError in Jupyter / FastAPI
asyncio.run(my_coroutine())

# Correct — inside an already-async context, just await
await my_coroutine()
```
