# Pydantic v2 — Complete Guide

## What is Pydantic?

Pydantic is the most widely used data validation library for Python. Using Python type annotations, Pydantic enforces type hints at runtime and provides user-friendly errors when data is invalid.

Version 2 was released in 2023 and is significantly faster than v1 (up to 50× in benchmarks) thanks to a Rust core.

---

## Installation

```bash
pip install pydantic
```

---

## Basic Models

Define a model by subclassing `BaseModel`:

```python
from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    is_active: bool = True

# Create an instance — Pydantic validates types
user = User(id=1, name="Alice", email="alice@example.com", age=30)
print(user.name)  # Alice
print(user.model_dump())  # {'id': 1, 'name': 'Alice', ...}
```

---

## Field Customisation

Use `Field` to add metadata, constraints, and defaults:

```python
from pydantic import BaseModel, Field
from typing import Optional

class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Product name")
    price: float = Field(..., gt=0, description="Price must be positive")
    discount: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    sku: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}-\d{4}$")
```

### Field constraints

| Constraint | Meaning              | Types         |
|-----------|----------------------|---------------|
| `gt`      | greater than         | numeric       |
| `ge`      | greater than or equal | numeric      |
| `lt`      | less than            | numeric       |
| `le`      | less than or equal   | numeric       |
| `min_length` | minimum length    | str, list     |
| `max_length` | maximum length    | str, list     |
| `pattern` | regex pattern        | str           |

---

## Validators

### Field validators (v2 style)

```python
from pydantic import BaseModel, field_validator

class UserModel(BaseModel):
    username: str
    email: str
    age: int

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        assert v.isalnum(), "Username must be alphanumeric"
        return v.lower()  # transform

    @field_validator("email")
    @classmethod
    def email_must_contain_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email address")
        return v

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age cannot be negative")
        return v
```

### Model validators (cross-field validation)

```python
from pydantic import BaseModel, model_validator

class PasswordForm(BaseModel):
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def passwords_match(self) -> "PasswordForm":
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self
```

---

## Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str = "US"

class Person(BaseModel):
    name: str
    address: Address

person = Person(
    name="Bob",
    address={"street": "123 Main St", "city": "Springfield"},
)
print(person.address.city)  # Springfield
```

---

## Serialisation

```python
user = User(id=1, name="Alice", email="alice@example.com")

# To dict
d = user.model_dump()
# {'id': 1, 'name': 'Alice', 'email': 'alice@example.com', ...}

# To JSON string
j = user.model_dump_json()

# To dict, excluding None values
d = user.model_dump(exclude_none=True)

# To dict, including only specific fields
d = user.model_dump(include={"id", "name"})
```

---

## Parsing / Deserialisation

```python
# From dict
user = User.model_validate({"id": 1, "name": "Alice", "email": "a@b.com"})

# From JSON string
user = User.model_validate_json('{"id": 1, "name": "Alice", "email": "a@b.com"}')
```

---

## Settings Management (pydantic-settings)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "My App"
    debug: bool = False
    database_url: str
    secret_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

settings = Settings()  # reads from environment + .env file
```

---

## Generic Models

```python
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")

class Response(BaseModel, Generic[T]):
    data: T
    message: str
    success: bool = True

# Use with any type
int_response = Response[int](data=42, message="OK")
user_response = Response[User](data=user, message="Found user")
```

---

## Computed Fields

```python
from pydantic import BaseModel, computed_field

class Rectangle(BaseModel):
    width: float
    height: float

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height
```

---

## Model Config

```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,             # no implicit coercion
        frozen=True,             # immutable instances
        extra="forbid",          # reject extra fields
        populate_by_name=True,   # allow field names AND aliases
        str_strip_whitespace=True,
    )

    name: str
    value: int
```

---

## Discriminated Unions

Type-safe handling of multiple possible schemas:

```python
from typing import Literal, Union
from pydantic import BaseModel

class Cat(BaseModel):
    animal_type: Literal["cat"]
    meows: int

class Dog(BaseModel):
    animal_type: Literal["dog"]
    barks: int

class Zoo(BaseModel):
    animals: list[Union[Cat, Dog]]

zoo = Zoo(animals=[
    {"animal_type": "cat", "meows": 3},
    {"animal_type": "dog", "barks": 7},
])
```
