"""Improve Naming refactoring prompt."""

from typing import List
from mcp.server.fastmcp.prompts.base import Message, UserMessage


def improve_naming(code: str, symbols: str, context: str = "") -> List[Message]:
    """Guide for improving variable and function names.

    Args:
        code: Python code with poor names
        symbols: Symbols to rename (comma-separated)
        context: Optional description of what the code does

    Returns:
        Structured guidance for improving names
    """
    symbols_list = [s.strip() for s in symbols.split(',') if s.strip()]

    guidance = rf"""# Improve Naming Guide

## Overview
Good names are the most important aspect of readable code. Names should reveal intent without requiring comments.

## Your Code
```python
{code}
```

## Code Context
{context if context else "General naming improvements"}

## Symbols to Rename
{', '.join(f'`{s}`' for s in symbols_list)}

## Python Naming Conventions (PEP 8)

### Functions and Variables: `snake_case`
```python
# Good
user_count = 0
def calculate_total_price():
    ...

# Bad
userCount = 0  # camelCase is for JavaScript
def CalculateTotalPrice():  # PascalCase is for classes
```

### Classes: `PascalCase`
```python
# Good
class UserAccount:
    ...

# Bad
class user_account:  # Should be PascalCase
class User_Account:  # No underscores in PascalCase
```

### Constants: `UPPER_SNAKE_CASE`
```python
# Good
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30

# Bad
max_connections = 100  # Should be uppercase for constants
```

### Private/Internal: Leading underscore
```python
class MyClass:
    def _internal_helper(self):  # Internal method
        ...

    def public_method(self):  # Public method
        ...
```

## Naming Guidelines

### 1. Be Descriptive and Specific
```python
# Bad - too vague
def process(d):
    result = calc(d)
    return result

# Good - clear intent
def calculate_order_total(order_data):
    total_price = sum_item_prices(order_data)
    return total_price
```

### 2. Avoid Abbreviations (except common ones)
```python
# Bad
usr_cnt = 0
tmp_val = x
btn_clk = func()

# Good
user_count = 0
temporary_value = x
button_clicked = func()

# OK (commonly understood)
html_content = ""
url_path = ""
id = 123
max_value = 100
```

### 3. Use Verbs for Functions, Nouns for Variables
```python
# Good functions (verbs)
def calculate_discount():
    ...
def send_email():
    ...
def validate_input():
    ...

# Good variables (nouns)
user_name = "Alice"
total_amount = 100
error_message = "Invalid input"
```

### 4. Boolean Variables Should Ask a Question
```python
# Good
is_valid = True
has_permission = False
can_edit = True
should_retry = False

# Bad
valid = True  # Is it valid or are we validating?
permission = False  # What about permission?
```

### 5. Avoid Single Letter Names (except loops)
```python
# Bad
def f(x, y, z):
    a = x + y
    b = a * z
    return b

# Good (loop counters)
for i in range(10):
    for j in range(5):
        matrix[i][j] = 0

# Good (descriptive)
def calculate_final_price(base_price, tax_rate, quantity):
    subtotal = base_price + (base_price * tax_rate)
    final_price = subtotal * quantity
    return final_price
```

### 6. Collections Should Be Plural
```python
# Good
users = [user1, user2, user3]
email_addresses = []
error_messages = {{}}

# Bad
user = [user1, user2, user3]  # Misleading
email_address = []  # Singular for collection
```

## Suggested Renamings

Here are some suggestions based on common patterns:

"""

    # Generate suggestions for common bad names
    suggestions = {
        'x': 'Consider: value, count, index, total (depending on usage)',
        'y': 'Consider: result, amount, price, quantity',
        'tmp': 'Consider: temporary_value, intermediate_result, buffer',
        'data': 'Consider: user_data, order_data, response_data (be specific)',
        'func': 'Consider: callback, handler, processor (what does it do?)',
        'func1': 'Consider: validate_input, process_order (describe action)',
        'val': 'Consider: value, amount, price (full word)',
        'arr': 'Consider: items, users, values (plural noun)',
        'obj': 'Consider: user, order, product (specific type)',
        'res': 'Consider: result, response, resource (full word)',
        'str': 'Consider: message, name, content (avoid type names)',
        'num': 'Consider: count, total, amount (be specific)',
        'i': 'OK for loop counter, otherwise: index, item_id, position',
        'j': 'OK for nested loop, otherwise: consider better name',
        'k': 'OK for third loop, otherwise: use descriptive name',
    }

    for symbol in symbols_list:
        lower_symbol = symbol.lower()
        if lower_symbol in suggestions:
            guidance += f"\n**`{symbol}`** → {suggestions[lower_symbol]}"
        else:
            guidance += f"\n**`{symbol}`** → Consider a name that describes its purpose in the context of {context if context else 'your code'}"

    guidance += r"""

## Refactoring Process

### 1. Understand the Purpose
- What does this variable/function actually do?
- What value does it hold?
- What action does it perform?

### 2. Choose a Clear Name
- Use full words, not abbreviations
- Be specific about what it represents
- Follow PEP 8 conventions

### 3. Rename Safely
```python
# Use find-and-replace carefully
# 1. Search for exact matches (whole word)
# 2. Review each occurrence
# 3. Update all usages consistently
```

### 4. Update Related Code
- Update comments if any
- Update docstrings
- Update tests

### 5. Run Tests
- Ensure no regressions
- Verify all references updated

## Examples of Good Naming

```python
# E-commerce context
def calculate_order_total(items, tax_rate, discount_code=None):
    subtotal = sum(item.price * item.quantity for item in items)
    discount_amount = apply_discount(subtotal, discount_code)
    tax_amount = (subtotal - discount_amount) * tax_rate
    final_total = subtotal - discount_amount + tax_amount
    return final_total
```

```python
# User validation
def is_valid_email(email_address):
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(email_pattern, email_address) is not None

def can_user_edit_post(user, post):
    is_post_author = user.id == post.author_id
    has_admin_rights = user.role == 'admin'
    return is_post_author or has_admin_rights
```

## Remember
- Names should make code self-documenting
- If you need a comment to explain a variable, the name is wrong
- Good names are worth the extra characters
- Consistency matters across your codebase

## Next Steps
1. Understand what each symbol represents
2. Choose descriptive, PEP 8 compliant names
3. Use find-and-replace to rename consistently
4. Update docstrings and comments
5. Run tests to ensure nothing broke"""

    return [UserMessage(content=guidance)]
