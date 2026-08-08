"""Remove Duplication refactoring prompt."""

from typing import List
from mcp.server.fastmcp.prompts.base import Message, UserMessage


def remove_duplication(code: str, duplicate_blocks: str) -> List[Message]:
    """Guide for removing code duplication.

    Args:
        code: Python code with duplication
        duplicate_blocks: Description of what's duplicated

    Returns:
        Structured guidance for removing duplication
    """

    guidance = rf"""# Remove Code Duplication Guide

## Overview
Code duplication is a major source of maintenance problems. When you need to fix a bug or add a feature, you must remember to update all copies.

## Your Code
```python
{code}
```

## Duplicated Code Detected
{duplicate_blocks}

## Common Duplication Patterns and Solutions

### 1. Exact Duplication → Extract Function

**Before:**
```python
# In function A
user_data = request.json
if 'email' not in user_data:
    return {{'error': 'Email required'}}, 400
if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', user_data['email']):
    return {{'error': 'Invalid email'}}, 400

# In function B (exact same code)
user_data = request.json
if 'email' not in user_data:
    return {{'error': 'Email required'}}, 400
if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', user_data['email']):
    return {{'error': 'Invalid email'}}, 400
```

**After:**
```python
def validate_email_input(user_data):
    if 'email' not in user_data:
        return {{'error': 'Email required'}}, 400
    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', user_data['email']):
        return {{'error': 'Invalid email'}}, 400
    return None

# In both functions
user_data = request.json
error = validate_email_input(user_data)
if error:
    return error
```

### 2. Similar Code with Variations → Parameterize

**Before:**
```python
def calculate_employee_bonus(employee):
    base_salary = employee.salary
    bonus = base_salary * 0.1
    return bonus

def calculate_manager_bonus(manager):
    base_salary = manager.salary
    bonus = base_salary * 0.15
    return bonus

def calculate_executive_bonus(executive):
    base_salary = executive.salary
    bonus = base_salary * 0.25
    return bonus
```

**After:**
```python
def calculate_bonus(person, bonus_rate):
    return person.salary * bonus_rate

# Usage
employee_bonus = calculate_bonus(employee, 0.1)
manager_bonus = calculate_bonus(manager, 0.15)
executive_bonus = calculate_bonus(executive, 0.25)

# Even better: use a config
BONUS_RATES = {{
    'employee': 0.1,
    'manager': 0.15,
    'executive': 0.25,
}}

def calculate_bonus(person):
    rate = BONUS_RATES[person.role]
    return person.salary * rate
```

### 3. Repeated Patterns → Use Data Structures

**Before:**
```python
if user_type == 'admin':
    permissions = ['read', 'write', 'delete', 'manage_users']
elif user_type == 'editor':
    permissions = ['read', 'write']
elif user_type == 'viewer':
    permissions = ['read']
elif user_type == 'guest':
    permissions = []
```

**After:**
```python
USER_PERMISSIONS = {{
    'admin': ['read', 'write', 'delete', 'manage_users'],
    'editor': ['read', 'write'],
    'viewer': ['read'],
    'guest': [],
}}

permissions = USER_PERMISSIONS.get(user_type, [])
```

### 4. Duplicated Setup/Teardown → Use Decorators or Context Managers

**Before:**
```python
def process_file_a():
    file = open('data.txt')
    try:
        # Process file A
        ...
    finally:
        file.close()

def process_file_b():
    file = open('data.txt')
    try:
        # Process file B
        ...
    finally:
        file.close()
```

**After:**
```python
def process_file_a():
    with open('data.txt') as file:
        # Process file A
        ...

def process_file_b():
    with open('data.txt') as file:
        # Process file B
        ...

# Or create a custom decorator for more complex setup
```

## Steps to Remove Duplication

1. **Identify the duplication precisely**
   - What's exactly the same?
   - What's different (parameters)?

2. **Extract the common code**
   - Create a new function with a descriptive name
   - Pass differences as parameters

3. **Replace all duplicates**
   - Replace each duplicate with a call to the new function
   - Ensure parameters are passed correctly

4. **Test thoroughly**
   - Test each caller of the new function
   - Ensure behavior is identical to before

5. **Consider further abstraction**
   - Can you use data structures instead of code?
   - Would a class or decorator be clearer?

## Warning Signs
- "I need to change this in multiple places"
- Copy-pasting code between functions
- Similar function/variable names (process_a, process_b, process_c)
- Long functions with repeated patterns

## Next Steps
1. Identify the exact duplicated code
2. Determine what varies between copies
3. Extract common code into a function with parameters
4. Replace all duplicates with calls to new function
5. Run tests to verify behavior unchanged"""

    return [UserMessage(content=guidance)]
