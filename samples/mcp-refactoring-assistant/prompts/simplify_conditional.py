"""Simplify Conditional refactoring prompt."""

from typing import List
from mcp.server.fastmcp.prompts.base import Message, UserMessage


def simplify_conditional(code: str, pattern: str = "guard_clause") -> List[Message]:
    """Guide for simplifying complex conditional logic.

    Args:
        code: Python code with complex conditionals
        pattern: Refactoring pattern (guard_clause, early_return, extract_condition)

    Returns:
        Structured guidance for simplifying conditionals
    """
    pattern_guides = {
        "guard_clause": """
## Guard Clause Pattern

Replace nested if statements with early returns that handle special cases first.

### Before (nested conditionals):
```python
def process_payment(amount, user):
    if user is not None:
        if user.is_active:
            if amount > 0:
                if user.balance >= amount:
                    user.balance -= amount
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False
```

### After (guard clauses):
```python
def process_payment(amount, user):
    # Handle invalid cases first
    if user is None:
        return False
    if not user.is_active:
        return False
    if amount <= 0:
        return False
    if user.balance < amount:
        return False

    # Happy path is now clear and un-nested
    user.balance -= amount
    return True
```

### Benefits:
- Reduces nesting depth
- Makes error conditions explicit
- Happy path is clear and readable
- Each condition is independent and easy to test""",

        "early_return": """
## Early Return Pattern

Exit the function as soon as you know the result, avoiding unnecessary processing.

### Before:
```python
def calculate_discount(customer, order_total):
    discount = 0
    if customer.is_premium:
        if order_total > 100:
            discount = order_total * 0.2
        else:
            discount = order_total * 0.1
    else:
        if order_total > 200:
            discount = order_total * 0.05
    return discount
```

### After:
```python
def calculate_discount(customer, order_total):
    if customer.is_premium:
        if order_total > 100:
            return order_total * 0.2
        return order_total * 0.1

    if order_total > 200:
        return order_total * 0.05

    return 0  # No discount
```

### Benefits:
- Reduces variable assignments
- Makes control flow clearer
- Each path is independent
- Less cognitive load""",

        "extract_condition": """
## Extract Condition Pattern

Move complex boolean expressions into well-named functions.

### Before:
```python
def can_approve_loan(applicant, amount):
    if (applicant.credit_score > 700 and
        applicant.income * 0.3 > amount / 12 and
        applicant.employment_years >= 2 and
        applicant.debt_to_income < 0.4):
        return True
    return False
```

### After:
```python
def has_good_credit(applicant):
    return applicant.credit_score > 700

def can_afford_payment(applicant, monthly_payment):
    return applicant.income * 0.3 > monthly_payment

def has_stable_employment(applicant):
    return applicant.employment_years >= 2

def has_manageable_debt(applicant):
    return applicant.debt_to_income < 0.4

def can_approve_loan(applicant, amount):
    monthly_payment = amount / 12
    return (has_good_credit(applicant) and
            can_afford_payment(applicant, monthly_payment) and
            has_stable_employment(applicant) and
            has_manageable_debt(applicant))
```

### Benefits:
- Complex conditions become self-documenting
- Each condition can be tested independently
- Easy to modify or add new conditions
- Improves readability dramatically"""
    }

    selected_guide = pattern_guides.get(pattern, pattern_guides["guard_clause"])

    guidance = f"""# Simplifying Conditional Logic

## Your Code
```python
{code[:500]}{'...' if len(code) > 500 else ''}
```

## Selected Pattern: {pattern.replace('_', ' ').title()}

{selected_guide}

## General Tips for Simplifying Conditionals

### 1. Avoid Deep Nesting
- Maximum 2-3 levels of nesting
- Use guard clauses for early exits
- Extract complex conditions into functions

### 2. Prefer Positive Conditions
```python
# Less clear
if not user.is_inactive:
    ...

# More clear
if user.is_active:
    ...
```

### 3. Use Lookup Tables for Long If/Elif Chains
```python
# Before
if action == 'create':
    return handle_create()
elif action == 'update':
    return handle_update()
elif action == 'delete':
    return handle_delete()

# After
handlers = {{
    'create': handle_create,
    'update': handle_update,
    'delete': handle_delete,
}}
return handlers[action]()
```

### 4. Consider Polymorphism for Type Checking
```python
# Before
if isinstance(shape, Circle):
    return 3.14 * shape.radius ** 2
elif isinstance(shape, Rectangle):
    return shape.width * shape.height

# After (each class has area() method)
return shape.area()
```

## Next Steps
1. Identify the most deeply nested or complex conditional
2. Choose the appropriate pattern ({', '.join(pattern_guides.keys())})
3. Refactor step by step
4. Test after each change
5. Verify the logic remains the same"""

    return [UserMessage(content=guidance)]
