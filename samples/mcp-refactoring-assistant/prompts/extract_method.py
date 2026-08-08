"""Extract Method refactoring prompt."""

from typing import List
from mcp.server.fastmcp.prompts.base import Message, UserMessage


def extract_method(code: str, target_lines: str = "auto") -> List[Message]:
    """Guide for extracting a method from complex code.

    Args:
        code: Full Python function/method code to refactor
        target_lines: Lines to extract (e.g., "10-25" or "auto" for suggestions)

    Returns:
        Structured guidance messages for Extract Method refactoring
    """

    guidance = f"""# Extract Method Refactoring Guide

## Overview
The Extract Method refactoring helps reduce function complexity by moving a coherent block of code into its own function with a descriptive name.

## Your Code
```python
{code}
```

## Target Lines
{target_lines}

## Step-by-Step Process

### 1. Identify the Extraction Boundary
- Look for a block of code that performs a single, clear task
- The block should be relatively independent
- Ideal candidates: complex calculations, data transformations, validation logic

### 2. Determine Parameters
Identify variables used in the target code that are:
- **Defined outside** the block → These become parameters
- **Modified and used after** → These need to be returned

Example:
```python
# Before: Variables a, b are from outside
result = a + b
result = result * 2
# If 'result' is used later, it must be returned
```

### 3. Determine Return Value
- If one variable is modified and used afterward → return it
- If multiple variables → return a tuple or dict
- If nothing is used afterward → no return needed

### 4. Choose a Descriptive Name
Follow PEP 8 conventions:
- Use `snake_case` for function names
- Name should describe WHAT the function does, not HOW
- Be specific: `calculate_total_price()` not `do_calculation()`

Good examples:
- `validate_user_input()`
- `format_address_string()`
- `calculate_discount_amount()`

### 5. Extract the Method
```python
# Before: Long function with mixed responsibilities
def process_order(order_data):
    # ... 20 lines of code ...

    # Calculate total (this can be extracted)
    subtotal = sum(item['price'] * item['qty'] for item in order_data['items'])
    tax = subtotal * 0.08
    shipping = 10 if subtotal < 50 else 0
    total = subtotal + tax + shipping

    # ... 20 more lines ...
    return result

# After: Extracted calculation logic
def calculate_order_total(items):
    subtotal = sum(item['price'] * item['qty'] for item in items)
    tax = subtotal * 0.08
    shipping = 10 if subtotal < 50 else 0
    return subtotal + tax + shipping

def process_order(order_data):
    # ... 20 lines ...
    total = calculate_order_total(order_data['items'])
    # ... 20 more lines ...
    return result
```

## When to Use Extract Method
✅ Function is longer than 30-40 lines
✅ Code block has a clear, single purpose
✅ Code block could be reused elsewhere
✅ Complex logic that deserves a descriptive name
✅ Nested loops or deep conditionals

## When NOT to Use
❌ Block is only 2-3 lines and already clear
❌ Block is tightly coupled with surrounding code
❌ Would need to pass >5 parameters
❌ Name would be as vague as the current code

## Testing After Refactoring
1. Run existing tests to ensure behavior unchanged
2. Test edge cases for the new function
3. Verify the extracted function can work independently

## Next Steps
1. Identify the code block to extract
2. List variables that need to be parameters
3. Determine what (if anything) to return
4. Choose a clear, descriptive name
5. Create the new function
6. Replace the original code with a function call
7. Test thoroughly"""

    return [UserMessage(content=guidance)]
