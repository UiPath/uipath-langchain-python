#!/usr/bin/env python3
"""
Simple test script that imports and uses functions from task_controller.py
"""

import asyncio
import json
import sys
import os

# Add the path to import from task_controller
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'overwatch-local-mcp'))

async def test_task_functions():
    """Test the task controller functions directly."""
    
    print("=== Testing Task Controller Functions ===\n")
    
    try:
        # Import the functions from task_controller
        from task_controller import get_tasks, get_unassigned_tasks, search_task_users, assign_task, add_note_to_task, get_task_data
        
        # Test 1: Get tasks
        print("1. Testing get_tasks...")
        result = await get_tasks(top=3)
        print(f"Result: {json.dumps(result, indent=2)}")
        
        print("\n" + "-" * 40 + "\n")
        
        # Test 2: Get unassigned tasks
        print("2. Testing get_unassigned_tasks...")
        result = await get_unassigned_tasks(top=3)
        print(f"Result: {json.dumps(result, indent=2)}")
        
        print("\n" + "-" * 40 + "\n")
        
        # Test 3: Search for users
        print("3. Testing search_task_users...")
        result = await search_task_users(search_term="admin", top=3)
        print(f"Result: {json.dumps(result, indent=2)}")
        
        print("\n" + "-" * 40 + "\n")
        
        # Test 4: Get task data (using a sample task ID)
        print("4. Testing get_task_data...")
        result = await get_task_data(task_id=12345)
        print(f"Result: {json.dumps(result, indent=2)}")
        
        print("\n" + "-" * 40 + "\n")
        
        # Test 5: Add note to task
        print("5. Testing add_note_to_task...")
        result = await add_note_to_task(task_id=12345, note="Test note from direct import")
        print(f"Result: {json.dumps(result, indent=2)}")
        
        print("\n" + "-" * 40 + "\n")
        
        # Test 6: Assign task
        print("6. Testing assign_task...")
        result = await assign_task(
            task_id=12345,
            assignment_criteria="SingleUser",
            assignees="123",
            reassign=False
        )
        print(f"Result: {json.dumps(result, indent=2)}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_task_functions()) 