#!/usr/bin/env python3
"""
Simple test to assign task 4651309 to user ID 3348246 using task_controller functions.
"""

import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Add the src directory to the path so we can import the task controller
sys.path.append('src/overwatch-local-mcp')

# Load environment variables
load_dotenv()

async def assign_simple_task():
    """Assign task 4651309 twice."""
    
    # Import the functions from task_controller
    from task_controller import assign_task, get_task_data
    
    task_id = 4651309
    user_id_1 = 3348246
    user_id_2 = 3540352
    
    print("=== Double Task Assignment Test ===")
    print(f"Task ID: {task_id}")
    print(f"First User ID: {user_id_1}")
    print(f"Second User ID: {user_id_2}")
    print()
    
    try:
        # Get task details BEFORE first assignment
        print("📋 Getting task details BEFORE first assignment...")
        before_first = await get_task_data(task_id=task_id)
        print(f"Task details BEFORE first assignment: {json.dumps(before_first, indent=2)}")
        print()
        
        # First assignment
        print("🔄 First assignment...")
        assign_result_1 = await assign_task(
            task_id=task_id,
            assignment_criteria="SingleUser",
            assignees=str(user_id_1),
            reassign=True
        )
        print(f"First assignment result: {json.dumps(assign_result_1, indent=2)}")
        print()
        
        # Get task details BETWEEN assignments
        print("📋 Getting task details BETWEEN assignments...")
        between_result = await get_task_data(task_id=task_id)
        print(f"Task details BETWEEN assignments: {json.dumps(between_result, indent=2)}")
        print()
        
        # Second assignment
        print("🔄 Second assignment...")
        assign_result_2 = await assign_task(
            task_id=task_id,
            assignment_criteria="SingleUser",
            assignees=str(user_id_2),
            reassign=True
        )
        print(f"Second assignment result: {json.dumps(assign_result_2, indent=2)}")
        print()
        
        # Get task details AFTER second assignment
        print("📋 Getting task details AFTER second assignment...")
        after_second = await get_task_data(task_id=task_id)
        print(f"Task details AFTER second assignment: {json.dumps(after_second, indent=2)}")
        
        if assign_result_1.get("success") and assign_result_2.get("success"):
            print("\n✅ Both assignments completed successfully!")
        else:
            print(f"\n❌ Assignment failed: First: {assign_result_1.get('error', 'Unknown')}, Second: {assign_result_2.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def print_environment_info():
    """Print current environment configuration."""
    print("=== Environment Configuration ===")
    print(f"UIPATH_BASE_URL: {os.getenv('UIPATH_BASE_URL', 'Not set')}")
    print(f"UIPATH_ORG_ID: {os.getenv('UIPATH_ORG_ID', 'Not set')}")
    print(f"UIPATH_TENANT: {os.getenv('UIPATH_TENANT', 'Not set')}")
    print(f"UIPATH_ACCESS_TOKEN: {'Set' if os.getenv('UIPATH_ACCESS_TOKEN') else 'Not set'}")
    print()

async def main():
    """Main test function."""
    print("🚀 Starting Simple Task Assignment Test\n")
    
    # Print environment info
    print_environment_info()
    
    # Assign the task
    await assign_simple_task()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(main()) 