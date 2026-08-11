from langchain.agents import create_agent
from uipath.platform.entities import DataFabricEntityItem

from uipath_langchain.agent.tools import create_datafabric_tool
from uipath_langchain.chat import UiPathChat

llm = UiPathChat(model="gpt-4.1-mini-2025-04-14")
system_prompt = "Answer questions using only the configured Data Fabric entities."

datafabric_tool = create_datafabric_tool(
    llm=llm,
    name="query_agent_test",
    description="Query the agentTest Data Fabric entity.",
    base_system_prompt=system_prompt,
    entities=[
        DataFabricEntityItem(
            id="1312e893-8295-f111-9b33-0022482a9eea",
            name="agentTest",
            folder_key="379fec63-62b1-41ec-b2fc-718f8f7dda3c",
        )
    ],
)

graph = create_agent(llm, tools=[datafabric_tool], system_prompt=system_prompt)
