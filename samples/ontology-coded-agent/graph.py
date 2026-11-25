from langchain.agents import create_agent

from uipath_langchain.agent.tools import create_ontology_tool
from uipath_langchain.chat import UiPathChat

llm = UiPathChat(model="gpt-4.1-mini-2025-04-14")
system_prompt = "Answer questions using only the configured ontology."

ontology_tool = create_ontology_tool(
    llm=llm,
    name="query_invoice_processing",
    description="Query invoices, suppliers and payments in the ontology.",
    ontology_id="b4bcf8d9-0361-46ce-b87a-f00e05f5d14c",
    base_system_prompt=system_prompt,
)

graph = create_agent(llm, tools=[ontology_tool], system_prompt=system_prompt)
