from typing import Optional, List, Literal, Union
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, validator
from uipath import UiPath
from langchain_core.output_parsers import PydanticOutputParser
import logging
import time
from uipath._models import InvokeProcess, IngestionInProgressException
from langchain_core.messages import HumanMessage, SystemMessage
from uipath_langchain.retrievers import ContextGroundingRetriever
from langchain_anthropic import ChatAnthropic


logger = logging.getLogger(__name__)

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

class QuizItem(BaseModel):
    question: str = Field(
        description="One quiz question"
    )
    difficulty: float = Field(
        description="How difficult is the question", ge=0.0, le=1.0
    )
    answer: str = Field(
        description="The expected answer to the question",
    )
class Quiz(BaseModel):
   quiz_items: List[QuizItem] = Field(
        description="A list of quiz items"
    )
class QuizOrInsufficientInfo(BaseModel):
    quiz: Optional[Quiz] = Field(
        description="A quiz based on user input and available documents."
    )
    additional_info: Optional[str] = Field(
        description="String that controls whether additional information is required",
    )

    @validator("quiz", always=True)
    def check_quiz(cls, v, values):
        if values.get("additional_info") == "false" and v is None:
            raise ValueError("Quiz should be None when additional_info is not 'false'")
        return v

output_parser = PydanticOutputParser(pydantic_object=QuizOrInsufficientInfo)

system_message ="""You are a quiz generator. Try to generate a quiz about {quiz_topic} with multiple questions ONLY based on the following documents. Do not use any extra knowledge.
If the documents do not provide enough info, respond with additional_info=<information that is required>.  
If they provide enough info, create the quiz and set additional_info='false'

{context}

{format_instructions}

Respond with the classification in the requested JSON format."""

uipath = UiPath()


class GraphOutput(BaseModel):
    quiz: Quiz

class GraphInput(BaseModel):
    general_category: str
    quiz_topic: str
    bucket_name: str
    index_name: str
    bucket_folder: Optional[str]

class GraphState(MessagesState):
    general_category: str
    quiz_topic: str
    bucket_name: str
    bucket_folder: Optional[str]
    index_name: str
    additional_info: Optional[bool]
    quiz: Optional[Quiz]

def prepare_input(state: GraphInput) -> GraphState:
    return GraphState(
        quiz_topic=state.quiz_topic,
        bucket_name=state.bucket_name,
        index_name=state.index_name,
        general_category=state.general_category,
        additional_info="false",
    )

async def invoke_researcher(state: GraphState) -> Command:
    print("INVOKE RESEARCHER")
    if state.get("additional_info", None) != "false":
        state["messages"].append(HumanMessage(f"{state['additional_info']}")),
    else:
        state["messages"].append(HumanMessage(f"Fetch data about {state['general_category']}")),
    input_args_json = {
            "messages": state["messages"],
            "bucket_name": state["bucket_name"],
            "bucket_folder": state.get("bucket_folder", None),
        }
    agent_response = interrupt(InvokeProcess(
        name = "researcher-and-uploader-agent",
        input_arguments = input_args_json,
    ))
    quiz_topic = state["quiz_topic"]
    return Command(
        update={
            "messages": [agent_response["messages"][-1], ("user", f"create a quiz about {quiz_topic}")],
        })

async def create_quiz(state: GraphState) -> Command:
    print("CREATE QUIZ")
    no_of_retries = 5
    context_data = None
    index = uipath.context_grounding.get_or_create_index(state["index_name"],storage_bucket_name=state["bucket_name"])
    uipath.context_grounding.ingest_data(index)
    while no_of_retries != 0:
        try:
            context_data = ContextGroundingRetriever(
                index_name=state["index_name"],
                uipath_sdk=uipath,
                number_of_results=10
            ).invoke(state["quiz_topic"])
            break
        except IngestionInProgressException as ex:
            logger.info(ex.message)
            no_of_retries -= 1
            logger.info(f"{no_of_retries} retries left")
            time.sleep(5)
    if not context_data:
        raise Exception("Ingestion is taking too long!")

    # state["messages"].append(SystemMessage(system_message.format(format_instructions=output_parser.get_format_instructions(),
    #     context= context_data)))
    print("INVOKE LLM")
    message= system_message.format(format_instructions=output_parser.get_format_instructions(),
        context= context_data,
        quiz_topic=state["quiz_topic"])
    print(message)
    result = llm.invoke(message)
    try:
        llm_response = output_parser.parse(result.content)
        print("LLM RESPONSE")
        print(llm_response)
        print("CONTEXT DATA")
        print(context_data)
        return Command(
            update={
                "quiz": llm_response.quiz if llm_response.additional_info == "false" else None,
                "additional_info": llm_response.additional_info,
            }
        )
    except Exception as e:
        print(f"Failed to parse {e}")
        return Command(goto=END)

def check_quiz_creation(state: GraphState) -> Literal["invoke_researcher", "return_quiz"]:
    print("CHECK QUIZ CREATION")
    print(state["additional_info"])
    if state["additional_info"] != "false":
        return "invoke_researcher"
    return "return_quiz"

def return_quiz(state: GraphState) -> GraphOutput:
    # print("RETURN QUIZ")
    # print(state["quiz"])
    return GraphOutput( quiz=state["quiz"])

# Build the state graph
builder = StateGraph(input=GraphInput, output=GraphOutput)
builder.add_node("invoke_researcher", invoke_researcher)
builder.add_node("create_quiz", create_quiz)
builder.add_node("return_quiz", return_quiz)
builder.add_node("prepare_input", prepare_input)

builder.add_edge(START, "prepare_input")
builder.add_edge("prepare_input", "invoke_researcher")
builder.add_edge("invoke_researcher", "create_quiz")
builder.add_conditional_edges("create_quiz", check_quiz_creation)
builder.add_edge("return_quiz", END)

# Compile the graph
graph = builder.compile()
