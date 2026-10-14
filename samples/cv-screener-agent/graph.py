"""CV screener agent whose guardrails read the attached files.

A coded port of the low-code CV screener agent in the Agents e2e suite, with the
guardrails split between the decorator and the middleware flavors. README.md
explains what each guardrail checks and how to run the fixtures.
"""

import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from uipath.agent.models.agent import (
    AgentInternalAnalyzeFilesToolProperties,
    AgentInternalCreateFileToolProperties,
    AgentInternalToolResourceConfig,
)
from uipath.core.guardrails import GuardrailScope
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.tools.internal_tools.analyze_files_tool import (
    create_analyze_file_tool,
)
from uipath_langchain.agent.tools.internal_tools.create_file_tool import (
    create_file_tool,
)
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    JOB_ATTACHMENT_DEFINITION,
)
from uipath_langchain.chat import UiPathChat
from uipath_langchain.chat.openai import UiPathChatOpenAI
from uipath_langchain.guardrails import (
    BlockAction,
    DetectionScope,
    GuardrailExecutionStage,
    HarmfulContentEntity,
    HarmfulContentValidator,
    IntellectualPropertyValidator,
    LLMAsJudgeValidator,
    LogAction,
    LoggingSeverityLevel,
    PromptInjectionValidator,
    UiPathHarmfulContentMiddleware,
    UiPathIntellectualPropertyMiddleware,
    UiPathLLMAsJudgeMiddleware,
    UiPathPromptInjectionMiddleware,
    guardrail,
    render_attachments_block,
)
from uipath_langchain.guardrails.enums import (
    HarmfulContentEntityType,
    IntellectualPropertyEntityType,
)

AGENT_MODEL = "gpt-4o-2024-11-20"
JUDGE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

PARITY_CHECK = os.environ.get("PARITY_CHECK") == "1"
DETECTION_SCOPE = DetectionScope(os.environ.get("GUARDRAILS_DETECTION_SCOPE", "Files"))

SYSTEM_PROMPT = (
    "You are an HR assistant that screens job applications. You only learn what a "
    "file contains by reading it with the Analyze Files tool, so always call "
    "Analyze Files on the file you are given before you answer. Never judge a file "
    "by its name. If the recruiter notes contain a code sample, save it with the "
    "Create File tool as code_sample.py and return the created file in codeSample. "
    "The notes write each line break as \\n: in the file, turn each \\n into a real "
    "line break and keep everything else unchanged."
)

PROMPT_INJECTION = "Prompt injection in attached files"
HARMFUL_CONTENT = "Harmful content in attached files"
OUTSIDE_EU = "Candidate based outside the EU"
ONLY_CVS = "Only CVs and cover letters"
PROTECTED_CODE = "Protected code in output files"

BLOCK_INJECTION = BlockAction(
    detail="An attached document contains hidden instructions."
)
BLOCK_HARMFUL = BlockAction(
    detail="An attached document contains threatening or harmful content."
)
LOG_WARNING = LogAction(severity_level=LoggingSeverityLevel.WARNING)
BLOCK_NOT_A_CV = BlockAction(detail="The attached file is not a CV or cover letter.")
BLOCK_PROTECTED_CODE = BlockAction(
    detail="A file the agent produced contains protected code."
)

HARMFUL_ENTITIES = [
    HarmfulContentEntity(entity, threshold=2)
    for entity in (
        HarmfulContentEntityType.HATE,
        HarmfulContentEntityType.SELF_HARM,
        HarmfulContentEntityType.SEXUAL,
        HarmfulContentEntityType.VIOLENCE,
    )
]

OUTSIDE_EU_RULE = """You are checking a candidate's CV for an EU-based role.

FAIL if the CV states that the candidate's current residence (home address, current city, or current employer location) is in a country that is NOT a member state of the European Union. Switzerland, Norway, the United Kingdom, the United States, Canada and all other non-member countries count as outside the EU.

PASS if the current residence is in an EU member state, or if the document gives no current residence at all, including a document that is not a CV. Nationality, place of birth, past employers and past studies do not matter; only the current residence does.

Content to evaluate:
{{content}}

Respond with:
- verdict: "fail" or "pass"
- reason: the country you identified as the current residence, or "no location stated"."""
OUTSIDE_EU_POSITIVE = [
    "Address: Rua Augusta 12, Lisbon, Portugal. Current role: Senior Engineer at "
    "Acme Lisboa.\nVerdict: pass\nReason: Portugal",
    "Based in Berlin since 2021; previously worked in New York.\nVerdict: pass\n"
    "Reason: Germany",
]
OUTSIDE_EU_NEGATIVE = [
    "Address: 500 King St W, Toronto, ON, Canada. Current role: Staff Engineer at "
    "Maple Systems, Toronto.\nVerdict: fail\nReason: Canada",
    "Currently in Zürich, Switzerland; Italian citizen.\nVerdict: fail\n"
    "Reason: Switzerland",
]

ONLY_CVS_RULE = """You are checking a file handed to a CV screening tool.

PASS if the file is a curriculum vitae, résumé, or cover letter: a document about one person's work history, skills, education, or motivation for a job. A summary or analysis of such a document also passes.

FAIL if the file is any other kind of document, for example a recipe, an invoice, a contract, a news article, source code, or meeting notes.

Content to evaluate:
{{content}}

Respond with:
- verdict: "fail" or "pass"
- reason: the kind of document you identified."""
ONLY_CVS_POSITIVE = [
    "Mariana Costa, Senior Software Engineer. Experience: Northwind Analytics "
    "2022 to present. Skills: Python, Kafka.\nVerdict: pass\n"
    "Reason: curriculum vitae",
    "Dear Hiring Team, I am applying for the Senior Python Engineer position..."
    "\nVerdict: pass\nReason: cover letter",
]
ONLY_CVS_NEGATIVE = [
    "Classic Lasagna (serves 6). Ingredients: 12 lasagna sheets, 500 g minced "
    "beef...\nVerdict: fail\nReason: recipe",
    "Invoice #2041, due 30 days, total EUR 4,200.\nVerdict: fail\nReason: invoice",
]

ANALYZE_FILES = AgentInternalToolResourceConfig(
    name="Analyze Files",
    description=(
        "Analyze one or more files with an LLM to extract, synthesize, or answer "
        "queries about their content."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "attachments": {
                "type": "array",
                "description": (
                    "Array of files, documents, images, or other attachments to process"
                ),
                "items": {"$ref": "#/definitions/job-attachment"},
            },
            "analysisTask": {
                "type": "string",
                "description": (
                    "The task, question, or instruction for processing the files "
                    "(e.g., 'summarize this document', 'extract key points', "
                    "'what is in this image')"
                ),
            },
        },
        "required": ["attachments", "analysisTask"],
        "definitions": {"job-attachment": JOB_ATTACHMENT_DEFINITION},
    },
    output_schema={
        "type": "object",
        "properties": {
            "analysisResult": {
                "type": "string",
                "description": (
                    "Analysis result of the attachments (analysis, summary, "
                    "extracted information, etc.)"
                ),
            }
        },
        "required": ["analysisResult"],
    },
    properties=AgentInternalAnalyzeFilesToolProperties(),
)

CREATE_FILE = AgentInternalToolResourceConfig(
    name="Create File",
    description="Create a file to pass to another tool or return as output.",
    input_schema={
        "type": "object",
        "properties": {
            "fileName": {
                "type": "string",
                "description": (
                    "Name of the file to create, including its extension (e.g., "
                    "'summary.md' or 'accounts.csv'). The extension sets the file "
                    "type, so it must match the content."
                ),
            },
            "content": {"type": "string", "description": "Text content of the file."},
        },
        "required": ["fileName"],
    },
    properties=AgentInternalCreateFileToolProperties(),
)


class Input(BaseModel):
    """A job application: the CV file and optional recruiter notes."""

    file: Attachment
    recruiterNotes: str = Field(
        default="",
        description=(
            "Optional notes from the recruiter, such as a code sample the candidate "
            "sent"
        ),
    )


class Output(BaseModel):
    """The CV summary, plus the file saved from a code sample in the notes."""

    summary: str = Field(description="A short summary of the candidate's CV")
    codeSample: Attachment | None = Field(
        default=None,
        description="The code sample from the recruiter notes, saved as a file",
    )


@guardrail(
    validator=PromptInjectionValidator(threshold=0.5),
    action=BLOCK_INJECTION,
    stage=GuardrailExecutionStage.PRE,
    name=PROMPT_INJECTION,
    detection_scope=DETECTION_SCOPE,
)
def make_llm() -> UiPathChat:
    """Create the agent's model; every call to it is checked first."""
    return UiPathChat(model=AGENT_MODEL, temperature=0)


llm = make_llm()
analyze_files = guardrail(
    validator=LLMAsJudgeValidator(
        guardrail_text=ONLY_CVS_RULE,
        model=JUDGE_MODEL,
        threshold=2,
        positive_examples=ONLY_CVS_POSITIVE,
        negative_examples=ONLY_CVS_NEGATIVE,
    ),
    action=BLOCK_NOT_A_CV,
    stage=GuardrailExecutionStage.PRE,
    name=ONLY_CVS,
    detection_scope=DETECTION_SCOPE,
)(
    create_analyze_file_tool(
        ANALYZE_FILES, UiPathChatOpenAI(model=AGENT_MODEL, temperature=0)
    )
)
create_file = create_file_tool(CREATE_FILE)

middleware = [
    *UiPathHarmfulContentMiddleware(
        name=HARMFUL_CONTENT,
        scopes=[GuardrailScope.AGENT],
        stage=GuardrailExecutionStage.PRE,
        action=BLOCK_HARMFUL,
        entities=HARMFUL_ENTITIES,
        detection_scope=DETECTION_SCOPE,
    ),
    *UiPathLLMAsJudgeMiddleware(
        name=OUTSIDE_EU,
        scopes=[GuardrailScope.AGENT],
        stage=GuardrailExecutionStage.PRE,
        action=LOG_WARNING,
        guardrail_text=OUTSIDE_EU_RULE,
        model=JUDGE_MODEL,
        threshold=2,
        positive_examples=OUTSIDE_EU_POSITIVE,
        negative_examples=OUTSIDE_EU_NEGATIVE,
        detection_scope=DETECTION_SCOPE,
    ),
    *UiPathIntellectualPropertyMiddleware(
        name=PROTECTED_CODE,
        scopes=[GuardrailScope.AGENT],
        action=BLOCK_PROTECTED_CODE,
        entities=[IntellectualPropertyEntityType.CODE],
        detection_scope=DETECTION_SCOPE,
    ),
]

if PARITY_CHECK:
    middleware.extend(
        UiPathPromptInjectionMiddleware(
            name=f"{PROMPT_INJECTION} (middleware)",
            action=BLOCK_INJECTION,
            threshold=0.5,
            detection_scope=DETECTION_SCOPE,
        )
    )
    middleware.extend(
        UiPathLLMAsJudgeMiddleware(
            name=f"{ONLY_CVS} (middleware)",
            scopes=[GuardrailScope.TOOL],
            tools=[analyze_files],
            stage=GuardrailExecutionStage.PRE,
            action=BLOCK_NOT_A_CV,
            guardrail_text=ONLY_CVS_RULE,
            model=JUDGE_MODEL,
            threshold=2,
            positive_examples=ONLY_CVS_POSITIVE,
            negative_examples=ONLY_CVS_NEGATIVE,
            detection_scope=DETECTION_SCOPE,
        )
    )


def make_agent() -> CompiledStateGraph:
    """Create the guarded agent."""
    return create_agent(
        model=llm,
        tools=[analyze_files, create_file],
        system_prompt=SYSTEM_PROMPT,
        response_format=Output,
        middleware=middleware,
    )


if PARITY_CHECK:
    make_agent = guardrail(
        validator=IntellectualPropertyValidator(
            entities=[IntellectualPropertyEntityType.CODE]
        ),
        action=BLOCK_PROTECTED_CODE,
        stage=GuardrailExecutionStage.POST,
        name=f"{PROTECTED_CODE} (decorator)",
        detection_scope=DETECTION_SCOPE,
    )(make_agent)
    make_agent = guardrail(
        validator=LLMAsJudgeValidator(
            guardrail_text=OUTSIDE_EU_RULE,
            model=JUDGE_MODEL,
            threshold=2,
            positive_examples=OUTSIDE_EU_POSITIVE,
            negative_examples=OUTSIDE_EU_NEGATIVE,
        ),
        action=LOG_WARNING,
        stage=GuardrailExecutionStage.PRE,
        name=f"{OUTSIDE_EU} (decorator)",
        detection_scope=DETECTION_SCOPE,
    )(make_agent)
    make_agent = guardrail(
        validator=HarmfulContentValidator(entities=HARMFUL_ENTITIES),
        action=BLOCK_HARMFUL,
        stage=GuardrailExecutionStage.PRE,
        name=f"{HARMFUL_CONTENT} (decorator)",
        detection_scope=DETECTION_SCOPE,
    )(make_agent)

agent = make_agent()


async def screen_cv(state: Input) -> Output:
    """Show the model the CV as an attachments block and return its screening."""
    attachment = state.file.model_dump(mode="json", exclude_none=True)
    prompt = (
        "Summarise the candidate's CV in the attached file using the Analyze Files "
        f"tool.\n{render_attachments_block([attachment])}\n"
        f"Recruiter notes: {state.recruiterNotes}"
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    return result["structured_response"]


builder = StateGraph(Input, input_schema=Input, output_schema=Output)
builder.add_node("screen_cv", screen_cv)
builder.add_edge(START, "screen_cv")
builder.add_edge("screen_cv", END)

graph = builder.compile()
