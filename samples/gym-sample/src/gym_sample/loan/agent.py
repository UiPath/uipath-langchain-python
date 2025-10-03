from typing import List

from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ..tools import EscalationTool
from ..uipath_gym_types import AgentBaseClass, Datapoint
from .tools import LoanAgent_ApplyUiBankLoanInput, LoanAgent_ModifyLoanContractInput, LoanAgent_SendSharepointDocumentViaDocuSignInput, GoogleInput
from .tools import LoanAgent_ApplyUiBankLoan, LoanAgent_ModifyLoanContract, LoanAgent_SendSharepointDocumentViaDocuSign, Google


SYSTEM_PROMPT = """
You are **Loan Agent**, an AI assistant tasked with automating loan creation in UiBank, modifying the loan contract and sending it to be signed via DocuSign. Your responsibilities include managing requests arrived via email. Follow these guidelines:

1. **Analyze the Email Body**:
   - Carefully read and interpret the email body to gather key information for the loan creation process.

2. **Validate Parameters Before Execution**:
   - For each function, ensure you have all the necessary parameters as per the payloadSchema.
     - If missing values are detected, initiate a task in the Action Center by calling **EscalationTool**.

3. **No Data Generation**:
   - Do **not** generate or assume missing data. All information must come from the **EmailBody** input.

4. **Dynamic Tool Selection Based on Email Input**:
   - Select the appropriate tool combination for each scenario based on the content of **EmailBody**.
    1. Apply to loan using the **LoanAgent_ApplyUiBankLoan** tool.
    2. Check the loan rate on Google using the **Google** tool.
    3. If the rate is less than the one on Google, modify the loan contract using the **LoanAgent_ModifyLoanContract** tool
    send it to be signed via DocuSign**. The location where to save the modified contract is this one: https://uipath-my.sharepoint.com/:f:/r/personal/alina_capota_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e.
    4. Send the loan contract to be signed via DocuSign using the **LoanAgent_SendSharepointDocumentViaDocuSign** tool.

5. Always escalate to a human, using EscalationTool, if:
    - There is any missing information from the request (e.g name, email address, address, loan amount, etc.)
    - There are multiple requests in the same email
    - The request is not clear
    - Input `reason` as a detailed list of missing parameters.
    - Input `assignee` using the email of the end-user.

6. **Execution Outputs**:
   - **ActionCenterTaskCreated**: Set to **True** if an Action Center task is created, otherwise set it to **False**.
   - **ActionCenterTaskURL**: Populate with the task URL if a task is created, otherwise leave blank.
   - **ExecutionSummary**: Provide a numbered step-by-step summary detailing all actions taken, including input/output values.

**Goal**:
   - Optimize and streamline loan creation operations by efficiently automating client registration with minimal user involvement while maintaining accuracy. Ensure that any escalations and additional information requirements are handled smoothly by creating Action Center tasks and resuming the workflow after tasks are completed.
"""


USER_PROMPT = """
The user received the following email from an loan requester:
<EmailBody> {EmailBody} </EmailBody>
"""


class AgentInputSchema(BaseModel):
    EmailBody: str = Field(description="The body of the request email")


class AgentOutputSchema(BaseModel):
    ExecutionDetails: str = Field(description="The execution details")
    ActionCenterTaskCreated: bool | None = Field(
        description="True if an Action Center task is created, otherwise set it to None"
    )
    ActionCenterTaskURL: str | None = Field(
        description="The URL of the Action Center task if it is created, otherwise set it to None"
    )


def get_tools() -> List[BaseTool]:
    return [
        StructuredTool.from_function(
            name="LoanAgent_ApplyUiBankLoan",
            description="Apply for a loan at UI Bank",
            func=LoanAgent_ApplyUiBankLoan,
            args_schema=LoanAgent_ApplyUiBankLoanInput,
        ),
        StructuredTool.from_function(
            name="LoanAgent_ModifyLoanContract",
            description="Modify a loan contract",
            func=LoanAgent_ModifyLoanContract,
            args_schema=LoanAgent_ModifyLoanContractInput,
        ),
        StructuredTool.from_function(
            name="LoanAgent_SendSharepointDocumentViaDocuSign",
            description="Send a sharepoint document via DocuSign",
            func=LoanAgent_SendSharepointDocumentViaDocuSign,
            args_schema=LoanAgent_SendSharepointDocumentViaDocuSignInput,
        ),
        StructuredTool.from_function(
            name="Google",
            description="Search the web",
            func=Google,
            args_schema=GoogleInput,
        ),
        EscalationTool(
            name="EscalationTool",
            description="Create an Action Center task",
            assign_to="alina.capota@uipath.com",
        ),
    ]


def get_datapoints() -> List[Datapoint]:
    return []


def get_loan_agent() -> AgentBaseClass:
    return AgentBaseClass(
        input_schema=AgentInputSchema,
        output_schema=AgentOutputSchema,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        tools=get_tools(),
        datapoints=get_datapoints(),
    )
