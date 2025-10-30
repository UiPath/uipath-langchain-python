from typing import List

from langchain_core.tools import BaseTool, StructuredTool
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
    send it to be signed via DocuSign**. The location where to save the modified contract is this one: https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e.
    4. Send the loan contract to be signed via DocuSign using the **LoanAgent_SendSharepointDocumentViaDocuSign** tool.

5. Always escalate to a human, using EscalationTool, if:
    - There is any missing information from the request (e.g name, email address, address, loan amount, etc.)
    - There are multiple requests in the same email
    - The request is not clear
    - Input `reason` as a detailed list of missing parameters.
    - Input `assignee` using the email of the end-user.
    - The email address contains too many special characters before @ or unallowed special characters after @ for it to be considered a genuine address

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
            assign_to="random@uipath.com",
        ),
    ]


def get_datapoints() -> List[Datapoint]:
    return [
        Datapoint(
            name="CompleteJohnLoanApplicationScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $50,000 for 5 years. My email is john.doe@example.com, my name is John Doe, my address is 123 Main St, Anytown, USA, my annual income is $75,000, and I am 35 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": False}},
                "ContainsEvaluator": {"search_text": "loan"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "Google", "LoanAgent_ModifyLoanContract", "LoanAgent_SendSharepointDocumentViaDocuSign"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "Google": ("=", 1), "LoanAgent_ModifyLoanContract": ("=", 1), "LoanAgent_SendSharepointDocumentViaDocuSign": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "john.doe@example.com", "LoanAmount": 50000, "LoanTerm": 5, "Income": 75000, "Age": 35}}, {"name": "LoanAgent_ModifyLoanContract", "args": {"SharepointFolderURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e", "InterestRate": 4, "LoanAmount": 50000, "BorrowerName": "John Doe", "BorrowerAddress": "123 Main St, Anytown, USA"}}, {"name": "LoanAgent_SendSharepointDocumentViaDocuSign", "args": {"SharepointFileURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e/New_doc.pdf", "RecipientLegalName": "John Doe", "RecipientEmail": "john.doe@example.com"}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Loan application process completed"}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should return a loan application process completed",
        ),
        Datapoint(
            name="CompleteJoeLoanApplicationScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $500,000 for 5 years. My email is joe.doe@example.com, my name is Joe Doe, my address is 123 Main St, Anytown, USA, my annual income is $735,000, and I am 55 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": False}},
                "ContainsEvaluator": {"search_text": "loan"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "Google", "LoanAgent_ModifyLoanContract", "LoanAgent_SendSharepointDocumentViaDocuSign"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "Google": ("=", 1), "LoanAgent_ModifyLoanContract": ("=", 1), "LoanAgent_SendSharepointDocumentViaDocuSign": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "joe.doe@example.com", "LoanAmount": 500000, "LoanTerm": 5, "Income": 735000, "Age": 55}}, {"name": "LoanAgent_ModifyLoanContract", "args": {"SharepointFolderURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e", "InterestRate": 4, "LoanAmount": 500000, "BorrowerName": "Joe Doe", "BorrowerAddress": "123 Main St, Anytown, USA"}}, {"name": "LoanAgent_SendSharepointDocumentViaDocuSign", "args": {"SharepointFileURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e/New_doc.pdf", "RecipientLegalName": "Joe Doe", "RecipientEmail": "joe.doe@example.com"}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Loan application process completed"}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should return a loan application process completed",
        ),
        Datapoint(
            name="CompleteBobLoanApplicationScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $100,000 for 5 years. My email is bob.smith@example.com, my name is Bob Smith, my annual income is $150,000, and I am 45 years old. My address is 123 Main St, Anytown, USA."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": False}},
                "ContainsEvaluator": {"search_text": "loan"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "Google", "LoanAgent_ModifyLoanContract", "LoanAgent_SendSharepointDocumentViaDocuSign"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "Google": ("=", 1), "LoanAgent_ModifyLoanContract": ("=", 1), "LoanAgent_SendSharepointDocumentViaDocuSign": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "bob.smith@example.com", "LoanAmount": 100000, "LoanTerm": 5, "Income": 150000, "Age": 45}}, {"name": "LoanAgent_ModifyLoanContract", "args": {"SharepointFolderURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e", "InterestRate": 4, "LoanAmount": 100000, "BorrowerName": "Bob Smith", "BorrowerAddress": "123 Main St, Anytown, USA"}}, {"name": "LoanAgent_SendSharepointDocumentViaDocuSign", "args": {"SharepointFileURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e/New_doc.pdf", "RecipientLegalName": "Bob Smith", "RecipientEmail": "bob.smith@example.com"}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Complete workflow executed successfully"}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should return complete workflow executed successfully",
        ),
        Datapoint(
            name="TestMissingInformationScenario",
            input={
                "EmailBody": "I would like to apply for a loan, my email is jane.doe@example.com"
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": True}},
                "ContainsEvaluator": {"search_text": "escalat"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["EscalationTool"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"EscalationTool": ("=", 1)}},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Missing information escalated"}},
            },
            simulation_instructions="Agent should escalate due to missing information",
        ),
        Datapoint(
            name="TestInvalidAgeScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $5000 for 3 years. My email is anna@email.com, my name is Anna, my address is 123 Main St, Anytown, USA, my income is 500$, and I am 17 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": True}},
                "ContainsEvaluator": {"search_text": "age"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["EscalationTool"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"EscalationTool": ("=", 1)}},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Invalid data detected: Age is below the minimum age for a loan."}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should reject due to invalid age",
        ),
        Datapoint(
            name="TestInvalidTermScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $5000 for 7 years. My email is anna@email.com, my name is Anna, my address is 123 Main St, Anytown, USA, my income is 500$, and I am 19 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": True}},
                "ContainsEvaluator": {"search_text": "term"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "EscalationTool"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "EscalationTool": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "anna@email.com", "LoanAmount": 5000, "LoanTerm": 7, "Income": 500, "Age": 19}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Invalid data detected: Term is above the maximum term for a loan."}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should reject due to invalid term",
        ),
        Datapoint(
            name="OneYearLoanScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $20,000 for 1 year. My email is sarah.quick@example.com, my name is Sarah Quick, my address is 567 Short St, Boston, USA, my annual income is $80,000, and I am 32 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": False}},
                "ContainsEvaluator": {"search_text": "loan"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "Google", "LoanAgent_ModifyLoanContract", "LoanAgent_SendSharepointDocumentViaDocuSign"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "Google": ("=", 1), "LoanAgent_ModifyLoanContract": ("=", 1), "LoanAgent_SendSharepointDocumentViaDocuSign": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "sarah.quick@example.com", "LoanAmount": 20000, "LoanTerm": 1, "Income": 80000, "Age": 32}}, {"name": "LoanAgent_ModifyLoanContract", "args": {"SharepointFolderURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e", "InterestRate": 4, "LoanAmount": 20000, "BorrowerName": "Sarah Quick", "BorrowerAddress": "567 Short St, Boston, USA"}}, {"name": "LoanAgent_SendSharepointDocumentViaDocuSign", "args": {"SharepointFileURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e/New_doc.pdf", "RecipientLegalName": "Sarah Quick", "RecipientEmail": "sarah.quick@example.com"}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Short-term loan application processed successfully"}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should return short-term loan processed",
        ),
        Datapoint(
            name="MissingNameScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $30,000 for 3 years. My email is missing.name@example.com, my address is 789 Nameless Ave, Chicago, USA, my annual income is $70,000, and I am 40 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": True}},
                "ContainsEvaluator": {"search_text": "name"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["EscalationTool"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"EscalationTool": ("=", 1)}},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Missing name information escalated"}},
            },
            simulation_instructions="Agent should escalate due to missing name",
        ),
        Datapoint(
            name="MissingEmailScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $45,000 for 5 years. My name is Emily NoEmail, my address is 123 Email-less St, Denver, USA, my annual income is $65,000, and I am 29 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": True}},
                "ContainsEvaluator": {"search_text": "email"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["EscalationTool"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"EscalationTool": ("=", 1)}},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Missing email information escalated"}},
            },
            simulation_instructions="Agent should escalate due to missing email",
        ),
        Datapoint(
            name="HighIncomeLowLoanScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $10,000 for 3 years. My email is william.wealthy@example.com, my name is William Wealthy, my address is 888 Rich Ave, Manhattan, USA, my annual income is $500,000, and I am 48 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": False}},
                "ContainsEvaluator": {"search_text": "loan"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "Google", "LoanAgent_ModifyLoanContract", "LoanAgent_SendSharepointDocumentViaDocuSign"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "Google": ("=", 1), "LoanAgent_ModifyLoanContract": ("=", 1), "LoanAgent_SendSharepointDocumentViaDocuSign": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "william.wealthy@example.com", "LoanAmount": 10000, "LoanTerm": 3, "Income": 500000, "Age": 48}}, {"name": "LoanAgent_ModifyLoanContract", "args": {"SharepointFolderURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e", "InterestRate": 4, "LoanAmount": 10000, "BorrowerName": "William Wealthy", "BorrowerAddress": "888 Rich Ave, Manhattan, USA"}}, {"name": "LoanAgent_SendSharepointDocumentViaDocuSign", "args": {"SharepointFileURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e/New_doc.pdf", "RecipientLegalName": "William Wealthy", "RecipientEmail": "william.wealthy@example.com"}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "High-income small loan application processed"}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should process high-income small loan",
        ),
        Datapoint(
            name="LowIncomeLargeRatioLoanScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $40,000 for 5 years. My email is peter.stretch@example.com, my name is Peter Stretch, my address is 456 Tight Blvd, Phoenix, USA, my annual income is $45,000, and I am 36 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": False}},
                "ContainsEvaluator": {"search_text": "loan"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "Google", "LoanAgent_ModifyLoanContract", "LoanAgent_SendSharepointDocumentViaDocuSign"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "Google": ("=", 1), "LoanAgent_ModifyLoanContract": ("=", 1), "LoanAgent_SendSharepointDocumentViaDocuSign": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "peter.stretch@example.com", "LoanAmount": 40000, "LoanTerm": 5, "Income": 45000, "Age": 36}}, {"name": "LoanAgent_ModifyLoanContract", "args": {"SharepointFolderURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e", "InterestRate": 4, "LoanAmount": 40000, "BorrowerName": "Peter Stretch", "BorrowerAddress": "456 Tight Blvd, Phoenix, USA"}}, {"name": "LoanAgent_SendSharepointDocumentViaDocuSign", "args": {"SharepointFileURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e/New_doc.pdf", "RecipientLegalName": "Peter Stretch", "RecipientEmail": "peter.stretch@example.com"}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Low-income high-ratio loan application processed"}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should process low-income high-ratio loan",
        ),
        Datapoint(
            name="SeniorApplicantScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $25,000 for 3 years. My email is martha.senior@example.com, my name is Martha Senior, my address is 123 Retirement Ln, Miami, USA, my annual income is $55,000, and I am 72 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": False}},
                "ContainsEvaluator": {"search_text": "loan"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "Google", "LoanAgent_ModifyLoanContract", "LoanAgent_SendSharepointDocumentViaDocuSign"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "Google": ("=", 1), "LoanAgent_ModifyLoanContract": ("=", 1), "LoanAgent_SendSharepointDocumentViaDocuSign": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "martha.senior@example.com", "LoanAmount": 25000, "LoanTerm": 3, "Income": 55000, "Age": 72}}, {"name": "LoanAgent_ModifyLoanContract", "args": {"SharepointFolderURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e", "InterestRate": 4, "LoanAmount": 25000, "BorrowerName": "Martha Senior", "BorrowerAddress": "123 Retirement Ln, Miami, USA"}}, {"name": "LoanAgent_SendSharepointDocumentViaDocuSign", "args": {"SharepointFileURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e/New_doc.pdf", "RecipientLegalName": "Martha Senior", "RecipientEmail": "martha.senior@example.com"}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Senior applicant loan processed"}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should process senior applicant loan",
        ),
        Datapoint(
            name="ExactlyMinimumAgeScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $15,000 for 3 years. My email is justin.young@example.com, my name is Justin Young, my address is 789 Youth St, Austin, USA, my annual income is $40,000, and I am 19 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": False}},
                "ContainsEvaluator": {"search_text": "loan"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "Google", "LoanAgent_ModifyLoanContract", "LoanAgent_SendSharepointDocumentViaDocuSign"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "Google": ("=", 1), "LoanAgent_ModifyLoanContract": ("=", 1), "LoanAgent_SendSharepointDocumentViaDocuSign": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "justin.young@example.com", "LoanAmount": 15000, "LoanTerm": 3, "Income": 40000, "Age": 19}}, {"name": "LoanAgent_ModifyLoanContract", "args": {"SharepointFolderURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e", "InterestRate": 4, "LoanAmount": 15000, "BorrowerName": "Justin Young", "BorrowerAddress": "789 Youth St, Austin, USA"}}, {"name": "LoanAgent_SendSharepointDocumentViaDocuSign", "args": {"SharepointFileURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e/New_doc.pdf", "RecipientLegalName": "Justin Young", "RecipientEmail": "justin.young@example.com"}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Minimum age applicant loan processed"}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should process minimum age applicant loan",
        ),
        Datapoint(
            name="InternationalAddressScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $60,000 for 5 years. My email is olivia.international@example.com, my name is Olivia International, my address is 42 Global Way, London, United Kingdom, my annual income is $90,000, and I am 33 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": False}},
                "ContainsEvaluator": {"search_text": "loan"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["LoanAgent_ApplyUiBankLoan", "Google", "LoanAgent_ModifyLoanContract", "LoanAgent_SendSharepointDocumentViaDocuSign"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"LoanAgent_ApplyUiBankLoan": ("=", 1), "Google": ("=", 1), "LoanAgent_ModifyLoanContract": ("=", 1), "LoanAgent_SendSharepointDocumentViaDocuSign": ("=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "LoanAgent_ApplyUiBankLoan", "args": {"RequestorEmailAddress": "olivia.international@example.com", "LoanAmount": 60000, "LoanTerm": 5, "Income": 90000, "Age": 33}}, {"name": "LoanAgent_ModifyLoanContract", "args": {"SharepointFolderURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e", "InterestRate": 4, "LoanAmount": 60000, "BorrowerName": "Olivia International", "BorrowerAddress": "42 Global Way, London, United Kingdom"}}, {"name": "LoanAgent_SendSharepointDocumentViaDocuSign", "args": {"SharepointFileURL": "https://uipath-my.sharepoint.com/:f:/r/personal/random_uipath_com/Documents/Documents?csf=1&web=1&e=HDt89e/New_doc.pdf", "RecipientLegalName": "Olivia International", "RecipientEmail": "olivia.international@example.com"}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "International address application processed"}},
            },
            simulation_instructions="Tool LoanAgent_ApplyUiBankLoan should process international address application",
        ),
        Datapoint(
            name="MultipleRequestsInOneEmailScenario",
            input={
                "EmailBody": "I would like to apply for two loans: one of $30,000 for 3 years and another of $50,000 for 5 years. My email is robert.multiple@example.com, my name is Robert Multiple, my address is 555 Double St, Seattle, USA, my annual income is $120,000, and I am 41 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": True}},
                "ContainsEvaluator": {"search_text": "Multiple"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["EscalationTool"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"EscalationTool": ("=", 1)}},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "Multiple requests clarification needed"}},
            },
            simulation_instructions="Agent should escalate due to multiple requests",
        ),
        Datapoint(
            name="SpecialCharactersInInputScenario",
            input={
                "EmailBody": "I would like to apply for a loan of $35,000 for 3 years. My email is xx_dudu!@$_xx@example.com, my name is Dudu, my address is 123 Symbol# St, L@$ Veg@$, USA, my annual income is $85,000, and I am 38 years old."
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"ActionCenterTaskCreated": True}},
                "ContainsEvaluator": {"search_text": "special characters"},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["EscalationTool"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"EscalationTool": ("=", 1)}},
                "LLMJudgeOutputEvaluator": {"expected_output": {"ExecutionDetails": "The email address contains too many special characters"}},
            },
            simulation_instructions="Agent should escalate due to special characters in input",
        ),
    ]


def get_loan_agent() -> AgentBaseClass:
    return AgentBaseClass(
        input_schema=AgentInputSchema,
        output_schema=AgentOutputSchema,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        tools=get_tools(),
        datapoints=get_datapoints(),
    )
