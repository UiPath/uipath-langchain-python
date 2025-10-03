from typing import Annotated

from pydantic import BaseModel, Field

# Add schema class for the add function


class LoanAgent_ApplyUiBankLoanInput(BaseModel):
    RequestorEmailAddress: str = Field(description="The email of the requester")
    LoanAmount: int = Field(description="The loan amount requested (integer)")
    LoanTerm: int = Field(description="The loan term (integer)")
    Income: int = Field(description="The requester's income (integer)")
    Age: int = Field(description="The requester's age (integer)")


class LoanAgent_ModifyLoanContractInput(BaseModel):
    SharepointFolderURL: str = Field(description="SharePoint URL")
    InterestRate: int = Field(description="The interest rate of the loan")
    LoanAmount: int = Field(description="The loan amount")
    BorrowerName: str = Field(description="The borrower name")
    BorrowerAddress: str | None = Field(description="The borrower address")


class LoanAgent_SendSharepointDocumentViaDocuSignInput(BaseModel):
    SharepointFileURL: str = Field(
        description="The SharePoint URL of the file created by the LoanAgent_ModifyLoanContract tool"
    )
    RecipientLegalName: str = Field(description="The recipient legal name")
    RecipientEmail: str = Field(description="The recipient email")


class GoogleInput(BaseModel):
    SearchText: str = Field(description="Search text")


def LoanAgent_ApplyUiBankLoan(
    RequestorEmailAddress: Annotated[str, "The email of the requester"],
    LoanAmount: Annotated[int, "The loan amount requested"],
    LoanTerm: Annotated[int, "The loan term"],
    Income: Annotated[int, "The requester's income"],
    Age: Annotated[int, "The requester's age"],
) -> str:
    """Apply for a loan at UI Bank.

    Args:
        RequestorEmailAddress: The email of the requester
        LoanAmount: The loan amount requested (integer)
        LoanTerm: The loan term (integer)
        Income: The requester's income (integer)
        Age: The requester's age (integer)

    Returns:
        string: Loan application details
    """
    if LoanTerm == 5 or LoanTerm == 3 or LoanTerm == 1:
        if Age > 18:
            return "Loan application was approved with a rate of 4% APR."
        else:
            return "The age of the applicant is below the minimum age for a loan."
    else:
        return "Loan application term is not supported."


def LoanAgent_ModifyLoanContract(
    SharepointFolderURL: Annotated[str, "SharePoint URL"],
    InterestRate: Annotated[int, "The interest rate of the loan"],
    LoanAmount: Annotated[int, "The loan amount"],
    BorrowerName: Annotated[str, "The borrower name"],
    BorrowerAddress: Annotated[str, "The borrower address"],
) -> str:
    """Modify a loan contract.

    Args:
        SharepointFolderURL: The SharePoint URL of the file
        InterestRate: The interest rate of the loan
        LoanAmount: The loan amount
        BorrowerName: The borrower name
        BorrowerAddress: The borrower address

    Returns:
        str: Sharepoint file URL
    """
    # Implementation goes here
    return SharepointFolderURL + "/New_doc.pdf"


def LoanAgent_SendSharepointDocumentViaDocuSign(
    SharepointFileURL: Annotated[
        str,
        "The SharePoint URL of the file",
    ],
    RecipientLegalName: Annotated[str, "The recipient legal name"],
    RecipientEmail: Annotated[str, "The recipient email"],
) -> str:
    """Send a sharepoint document via DocuSign.

    Args:
        SharepointFileURL: The SharePoint URL of the file
        RecipientLegalName: The recipient legal name
        RecipientEmail: The recipient email

    Returns:
        str: True if document is sent successfully, False otherwise
    """
    # Implementation goes here
    return "Document New_doc.pdf was sent to: alina.capota@uipath.com"


def Google(SearchText: Annotated[str, "Search text"]) -> str:
    """Search the web for a given text.

    Args:
        SearchText: The text to search for

    Returns:
        str: Search results
    """
    return "The National Bank kept its benchmark interest rate unchanged at 6.5% during its meeting, following two consecutive rate cuts."
