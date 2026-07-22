import enum

from pydantic import BaseModel, Field, ConfigDict


class Company(BaseModel):
    id: str = Field(alias="Id")
    name: str = Field(alias="Name")
    billing_address: str = Field(alias="BillingAddress")
    address: str = Field(alias="Address")
    email_domain: str = Field(alias="EmailDomain")

    model_config = ConfigDict(
        populate_by_name=True,
        validate_by_alias=True,
        extra="allow"
    )

class OrderStatus(int, enum.Enum):
    DELIVERED = 0
    IN_PROGRESS = 1
    PENDING = 2
    READY = 3

class Order(BaseModel):
    order_date: str = Field(alias="OrderDate")
    company_id: str = Field(alias="CompanyName")
    discount: float = Field(alias="Discount")
    final_price: float = Field(alias="FinalPrice")
    order_status: OrderStatus = Field(alias="Status")

    model_config = ConfigDict(
        populate_by_name=True,
        validate_by_alias=True,
        extra="allow"
    )
