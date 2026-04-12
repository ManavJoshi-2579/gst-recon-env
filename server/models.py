from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Literal
from enum import Enum
import datetime

class ActionType(str, Enum):
    MATCH = "match"
    REJECT = "reject"
    CLAIM_ITC = "claim_itc"
    QUERY_VENDOR = "query_vendor"
    SUBMIT_REPORT = "submit_report"

class Action(BaseModel):
    type: ActionType = Field(..., description="Type of action")
    invoice_id: Optional[str] = Field(None, description="Target invoice ID for match/reject/claim")
    reason: Optional[str] = Field(None, description="Reason for query or reject")

class Invoice(BaseModel):
    id: str = Field(..., description="Unique invoice ID")
    gstin: str = Field(..., description="Supplier GSTIN (15 chars)")
    date: str = Field(..., description="Invoice date YYYY-MM-DD")
    value: float = Field(..., description="Invoice value")
    hsn: str = Field(..., description="HSN code (4 or 6 digits)")
    igst: float = Field(..., description="IGST amount")
    cgst: float = Field(..., description="CGST amount")
    sgst: float = Field(..., description="SGST amount")
    is_einvoice: bool = Field(..., description="Has valid e-invoice QR")
    is_fraud: bool = Field(default=False, description="Ground truth fraud flag (hidden)")

class GSTR2BEntry(BaseModel):
    invoice_id: str = Field(..., description="Matching GSTR-2B invoice ID")
    gstin: str = Field(..., description="Supplier GSTIN")
    date: str = Field(..., description="Date")
    value: float = Field(..., description="Value")
    igst: float = Field(..., description="IGST")
    cgst: float = Field(..., description="CGST")
    sgst: float = Field(..., description="SGST")

class Observation(BaseModel):
    current_invoice: Optional[Invoice] = Field(None, description="Current invoice under review")
    available_gstr2b: List[GSTR2BEntry] = Field(default_factory=list, description="Available GSTR-2B entries")
    matched: List[str] = Field(default_factory=list, description="Successfully matched invoice IDs")
    mismatches: List[str] = Field(default_factory=list, description="Detected mismatches")
    current_itc: float = Field(0.0, description="Current claimed ITC")
    total_itc_possible: float = Field(0.0, description="Maximum possible ITC")
    progress: float = Field(0.0, description="Progress 0-1")
    warnings: List[str] = Field(default_factory=list, description="Compliance warnings")
    step_count: int = Field(0, description="Steps taken")

class State(BaseModel):
    episode_id: str
    step_count: int
    task_name: str