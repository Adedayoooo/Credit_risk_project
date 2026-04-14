from pydantic import BaseModel, Field

class BorrowerInput(BaseModel):
  revolving_utilization:float = Field(...,ge=0,le=10,description="Revolving utilization of unsecured lines(0-10)")
  age: int = Field(...,ge=18,le=100,description="Age of borrower (18-100)")
  times_30_59_late: int = Field(...,ge=0,le=30,description="Number of times 30-59 days past due")
  debt_ratio: float = Field(...,ge=0,le=2,description="Debt ratio (0-2)")
  monthly_income: float = Field(...,ge=0,le=1000000,description="Monthly income in dollars")
  open_credit_lines: int = Field(...,ge=0,le=40,description="Number of open credit lines and loans")
  times_90_late: int = Field(...,ge=0,le=10,description="Number of times 90+ days late")
  real_estate_loans: int = Field(...,ge=0,le=20,description="Number of real estate loans or lines")
  times_60_89_late: int = Field(...,ge=0,le=20,description="Number of times 60-89 days past due")
  dependents: int = Field(...,ge=0,le=20,description="Number of dependents")
    
  class Config:
    schema_extra = {
      "example": {
        "revolving_utilization": 0.99,
        "age": 34,
        "times_30_59_late": 0,
        "debt_ratio": 0.69,
        "monthly_income": 4317,
        "open_credit_lines": 7,
        "times_90_late": 0,
        "real_estate_loans": 2,
        "times_60_89_late": 0,
        "dependents": 2
      }
    }

class PredictionOutput(BaseModel):
  default_probability:float=Field(...,description="Probability of default (0-1)")
  decision:str=Field(...,description"Loan decision: 'approve' or 'reject'")
  confidence:float=Field(...,description="Confidence in prediction (0-1)")
  risk_level:str=Field(...,description="Risk category: 'low', 'medium', or 'high'")
