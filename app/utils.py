import numpy as np

def engineer_features(borrower_data:dict)->np.ndarray:
  total_late_days=(borrower_data['times_30_59_late']+borrower_data['times_60_89_late']+borrower_data['times_90_late'])
  has_late_days = 1 if total_late_days > 0 else 0
  high_utilization = 1 if borrower_data['revolving_utilization'] > 0.7 else 0
  income_per_dependents = (borrower_data['monthly_income'] /(borrower_data['dependents'] + 1))
    
    features = np.array([[
        borrower_data['revolving_utilization'],
        borrower_data['age'],
        borrower_data['times_30_59_late'],
        borrower_data['debt_ratio'],
        borrower_data['monthly_income'],
        borrower_data['open_credit_lines'],
        borrower_data['times_90_late'],
        borrower_data['real_estate_loans'],
        borrower_data['times_60_89_late'],
        borrower_data['dependents'],
        total_late_days,
        has_late_days,
        high_utilization,
        income_per_dependents
    ]])
    return features

def calculate_risk_level(probability: float) -> str:
  if probability < 0.3:
    return "low"
    elif probability < 0.7:
      return "medium"
    else:
      return "high"
