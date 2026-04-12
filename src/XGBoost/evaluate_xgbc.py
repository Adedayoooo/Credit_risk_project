from XGBoost.xgboost_train import main 
from model import xgboost_model
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,precision_recall_curve
import numpy as np 

def metrics(y_val, y_pred):
  try:
    logger.info("Checking metrics...")
    precision,recall,fscore,support=precision_recall_fscore_support(y_val,y_pred,average=None)
    
    cm=confusion_matrix(y_val, y_pred)
    logger.info(f"\nPrecision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-score: {fscore}")
    logger.info(f"Support: {support}")
    logger.info(f"Confusion Matrix:\n{cm}")
    return precision, recall, fscore, support, cm
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise 

def business_impact(y_val,y_pred,cm):
  try:
    logger.info("Evaluating business impact...")
    TN,FP,FN,TP=cm[0][0],cm[0][1],cm[1][0],cm[1][1]
 
  # Assumptions 
    avg_loan_amount=10000 # $10,000 average loan
    interest_rate=0.15 # 15% interest per year
    loan_term_years=2 # 2-year loan 
    
    # Costs
    fp_cost_per_loan=avg_loan_amount * interest_rate * loan_term_years
    
    fn_cost_per_loan = avg_loan_amount
 
    total_fp_cost = FP * fp_cost_per_loan
    total_fn_cost = FN * fn_cost_per_loan
    total_cost = total_fp_cost + total_fn_cost
    
    logger.info(f"For an assumed average loan amount of ${avg_loan_amount}, assumed interest rate of {interest_rate}% and assumed loan_term_years of {loan_term_years}years, the total model cost is ${total_cost}")
    
    # Baseline (approve everyone)
    total_defaults=FN + TP
    baseline_cost = total_defaults * fn_cost_per_loan
 
    savings = baseline_cost - total_cost
 
    logger.info(f"True Negatives (correctly approved): {TN:,}")
    logger.info(f"False Positives (wrongly rejected): {FP:,}")
    logger.info(f"False Negatives (wrongly approved): {FN:,}")
    logger.info(f"True Positives (correctly rejected): {TP:,}")
    logger.info(f"\nCost breakdown:")
    logger.info(f"FP Cost (lost profit from rejections): ${total_fp_cost:,.2f}")
    logger.info(f"FN Cost (lost principal from defaults): ${total_fn_cost:,.2f}")
    logger.info(f"Total model cost: ${total_cost:,.2f}")
    logger.info(f"\nBaseline cost (approve all): ${baseline_cost:,.2f}")
 
    if savings > 0:
      logger.info(f"Model saves: ${savings:,.2f}")
      logger.info(f"ROI: {(savings/baseline_cost)*100:.1f}% improvement")
    else:
      logger.info(f"Model costs: ${abs(savings):,.2f} more than baseline")

    return savings
 
  except Exception as e:
     logger.error(f"An error occurred: {e}")
     raise
   
if __name__ == "__main__":
  y_val,y_pred=main()
  precision, recall, fscore, support, cm = metrics(y_val,y_pred)
  savings=business_impact(y_val,y_pred,cm)