import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 
import pickle
import json
from pathlib import Path
from src.data_loader import load_data, data_preprocessing, feature_engineering
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,precision_recall_curve
from huggingface_hub import HfApi, login

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

RF_CONFIG = json.loads(Path("src/RandomForest/RF_CONFIG.json").read_text())

token = Path("hf_token.txt").read_text().strip()

raw_data = load_data()
processed_data = data_preprocessing(raw_data)
data = feature_engineering(processed_data)

def split_data(data: pd.DataFrame)->tuple[pd.DataFrame,pd.Series]:
  try:
    logger.info("Splitting target column from features...")
    X = data.drop(['SeriousDlqin2yrs'], axis=1)
    y = data['SeriousDlqin2yrs']
    logger.info("Data split successful")
    return X, y
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise 
 
def training(X: pd.DataFrame, y: pd.Series):
  try:
    logger.info("Loading stratifiedKFold with 5 splits...")
    skf = StratifiedKFold(n_splits=RF_CONFIG["validation"]["n_splits"],shuffle=True,random_state=RF_CONFIG["model_params"]["random_state"])
       
    model_scores = []
    baseline_scores = []
    fold=1
    best_model = None
    best_score = -np.inf
     
    logger.info("Training each split...")
    for train_idx, val_idx in skf.split(X, y):
      X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
      y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
       
      logger.info(f"Implementing SMOTE for training data...")
      smote=SMOTE(random_state=RF_CONFIG["model_params"]["random_state"])
      X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
      majority_class = y_val.value_counts().idxmax()
      baseline_pred = np.full(len(y_val),majority_class)
      baseline_scores.append((baseline_pred == y_val).mean())
      fold+=1
       
      model = RandomForestClassifier(class_weight='balanced',
      n_estimators=RF_CONFIG["model_params"]["n_estimators"],
      max_depth=RF_CONFIG["model_params"]["max_depth"],
      max_features=RF_CONFIG["model_params"]["max_features"],
      random_state=RF_CONFIG["model_params"]["random_state"])
        
      logger.info("Training model...")
      model.fit(X_train_res, y_train_res) 
      logger.info("Trained")

      y_proba=model.predict_proba(X_val)[:,1]
      precisions, recalls, thresholds=precision_recall_curve(y_val,y_proba)
        
      logger.info("Checking for optimal threshold value...")
      for p, r, t in zip(precisions, recalls, thresholds):
        if t >= 0.78:
          optimal_threshold = t 
          print(f"Chosen threshold:{optimal_threshold:.4f}") 
          print(f"At this threshold: Precision={p:.4f}, Recall={r:.4f}")
          break
          
      model_score=(y_pred==y_val).mean()
      model_scores.append(model_score)
      if model_score > best_score:
        best_score = model_score
        best_model = model
 
      logger.info(f"\nMean model CV score: {np.mean(model_scores):.4f}")
      logger.info(f"Mean baseline score: {np.mean(baseline_scores):.4f}")
      logger.info("Predicting on test data...")
      y_pred = (y_proba >= optimal_threshold).astype(int)
      
    logger.info("Saving best model...")
    with open("RFmodel.pkl", "wb") as f:
      pickle.dump(best_model, f)
    logger.info("Model saved")
    return best_model, y_pred, y_val
      
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

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

def business_impact(cm):
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
   
def push_model_to_huggingface(model, repo_name: str, token: str = None):
  try:
    logger.info("Logging into Hugging Face Hub...")
    if token:
      login(token=token)
    else:
      login()
    model_filename="credit_default_rf_model.pkl"
    with open(model_filename, 'wb') as f:
      pickle.dump(model, f)
    logger.info(f"Model saved locally as {model_filename}")
    api = HfApi()
    api.upload_file(path_or_fileobj=model_filename,path_in_repo=model_filename,repo_id=repo_name,repo_type="model",commit_message="Upload trained RandomForest credit-default prediction model")
    logger.info(f"Model successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_name}")
  except Exception as e:
    logger.error(f"Failed to push model to HuggingFace due to the following: {e}")
    raise
      
def main():
  try:
    logger.info("Starting ML pipeline...")
    
    X, y = split_data(data)
    
    logger.info("\nTraining features:")
    for i, col in enumerate(X.columns):
      logger.info(f"{i}: {col}")
    
    model,y_pred,y_val=training(X, y)
    
    precision,recall,fscore,support,cm=metrics(y_val,y_pred)
    
    business_impact(cm)
    
    push_model_to_huggingface(model,"credit_default_prediction_model")
    
    logger.info("Pipeline completed.")
    
    return model
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise
  
if __name__ == "__main__":
  main()