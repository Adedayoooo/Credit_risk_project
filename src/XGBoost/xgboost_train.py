import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from huggingface_hub import HfApi, login
import pickle
import json
from pathlib import Path
from src.data_loader import load_data, data_preprocessing, feature_engineering

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

RF_CONFIG = json.loads(Path("src/XGBoost/XG_CONFIG.json").read_text())

token = Path("hf_token.txt").read_text().strip()
    
raw_data = load_data()
processed_data = data_preprocessing(raw_data)
data = feature_engineering(processed_data)

def split_data(data: pd.DataFrame)->tuple[pd.DataFrame,pd.Series]:
  try:
    logger.info("Splitting target column from features...")
    X= data.drop(['SeriousDlqin2yrs'], axis=1)
    y = data['SeriousDlqin2yrs']
    return X, y
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise 

def training(X: pd.DataFrame, y: pd.Series):
  try:
    logger.info("Training model...")
    skf = StratifiedKFold(n_splits=XG_CONFIG["validation"]["n_splits"],shuffle=True,random_state=XG_CONFIG["model_params"]["random_state"])
    
    model_scores = []
    baseline_scores = []
    fold=1
    
    for train_idx, val_idx in skf.split(X,y):
      X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
      y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
 
      majority_class = y_val.value_counts().idxmax()
      baseline_pred = np.full(len(y_val), majority_class)
      baseline_scores.append((baseline_pred == y_val).mean())
      fold+=1
 
      neg, pos = y_train.value_counts()
      model = XGBClassifier(scale_pos_weight=neg / pos,n_estimators=XG_CONFIG["model_params"]["n_estimators"],max_depth=XG_CONFIG["model_params"]["max_depth"],learning_rate=XG_CONFIG["model_params"]["learning_rate"],eval_metric='logloss',random_state=XG_CONFIG["model_params"]["random_state"])
      model.fit(X_train, y_train) 
      logger.info("Training successful")
      
      logger.info("Saving trained model...")
      with open("XGmodel.pkl", "wb") as f:
        pickle.dump(model, f)
      logger.info("Model saved")
        
      y_proba=model.predict_proba(X_val)[:,1]
      precisions, recalls, thresholds=precision_recall_curve(y_val,y_proba)
      
      logger.info("Checking for optimal threshold value...")
      for p, r, t in zip(precisions, recalls, thresholds):
        if t>0.83:
          optimal_threshold = t 
          print(f"Chosen threshold: {optimal_threshold:.4f}") 
          print(f"At this threshold: precision={p:.4f}, recall={r:.4f}")
          break
        
      model_score = model.score(X_val, y_val)
      model_scores.append(model_score)
      
    logger.info(f"\nMean model CV score: {np.mean(model_scores):.4f}")
    logger.info(f"Mean baseline score: {np.mean(baseline_scores):.4f}")
    logger.info("Predicting on test data...")
    y_pred = (y_proba >= optimal_threshold).astype(int)
    return model, y_pred, y_val
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

def push_model_to_huggingface(model, repo_name: str, token: str = None):
  try:
    logger.info("Logging into Hugging Face Hub...")
    if token:
      login(token=token)
    else:
      login()
    model_filename="credit_default_xgb_model.pkl"
    with open(model_filename, 'wb') as f:
      pickle.dump(model, f)
    logger.info(f"Model saved locally as {model_filename}")
    api = HfApi()
    api.upload_file(path_or_fileobj=model_filename,path_in_repo=model_filename,repo_id=repo_name,repo_type="model",commit_message="Upload trained XGBoost credit-default prediction model")
    logger.info(f"Model successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_name}")
  except Exception as e:
    logger.error(f"Failed to push model to HuggingFace due to the following: {e}")
    raise
      
def main():
  try:
    logger.info("Starting ML pipeline...")
    
    X, y = split_data(data)
    print(f"\nTraining features (in order):")
    for i, col in enumerate(X.columns):
      print(f"{i}: {col}")
    
    model, y_pred, y_val = training(X, y)
    
    #new_data = np.array([[0.99382,34,0,0.693608,4317,7,0,2,0,2,0,0,1,1439]])
    
    push_model_to_huggingface(model,"credit_default_prediction_model")
 
    logger.info("Pipeline completed.")
 
    return model,y_val,y_pred
 
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise
  
if __name__ == "__main__":
 main()