import numpy as np 
from model import random_forest_model
from RandomForest.random_forest_train import main
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

def load_model():
  try:
    # Path(__file__) = path to predict.py itself
    #.parent = src/
    #.parent.parent = project root
    root_directory=Path(__file__).parent.parent.parent
    model_path=root_directory/"model"/"random_forest_model.pkl"

    logger.info(f"Loading model from {model_path}")
    if model_path.exists():
      with model_path.open('rb') as f:
        model=pickle.load(f)
    else:
      raise FileNotFoundError(f"Model not found at {model_path}")
    return model
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

def predict(model:random_forest_model.pkl,new_data:np.ndarray)->tuple[int,float]:
  try:
    logger.info("Predicting...")
    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]
    
    if prediction == 0:
      logger.info("Client is NOT likely to default.")
      logger.info(f"Confidence: {probabilities[0]*100:.2f}%")
    else:
      logger.info("Client is LIKELY to default")
      logger.info(f"Confidence:{probabilities[1]*100:.2f}%")
      
    return prediction, probabilities
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise


#new_data = np.array([[0.99382,34,0,0.693608,4317,7,0,2,0,2,0,0,1,1439]])
    
    reshaped_data=new_data.reshape(1,-1)
prediction, probs = predict(model, reshaped_data)
