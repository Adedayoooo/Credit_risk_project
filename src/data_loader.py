import pandas as pd 
import pickle
import logging 
import os

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def load_data(data_path):
  try:
    logger.info("Loading data...")
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "cs-training.csv")
    raw_data=pd.read_csv(DATA_PATH)
    
    #For training in kaggle
    #raw_data = pd.read_csv("/kaggle/input/datasets/adedayoadebayo23/give-me-some-credit/cs-training.csv")
    
    logger.info(f"""Shape: Dataset has {raw_data.shape[0]} rows and {raw_data.shape[1]} columns""")
    logger.info(f"Columns: {list(raw_data.columns)}")
    return raw_data
  except Exception as e:
    logger.error(f"Error occurred while loading data:{e}")
    raise
  
def data_preprocessing(raw_data:pd.DataFrame):
  try:
    logger.info("Preprocessing data...if :")
    missing_counts = raw_data.isnull().sum()
    
    if (missing_counts.sum()>0):
      logger.info(f"\nMissing values per features:\n{missing_counts}")
      logger.info(f"\nTotal missing values: {missing_counts.sum()}")
      missing_features=missing_counts[missing_counts>0]
      logger.info(f"\nColumns with missing values are:\n{missing_features}")
      
      raw_data['MonthlyIncome'] = raw_data['MonthlyIncome'].fillna(raw_data['MonthlyIncome'].mean())
      raw_data['NumberOfDependents'] = 
      raw_data['NumberOfDependents'].fillna(raw_data['NumberOfDependents'].median())
      raw_data.drop(raw_data.columns[0], axis=1, inplace=True)
      
    data = raw_data.copy()
    return data
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise
  
def feature_engineering(data:pd.DataFrame):
  try:
    logger.info("Creating new features...")
      
    data['TotalLateDays']=(data['NumberOfTime30-59DaysPastDueNotWorse']+data['NumberOfTime60-89DaysPastDueNotWorse']+data['NumberOfTimes90DaysLate'])
      
    data['HasLateDays']=(data['TotalLateDays']>0).astype(int)
      
    data['HighUtilization']=(data['RevolvingUtilizationOfUnsecuredLines']>0.7).astype(int)
      
    data['IncomePerDependents']=data['MonthlyIncome']/(data['NumberOfDependents']+1)
      
    logger.info(f"\nValue counts showing newly created features after preprocessing:\n{data.isnull().sum()}")
    defaulters = (data['SeriousDlqin2yrs'] == 1).sum()
    non_defaulters = (data['SeriousDlqin2yrs'] == 0).sum()
    logger.info(f"\nNumber of Defaulters: {defaulters}")
    logger.info(f"Number of Non-defaulters: {non_defaulters}")
      
    defaulters_percentage = defaulters / len(data) * 100
    logger.info(f"Defaulters percentage: {defaulters_percentage:.4f}%")
    logger.info(f"Non-defaulters percentage: {100 - defaulters_percentage:.4f}%")
    logger.info(f"Finally,the features are:\n{data.columns.to_list()}")
      
    return data 
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

def save_processed_data(data,path="data.pkl"):
  try:
    logger.info("Saving processed data...")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Preprocessed data saved")
    return
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

def main():
  try:
    logger.info("Loading EDA pipeline...")
    raw_data=load_data(data_path)
    data=data_preprocessing(raw_data)
    data=feature_engineering(data)
    save_processed_data(data,path="data.pkl")
    return data
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise 