from pathlib import Path

ROOT_DIR=Path(__file__).parent.parent

APP_DIR=ROOT_DIR/"app"
DATA_DIR=ROOT_DIR/"data"
MODEL_DIR=ROOT_DIR/"model"
REQUIREMENTS_PATH=ROOT_DIR/"requirements.txt"
HF_TOKEN_PATH=ROOT_DIR/"hf_token.txt"
SRC_DIR=ROOT_DIR/"src"

#Data files
TRAIN_DATA_PATH=DATA_DIR/"cs-training.csv"
TEST_DATA_PATH=DATA_DIR/"cs-test.csv"
PROCESSED_DATA=DATA_DIR/"data.joblib"

#App file
APP_PATH=APP_DIR/"app.py"

#Model files
RF_MODEL_PATH=MODEL_DIR/"random_forest_model.pkl"
XGB_MODEL_PATH=MODEL_DIR/"xgboost_model.pkl"

#Training code files
RF_TRAIN_PATH=SRC_DIR/"RandomForest"/"random_forest_train.py"
XGB_TRAIN_PATH=SRC_DIR/"XGBoost"/"xgboost_train.py"

#Test code files
RF_TEST_PATH=SRC_DIR/"RandomForest"/"predict_rf.py"
XGB_TEST_PATH=SRC_DIR/"XGBoost"/"predict_xgbc.py"

# Config files
RF_CONFIG_PATH=SRC_DIR/"RandomForest"/"RF_CONFIG.json"
XGB_CONFIG_PATH=SRC_DIR/"XGBoost"/"XGB_CONFIG.json"

#EDA file
EDA_PATH=SRC_DIR/"data_loader.py"