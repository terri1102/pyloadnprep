#class Engineer:
 #   def __init__(self, df):
  #      self.df = df
import pandas as pd
import numpy as np
import xgboost as xgb

def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['종가']
    X = data.drop(columns=['종가','날짜'])
    
    train_samples = int(X.shape[0] * 0.65)
 
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]
    
    #return (X_train, y_train), (X_test, y_test)

# Get training and test data
    X_train_FI, y_train_FI, X_test_FI, y_test_FI = X_train, y_train, X_test, y_test

    regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)

    xgbModel = regressor.fit(X_train_FI,y_train_FI, \
                            eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], \
                            verbose=False)

    #eval_result = regressor.evals_result()

    #df로 보기
    result_frame = pd.DataFrame(data=[X_test_FI.columns, xgbModel.feature_importances_.tolist()])

    new_result_frame = result_frame.T.rename(columns = {0:"feature_name",1:'feature_importance'})

    new_result_frame = new_result_frame.sort_values(by='feature_importance',ascending=False)
    return new_result_frame

# '000270','000660','005380','005490','005930','009240','009540','015760','030200','035250'