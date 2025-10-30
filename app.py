import pandas as pd 
from function import DataProcessor, FeatureEngineer, DataCleanerAndPreparer, PreprearerToSubmit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

import numpy as np 

data_processor = DataProcessor('train.csv', 'test.csv', 'stores.csv', 'oil.csv', 'transactions.csv', 'holidays_events.csv')
df = data_processor.process_data()
df = data_processor.date_process(df)
df = data_processor.drop_duplicates_by_id(df)
df = FeatureEngineer.add_last_week_flag(df)
df = FeatureEngineer.add_national_locale_flag(df)
df = FeatureEngineer.compute_monthly_sales_increase(df)
df = FeatureEngineer.on_promotion_flag(df)
df = FeatureEngineer.add_holiday_flag(df)
df = FeatureEngineer.one_hot_encode_categorical(
    df, 
    ['family', 'city', 'state', 'store_type', 'transferred']
)

df, df_sub = data_processor.modelling_treatment(df)
df_sub['id'] = df_sub.index


df = DataCleanerAndPreparer.remove_outliers_iqr(df, 'sales')

df_num = df.select_dtypes(include=np.number)
df_num = df_num.dropna()





X = df_num.drop(['id', 'sales'], axis = 1)
y = np.log1p(df_num['sales'])


train_size = int(len(df_num) * 0.7)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)

model.fit(
    X_train, y_train
)

y_pred = np.expm1(model.predict(X_test))
y_true = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
print(f'RMSE: {rmse:.2f}  |  MAE: {mae:.2f}')



df_num_sub = df_sub.select_dtypes(include=np.number)

X_sub = df_num_sub.drop(['id', 'sales'], axis = 1)

y_pred_sub = model.predict(X_sub)


def prepare_submission(predictions, test_data, filename='submission.csv'):
    submission = pd.DataFrame({
        'id': test_data['id'],
        'sales': predictions
    })
    submission.to_csv(filename, index=False)


preparer = PreprearerToSubmit()
preparer.prepare_submission(y_pred_sub, df_sub, filename='submission.csv')
