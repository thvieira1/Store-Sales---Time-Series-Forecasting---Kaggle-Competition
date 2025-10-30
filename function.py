import pandas as pd 
import numpy as np

class DataProcessor:

    def __init__(self, train_path, test_path, stores_path, oil_path, transactions_path, holidays_path):
        self.train = pd.read_csv(train_path, parse_dates=['date'])
        self.test = pd.read_csv(test_path, parse_dates=['date'])
        self.stores = pd.read_csv(stores_path)
        self.rename_columns(self.stores, 'type', 'store_type')
        self.oil = pd.read_csv(oil_path, parse_dates=['date'])
        self.transactions = pd.read_csv(transactions_path, parse_dates=['date'])
        self.holidays = pd.read_csv(holidays_path, parse_dates=['date'])
        self.rename_columns(self.holidays, 'type', 'holiday_type')

    def process_data(self):
        data = pd.concat([self.train, self.test], axis=0)
        data = data.sort_values(by=['date'])
        data = pd.merge(data, self.stores, on='store_nbr', how='left')
        data = pd.merge(data, self.oil, on='date', how='left')
        data = pd.merge(data, self.transactions, on=['date', 'store_nbr'], how='left')
        data = pd.merge(data, self.holidays, on=['date'], how='left')
        return data

    def drop_duplicates_by_id(self, data):
        return data.drop_duplicates(['id'])

    def date_process(self, data):
        data['date'] = pd.to_datetime(data['date'])
        data['date_month_year'] = data['date'].dt.strftime('%Y-%m')
        data['date_month_year'] = pd.to_datetime(data['date_month_year'])

        data['date_day'] = data['date'].dt.day
        data['date_month'] = data['date'].dt.month
        data['date_year'] = data['date'].dt.year

        return data
    
    def rename_columns(self, data, old_name, new_name):
        data.rename(columns={old_name: new_name}, inplace=True)

    def modelling_treatment(self, data):
        return data.loc[data['date'] < self.test['date'].min()], data.loc[data['date'] >= self.test['date'].min()]

class FeatureEngineer:

    @staticmethod
    def add_last_week_flag(data):
        month_end = data['date'].dt.to_period('M').dt.to_timestamp('M')
        data['flag_last_week'] = (month_end - data['date']).dt.days < 7
        data['flag_last_week'] = data['flag_last_week'].astype('int8')
        return data

    
    @staticmethod
    def add_national_locale_flag(data):
        data.loc[data['locale'] == "National", 'flag_national_locale'] = 1
        data['flag_national_locale'] = data['flag_national_locale'].fillna(0)
        return data
    
    def compute_monthly_sales_increase(data):
        monthly_sales = data.groupby('date_month_year')['sales'].sum().reset_index()
        monthly_sales['sales_increase'] = monthly_sales['sales'].pct_change().fillna(0)
        data = pd.merge(data, monthly_sales[['date_month_year', 'sales_increase']], on='date_month_year', how='left')
        return data
    
    def on_promotion_flag(data):
        data['flag_onpromotion'] = data['onpromotion'].apply(lambda x: 1 if x > 0 else 0)
        return data
    
    @staticmethod
    def add_holiday_flag(data):
        data.loc[~data['holiday_type'].isnull(), 'flag_holiday'] = 1
        data['flag_holiday'] = data['flag_holiday'].fillna(0)
        return data
    
    def one_hot_encode_categorical(data, categorical_columns):
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True, dtype='int64')
        return data
    

    def lag_features(data, target_col, lags):
        for lag in lags:
            data[f'{target_col}_lag_{lag}'] = data.groupby('store_nbr')[target_col].shift(lag)
        return data

    def rolling_mean_features(data, target_col, windows):
        for window in windows:
            roll_col_name = f"{target_col}_roll_mean_{window}"
            data[roll_col_name] = data.groupby('store_nbr')[target_col].transform(lambda x: x.shift(1).rolling(window).mean())
        return data

class DataCleanerAndPreparer:

    def remove_outliers_iqr(data, column_name):
        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
        cleaned_data = data.loc[~data['id'].isin(outliers['id'])]
        return cleaned_data
    
class PreprearerToSubmit:

    @staticmethod
    def prepare_submission(predictions, test_data, filename='submission.csv'):
        submission = pd.DataFrame({
            'id': test_data['id'],
            'sales': predictions
        })
        submission.to_csv(filename, index=False)