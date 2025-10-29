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


    def date_process(self, data):
        data['date'] = pd.to_datetime(data['date'])
        data['date_month_year'] = data['date'].dt.strftime('%Y-%m')
        data['date_month_year'] = pd.to_datetime(data['date_month_year'])

        data['date_month'] = data['date'].dt.month
        data['date_year'] = data['date'].dt.year

        return data
    
    def rename_columns(self, data, old_name, new_name):
        data.rename(columns={old_name: new_name}, inplace=True)

class FeatureEngineer:

    @staticmethod
    def add_last_week_flag(data):
        data.loc[data['date'].dt.day >= (31-7), 'flag_last_week'] = 1
        data['flag_last_week'] = data['flag_last_week'].fillna(0)
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
        data['flag_holiday'] = data['holiday_type'].apply(lambda x: 1 if x in ['Holiday', 'Bridge', 'Event'] else 0)
        data['flag_holiday'] = data['flag_holiday'].fillna(0)
        return data
    
    def one_hot_encode_categorical(data, categorical_columns):
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
        return data
