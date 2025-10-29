import pandas as pd 
from function import DataProcessor, FeatureEngineer
import numpy as np 

data_processor = DataProcessor('train.csv', 'test.csv', 'stores.csv', 'oil.csv', 'transactions.csv', 'holidays_events.csv')
df = data_processor.date_process()

# df = FeatureEngineer.add_last_week_flag(df)
# df = FeatureEngineer.add_national_locale_flag(df)
# df = FeatureEngineer.compute_monthly_sales_increase(df)
# df = FeatureEngineer.on_promotion_flag(df)
# df = FeatureEngineer.add_holiday_flag(df)
# df = FeatureEngineer.one_hot_encode_categorical(df, ['store_nbr', 'item_nbr'])

print(df.head(), df.tail(), df['sales'])
