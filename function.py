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
        
        # NOVOS: Features temporais importantes
        data['day_of_week'] = data['date'].dt.dayofweek
        data['week_of_year'] = data['date'].dt.isocalendar().week
        data['quarter'] = data['date'].dt.quarter
        data['is_weekend'] = (data['day_of_week'] >= 5).astype('int8')
        data['is_month_start'] = data['date'].dt.is_month_start.astype('int8')
        data['is_month_end'] = data['date'].dt.is_month_end.astype('int8')
        data['days_to_month_end'] = (data['date'].dt.to_period('M').dt.to_timestamp('M') - data['date']).dt.days

        return data
    
    def rename_columns(self, data, old_name, new_name):
        data.rename(columns={old_name: new_name}, inplace=True)

    def modelling_treatment(self, data):
        return data.loc[data['date'] < self.test['date'].min()].copy(), data.loc[data['date'] >= self.test['date'].min()].copy()

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
        data['flag_national_locale'] = data['flag_national_locale'].fillna(0).astype('int8')
        return data
    
    @staticmethod
    def compute_monthly_sales_increase(data):
        monthly_sales = data.groupby('date_month_year')['sales'].sum().reset_index()
        monthly_sales['sales_increase'] = monthly_sales['sales'].pct_change().fillna(0)
        data = pd.merge(data, monthly_sales[['date_month_year', 'sales_increase']], on='date_month_year', how='left')
        return data
    
    @staticmethod
    def on_promotion_flag(data):
        data['flag_onpromotion'] = (data['onpromotion'] > 0).astype('int8')
        return data
    
    @staticmethod
    def add_holiday_flag(data):
        data.loc[~data['holiday_type'].isnull(), 'flag_holiday'] = 1
        data['flag_holiday'] = data['flag_holiday'].fillna(0).astype('int8')
        return data
    
    @staticmethod
    def one_hot_encode_categorical(data, categorical_columns):
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True, dtype='int8')
        return data
    
    # NOVO: Agregar médias históricas (apenas com dados de treino!)
    @staticmethod
    def add_historical_means(data, train_data):
        """
        Adiciona médias históricas calculadas APENAS com dados de treino
        para evitar data leakage
        """
        # Média por loja e família
        store_family_mean = train_data.groupby(['store_nbr', 'family'])['sales'].mean()
        data['store_family_mean'] = data.set_index(['store_nbr', 'family']).index.map(store_family_mean).values
        
        # Média por família e dia da semana
        family_dow_mean = train_data.groupby(['family', 'day_of_week'])['sales'].mean()
        data['family_dow_mean'] = data.set_index(['family', 'day_of_week']).index.map(family_dow_mean).values
        
        # Média por loja
        store_mean = train_data.groupby('store_nbr')['sales'].mean()
        data['store_mean'] = data['store_nbr'].map(store_mean).values
        
        return data

    @staticmethod
    def lag_features(data, target_col, lags):
        """
        CRÍTICO: Esta função deve ser chamada APENAS com dados de treino
        e aplicada depois no teste de forma segura
        """
        group_cols = ['store_nbr', 'family']
        data = data.sort_values(['store_nbr', 'family', 'date'])
        
        for lag in lags:
            data[f'{target_col}_lag_{lag}'] = (
                data.groupby(group_cols)[target_col].shift(lag)
            )
        return data

    @staticmethod
    def rolling_mean_features(data, target_col, windows):
        """
        CRÍTICO: Esta função deve ser chamada APENAS com dados de treino
        """
        group_cols = ['store_nbr', 'family']
        data = data.sort_values(['store_nbr', 'family', 'date'])
        
        for window in windows:
            roll_col_name = f"{target_col}_roll_mean_{window}"
            # Usar min_periods para evitar warnings de "mean of empty slice"
            data[roll_col_name] = (
                data.groupby(group_cols)[target_col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
        return data
    
    @staticmethod
    def add_diff_features(data, target_col):
        """NOVO: Features de diferença (tendência)"""
        group_cols = ['store_nbr', 'family']
        data = data.sort_values(['store_nbr', 'family', 'date'])
        
        data[f'{target_col}_diff_1'] = data.groupby(group_cols)[target_col].diff(1)
        data[f'{target_col}_diff_7'] = data.groupby(group_cols)[target_col].diff(7)
        
        return data
    
    @staticmethod
    def target_encoding(train_data, test_data, cat_col, target_col='sales', smoothing=10):
        """
        NOVO: Target encoding - encode categorias com média do target
        IMPORTANTE: Calcular apenas com dados de treino!
        """
        # Calcular média global
        global_mean = train_data[target_col].mean()
        
        # Calcular média por categoria
        agg = train_data.groupby(cat_col)[target_col].agg(['mean', 'count'])
        
        # Aplicar smoothing
        counts = agg['count']
        means = agg['mean']
        smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
        
        # Criar mapeamento
        encoding_map = smooth.to_dict()
        
        # Aplicar encoding
        col_name = f'{cat_col}_target_enc'
        train_data[col_name] = train_data[cat_col].map(encoding_map).fillna(global_mean)
        test_data[col_name] = test_data[cat_col].map(encoding_map).fillna(global_mean)
        
        return train_data, test_data
    
    @staticmethod
    def add_promotion_features(data):
        """NOVO: Features mais detalhadas de promoção"""
        group_cols = ['store_nbr', 'family']
        data = data.sort_values(['store_nbr', 'family', 'date'])
        
        # Contagem de promoções consecutivas
        data['promo_consecutive'] = data.groupby(group_cols)['flag_onpromotion'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
        )
        
        # Número de itens em promoção (se > 0)
        data['num_items_promo'] = data['onpromotion'].clip(lower=0)
        
        # Lag de promoção (estava em promoção ontem?)
        data['promo_lag_1'] = data.groupby(group_cols)['flag_onpromotion'].shift(1)
        data['promo_lag_7'] = data.groupby(group_cols)['flag_onpromotion'].shift(7)
        
        return data
    
    @staticmethod
    def add_interaction_features(data):
        """NOVO: Features de interação"""
        # Interação: promoção x fim de semana
        if 'flag_onpromotion' in data.columns and 'is_weekend' in data.columns:
            data['promo_x_weekend'] = data['flag_onpromotion'] * data['is_weekend']
        
        # Interação: feriado x fim de semana
        if 'flag_holiday' in data.columns and 'is_weekend' in data.columns:
            data['holiday_x_weekend'] = data['flag_holiday'] * data['is_weekend']
        
        # Interação: promoção x feriado
        if 'flag_onpromotion' in data.columns and 'flag_holiday' in data.columns:
            data['promo_x_holiday'] = data['flag_onpromotion'] * data['flag_holiday']
        
        return data

class DataCleanerAndPreparer:

    @staticmethod
    def remove_outliers_iqr(data, column_name):
        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
        cleaned_data = data.loc[~data['id'].isin(outliers['id'])]
        print(f"Outliers removidos: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
        return cleaned_data
    
    @staticmethod
    def fill_missing_values(data):
        """NOVO: Preencher valores faltantes de forma inteligente"""
        
        # Oil: interpolação linear (se existir)
        if 'dcoilwtico' in data.columns:
            data['dcoilwtico'] = data['dcoilwtico'].interpolate(method='linear')
            data['dcoilwtico'] = data['dcoilwtico'].fillna(method='bfill').fillna(method='ffill')
        
        # Transactions: preencher com mediana da loja (se existir)
        if 'transactions' in data.columns and 'store_nbr' in data.columns:
            data['transactions'] = data.groupby('store_nbr')['transactions'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Onpromotion: preencher com 0 (se existir)
        if 'onpromotion' in data.columns:
            data['onpromotion'] = data['onpromotion'].fillna(0)
        
        # Lag features: forward fill limitado (sem groupby se family não existe)
        lag_cols = [col for col in data.columns if 'lag' in col or 'roll' in col or 'diff' in col]
        
        if 'store_nbr' in data.columns and 'family' in data.columns:
            # Se ainda temos as colunas originais, fazer groupby
            for col in lag_cols:
                data[col] = data.groupby(['store_nbr', 'family'])[col].fillna(method='ffill', limit=3)
        else:
            # Senão, apenas forward fill simples
            for col in lag_cols:
                data[col] = data[col].fillna(method='ffill', limit=3)
        
        # Médias históricas: preencher com média global se necessário
        mean_cols = [col for col in data.columns if 'mean' in col]
        for col in mean_cols:
            data[col] = data[col].fillna(data[col].median())
        
        # Preencher qualquer NaN restante com 0
        data = data.fillna(0)
        
        return data
    
class PreprearerToSubmit:

    @staticmethod
    def prepare_submission(predictions, test_data, filename='submission.csv'):
        # NOVO: Garantir que não há valores negativos
        predictions = np.maximum(predictions, 0)
        
        submission = pd.DataFrame({
            'id': test_data['id'],
            'sales': predictions
        })
        submission.to_csv(filename, index=False)
        print(f"Submission salvo em {filename}")
        print(f"Total de predições: {len(submission)}")
        print(f"Média de vendas preditas: {predictions.mean():.2f}")
        print(f"Mediana de vendas preditas: {np.median(predictions):.2f}")