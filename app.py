import pandas as pd 
from function import DataProcessor, FeatureEngineer, DataCleanerAndPreparer, PreprearerToSubmit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np 

print("="*50)
print("INICIANDO PROCESSAMENTO - VERSÃO 2.0")
print("="*50)

data_processor = DataProcessor('train.csv', 'test.csv', 'stores.csv', 'oil.csv', 'transactions.csv', 'holidays_events.csv')
df = data_processor.process_data()
df = data_processor.date_process(df)
df = data_processor.drop_duplicates_by_id(df)

df = FeatureEngineer.add_last_week_flag(df)
df = FeatureEngineer.add_national_locale_flag(df)
df = FeatureEngineer.on_promotion_flag(df)
df = FeatureEngineer.add_holiday_flag(df)

print("\n" + "="*50)
print("SEPARANDO TREINO E TESTE")
print("="*50)
df_train, df_test = data_processor.modelling_treatment(df)
print(f"Tamanho treino: {len(df_train)}")
print(f"Tamanho teste: {len(df_test)}")

print("\n" + "="*50)
print("APLICANDO TARGET ENCODING")
print("="*50)
for col in ['family', 'store_nbr']:
    if col in df_train.columns:
        df_train, df_test = FeatureEngineer.target_encoding(df_train, df_test, col, 'sales', smoothing=10)
        print(f"Target encoding aplicado em: {col}")

print("\n" + "="*50)
print("CRIANDO FEATURES DE PROMOÇÃO E INTERAÇÃO")
print("="*50)
df_train = FeatureEngineer.add_promotion_features(df_train)
df_test = FeatureEngineer.add_promotion_features(df_test)
df_train = FeatureEngineer.add_interaction_features(df_train)
df_test = FeatureEngineer.add_interaction_features(df_test)

print("\n" + "="*50)
print("CRIANDO FEATURES DE LAG E ROLLING")
print("="*50)

df_train = FeatureEngineer.lag_features(
    df_train, 
    target_col='sales',
    lags=[1, 2, 3, 7, 14, 21, 28]
)

df_train = FeatureEngineer.rolling_mean_features(
    df_train, 
    target_col='sales',
    windows=[7, 14, 28, 56]
)

df_train = FeatureEngineer.add_diff_features(df_train, 'sales')

df_combined = pd.concat([df_train, df_test], axis=0).sort_values(['store_nbr', 'family', 'date'])

df_combined = FeatureEngineer.lag_features(
    df_combined, 
    target_col='sales',
    lags=[1, 2, 3, 7, 14, 21, 28]
)

df_combined = FeatureEngineer.rolling_mean_features(
    df_combined, 
    target_col='sales',
    windows=[7, 14, 28, 56]
)

df_combined = FeatureEngineer.add_diff_features(df_combined, 'sales')

df_train = df_combined[df_combined['date'] < df_test['date'].min()].copy()
df_test = df_combined[df_combined['date'] >= df_test['date'].min()].copy()

print("\n" + "="*50)
print("ADICIONANDO MÉDIAS HISTÓRICAS")
print("="*50)
df_train = FeatureEngineer.add_historical_means(df_train, df_train)
df_test = FeatureEngineer.add_historical_means(df_test, df_train)

df_train = FeatureEngineer.compute_monthly_sales_increase(df_train)
monthly_sales_train = df_train[['date_month_year', 'sales_increase']].drop_duplicates()
df_test = pd.merge(df_test, monthly_sales_train, on='date_month_year', how='left', suffixes=('', '_new'))
if 'sales_increase_new' in df_test.columns:
    df_test['sales_increase'] = df_test['sales_increase_new'].fillna(df_test['sales_increase'])
    df_test = df_test.drop('sales_increase_new', axis=1)

print("\n" + "="*50)
print("PREENCHENDO VALORES FALTANTES")
print("="*50)
df_train = DataCleanerAndPreparer.fill_missing_values(df_train)
df_test = DataCleanerAndPreparer.fill_missing_values(df_test)

print("\n" + "="*50)
print("ONE-HOT ENCODING")
print("="*50)
df_combined = pd.concat([df_train, df_test], axis=0)
df_combined = FeatureEngineer.one_hot_encode_categorical(
    df_combined, 
    ['family', 'city', 'state', 'store_type', 'transferred']
)

df_train = df_combined[df_combined['date'] < df_test['date'].min()].copy()
df_test = df_combined[df_combined['date'] >= df_test['date'].min()].copy()

print("\n" + "="*50)
print("REMOVENDO OUTLIERS")
print("="*50)
df_train = DataCleanerAndPreparer.remove_outliers_iqr(df_train, 'sales')

print("\n" + "="*50)
print("PREPARANDO DADOS PARA MODELAGEM")
print("="*50)

df_train_num = df_train.select_dtypes(include=np.number)
df_test_num = df_test.select_dtypes(include=np.number)

print(f"Shape treino antes de dropar NaN: {df_train_num.shape}")
df_train_num = df_train_num.dropna()
print(f"Shape treino depois de dropar NaN: {df_train_num.shape}")

df_train_with_dates = df_train_num.merge(df_train[['id', 'date']], on='id', how='left')

X_full = df_train_with_dates.drop(['id', 'sales', 'date'], axis=1)
y_full = np.log1p(df_train_with_dates['sales'])
dates = df_train_with_dates['date']

df_test_num = df_test_num.fillna(0)
missing_cols = set(X_full.columns) - set(df_test_num.columns)
for col in missing_cols:
    df_test_num[col] = 0
X_test = df_test_num[X_full.columns]

print(f"Features no modelo: {X_full.shape[1]}")
print(f"Amostras treino: {len(X_full)}")
print(f"Amostras teste: {len(X_test)}")

print("\n" + "="*50)
print("TREINANDO ENSEMBLE: XGBOOST + LIGHTGBM")
print("="*50)

split_date = df_train['date'].max() - pd.Timedelta(days=15)
train_mask = dates < split_date
val_mask = dates >= split_date

X_t = X_full[train_mask]
y_t = y_full[train_mask]
X_v = X_full[val_mask]
y_v = y_full[val_mask]

print(f"\n>>> Treinando XGBoost...")
model_xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    tree_method='hist',
    n_jobs=-1,
    early_stopping_rounds=50
)

model_xgb.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=100)

print(f"\n>>> Treinando LightGBM...")
model_lgbm = LGBMRegressor(
    objective='regression',
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)

model_lgbm.fit(
    X_t, y_t,
    eval_set=[(X_v, y_v)],
    eval_metric='rmse'
)

print("\n" + "="*50)
print("AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO")
print("="*50)

y_pred_xgb_log = model_xgb.predict(X_v)
y_pred_lgbm_log = model_lgbm.predict(X_v)

weights = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.4, 0.6)]
best_weight = None
best_rmsle = float('inf')

for w_xgb, w_lgbm in weights:
    y_pred_ensemble_log = w_xgb * y_pred_xgb_log + w_lgbm * y_pred_lgbm_log
    y_pred_ensemble = np.expm1(y_pred_ensemble_log)
    y_true_val = np.expm1(y_v)
    
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true_val), np.log1p(y_pred_ensemble)))
    print(f"Peso XGB={w_xgb:.1f}, LGBM={w_lgbm:.1f} -> RMSLE: {rmsle:.4f}")
    
    if rmsle < best_rmsle:
        best_rmsle = rmsle
        best_weight = (w_xgb, w_lgbm)

print(f"\nMelhor combinação: XGB={best_weight[0]:.1f}, LGBM={best_weight[1]:.1f} -> RMSLE: {best_rmsle:.4f}")

print("\n" + "="*50)
print("PREDIÇÃO NO SET DE TESTE")
print("="*50)

y_pred_xgb_test_log = model_xgb.predict(X_test)
y_pred_lgbm_test_log = model_lgbm.predict(X_test)

y_pred_test_log = best_weight[0] * y_pred_xgb_test_log + best_weight[1] * y_pred_lgbm_test_log
y_pred_test = np.expm1(y_pred_test_log)


y_pred_test = np.maximum(y_pred_test, 0)

q99 = df_train['sales'].quantile(0.99)
y_pred_test = np.clip(y_pred_test, 0, q99)

PreprearerToSubmit.prepare_submission(y_pred_test, df_test, filename='submission.csv')

print("\n" + "="*50)
print("PROCESSO FINALIZADO - VERSÃO 2.0!")
print("="*50)
print(f"\nRESUMO:")
print(f"- RMSLE Validação: {best_rmsle:.4f}")
print(f"- Melhor peso: XGB={best_weight[0]:.1f}, LGBM={best_weight[1]:.1f}")
print(f"- XGBoost best iteration: {model_xgb.best_iteration if hasattr(model_xgb, 'best_iteration') else 'N/A'}")
print(f"- LightGBM best iteration: {model_lgbm.best_iteration_ if hasattr(model_lgbm, 'best_iteration_') else 'N/A'}")
print(f"\n⚠️ IMPORTANTE: Espere RMSLE Kaggle próximo de {best_rmsle:.4f}") 

print("="*50)
print("INICIANDO PROCESSAMENTO")
print("="*50)

data_processor = DataProcessor('train.csv', 'test.csv', 'stores.csv', 'oil.csv', 'transactions.csv', 'holidays_events.csv')
df = data_processor.process_data()
df = data_processor.date_process(df)
df = data_processor.drop_duplicates_by_id(df)

df = FeatureEngineer.add_last_week_flag(df)
df = FeatureEngineer.add_national_locale_flag(df)
df = FeatureEngineer.on_promotion_flag(df)
df = FeatureEngineer.add_holiday_flag(df)

print("\n" + "="*50)
print("SEPARANDO TREINO E TESTE")
print("="*50)
df_train, df_test = data_processor.modelling_treatment(df)
print(f"Tamanho treino: {len(df_train)}")
print(f"Tamanho teste: {len(df_test)}")

print("\n" + "="*50)
print("CRIANDO FEATURES DE LAG E ROLLING")
print("="*50)

df_train = FeatureEngineer.lag_features(
    df_train, 
    target_col='sales',
    lags=[1, 2, 3, 7, 14, 21, 28]  # Adicionei lag de 28 dias
)

df_train = FeatureEngineer.rolling_mean_features(
    df_train, 
    target_col='sales',
    windows=[7, 14, 28, 56]  # Adicionei janela de 56 dias
)

df_train = FeatureEngineer.add_diff_features(df_train, 'sales')

df_combined = pd.concat([df_train, df_test], axis=0).sort_values(['store_nbr', 'family', 'date'])

df_combined = FeatureEngineer.lag_features(
    df_combined, 
    target_col='sales',
    lags=[1, 2, 3, 7, 14, 21, 28]
)

df_combined = FeatureEngineer.rolling_mean_features(
    df_combined, 
    target_col='sales',
    windows=[7, 14, 28, 56]
)

df_combined = FeatureEngineer.add_diff_features(df_combined, 'sales')

df_train = df_combined[df_combined['date'] < df_test['date'].min()].copy()
df_test = df_combined[df_combined['date'] >= df_test['date'].min()].copy()

print("\n" + "="*50)
print("ADICIONANDO MÉDIAS HISTÓRICAS")
print("="*50)
df_train = FeatureEngineer.add_historical_means(df_train, df_train)
df_test = FeatureEngineer.add_historical_means(df_test, df_train)  # Usar médias do treino!

df_train = FeatureEngineer.compute_monthly_sales_increase(df_train)
monthly_sales_train = df_train[['date_month_year', 'sales_increase']].drop_duplicates()
df_test = pd.merge(df_test, monthly_sales_train, on='date_month_year', how='left', suffixes=('', '_new'))
if 'sales_increase_new' in df_test.columns:
    df_test['sales_increase'] = df_test['sales_increase_new'].fillna(df_test['sales_increase'])
    df_test = df_test.drop('sales_increase_new', axis=1)

print("\n" + "="*50)
print("PREENCHENDO VALORES FALTANTES")
print("="*50)
df_train = DataCleanerAndPreparer.fill_missing_values(df_train)
df_test = DataCleanerAndPreparer.fill_missing_values(df_test)

print("\n" + "="*50)
print("ONE-HOT ENCODING")
print("="*50)
df_combined = pd.concat([df_train, df_test], axis=0)
df_combined = FeatureEngineer.one_hot_encode_categorical(
    df_combined, 
    ['family', 'city', 'state', 'store_type', 'transferred']
)

df_train = df_combined[df_combined['date'] < df_test['date'].min()].copy()
df_test = df_combined[df_combined['date'] >= df_test['date'].min()].copy()

print("\n" + "="*50)
print("REMOVENDO OUTLIERS")
print("="*50)
df_train = DataCleanerAndPreparer.remove_outliers_iqr(df_train, 'sales')

print("\n" + "="*50)
print("PREPARANDO DADOS PARA MODELAGEM")
print("="*50)

df_train_num = df_train.select_dtypes(include=np.number)
df_test_num = df_test.select_dtypes(include=np.number)

print(f"Shape treino antes de dropar NaN: {df_train_num.shape}")
print(f"NaN no treino: {df_train_num.isnull().sum().sum()}")

df_train_num = df_train_num.dropna()
print(f"Shape treino depois de dropar NaN: {df_train_num.shape}")

X_full = df_train_num.drop(['id', 'sales'], axis=1)
y_full = np.log1p(df_train_num['sales'])

df_test_num = df_test_num.fillna(0)

missing_cols = set(X_full.columns) - set(df_test_num.columns)
for col in missing_cols:
    df_test_num[col] = 0

X_test = df_test_num[X_full.columns]

print(f"Features no modelo: {X_full.shape[1]}")
print(f"Amostras treino: {len(X_full)}")
print(f"Amostras teste: {len(X_test)}")

print("\n" + "="*50)
print("TIME SERIES CROSS-VALIDATION")
print("="*50)

df_train_with_dates = df_train_num.merge(df_train[['id', 'date']], on='id', how='left')

X_full = df_train_with_dates.drop(['id', 'sales', 'date'], axis=1)
y_full = np.log1p(df_train_with_dates['sales'])
dates = df_train_with_dates['date']

n_folds = 3
total_days = (df_train['date'].max() - df_train['date'].min()).days
fold_size = total_days // (n_folds + 1)

print(f"Total de dias no treino: {total_days}")
print(f"Tamanho de cada fold: {fold_size} dias")

best_iteration = 0
cv_scores = []

for fold in range(1, n_folds + 1):
    print(f"\n{'='*30}")
    print(f"FOLD {fold}/{n_folds}")
    print(f"{'='*30}")
    
    val_start = df_train['date'].min() + pd.Timedelta(days=fold_size * fold)
    val_end = val_start + pd.Timedelta(days=fold_size)
    
    train_mask = dates < val_start
    val_mask = (dates >= val_start) & (dates < val_end)
    
    X_t = X_full[train_mask]
    y_t = y_full[train_mask]
    X_v = X_full[val_mask]
    y_v = y_full[val_mask]
    
    print(f"Treino: {len(X_t)} amostras (até {val_start.date()})")
    print(f"Validação: {len(X_v)} amostras ({val_start.date()} a {val_end.date()})")
    
    if len(X_v) == 0:
        print("Validação vazia, pulando fold...")
        continue
    
    model_fold = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        tree_method='hist',
        n_jobs=-1,
        early_stopping_rounds=50  # ADICIONAR ISSO!
    )
    
    model_fold.fit(
        X_t, y_t,
        eval_set=[(X_v, y_v)],
        verbose=False
    )
    
    y_pred_val_log = model_fold.predict(X_v)
    y_pred_val = np.expm1(y_pred_val_log)
    y_true_val = np.expm1(y_v)
    
    rmsle_val = np.sqrt(mean_squared_error(np.log1p(y_true_val), np.log1p(y_pred_val)))
    rmse_val = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
    mae_val = mean_absolute_error(y_true_val, y_pred_val)
    
    print(f'Fold {fold} - RMSLE: {rmsle_val:.4f} | RMSE: {rmse_val:.2f}  |  MAE: {mae_val:.2f}')
    cv_scores.append(rmsle_val)  # Usar RMSLE para CV
    
    if hasattr(model_fold, 'best_iteration') and model_fold.best_iteration > 0:
        best_iteration += model_fold.best_iteration
        print(f'   Melhor iteração: {model_fold.best_iteration}')
    else:
        best_iteration += model_fold.n_estimators

print(f"\n{'='*50}")
print(f"RESULTADOS DA VALIDAÇÃO CRUZADA")
print(f"{'='*50}")
print(f"RMSLE médio: {np.mean(cv_scores):.4f} (± {np.std(cv_scores):.4f})")
print(f"Melhor iteração média: {best_iteration // max(n_folds, 1)}")

print("\n" + "="*50)
print("TREINANDO MODELO FINAL COM TODOS OS DADOS")
print("="*50)

split_date = df_train['date'].max() - pd.Timedelta(days=15)
train_mask = dates < split_date
val_mask = dates >= split_date

X_t = X_full[train_mask]
y_t = y_full[train_mask]
X_v = X_full[val_mask]
y_v = y_full[val_mask]

model_final = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=int(best_iteration // max(n_folds, 1) * 1.1) if best_iteration > 0 else 2000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    tree_method='hist',
    n_jobs=-1,
    early_stopping_rounds=50  # ADICIONAR ISSO!
)

print(f"Treinando com {model_final.n_estimators} estimadores...")

model_final.fit(
    X_t, y_t,
    eval_set=[(X_v, y_v)],
    verbose=100
)

y_pred_val_log = model_final.predict(X_v)
y_pred_val = np.expm1(y_pred_val_log)
y_true_val = np.expm1(y_v)

rmsle_final = np.sqrt(mean_squared_error(np.log1p(y_true_val), np.log1p(y_pred_val)))
rmse_final = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
mae_final = mean_absolute_error(y_true_val, y_pred_val)
print(f'\nValidação Final - RMSLE: {rmsle_final:.4f} | RMSE: {rmse_final:.2f}  |  MAE: {mae_final:.2f}')

print("\n" + "="*50)
print("PREDIÇÃO NO SET DE TESTE")
print("="*50)

y_pred_test_log = model_final.predict(X_test)
y_pred_test = np.expm1(y_pred_test_log)

PreprearerToSubmit.prepare_submission(y_pred_test, df_test, filename='submission.csv')

print("\n" + "="*50)
print("PROCESSO FINALIZADO!")
print("="*50)
print(f"\nRESUMO:")
print(f"- RMSLE CV médio: {np.mean(cv_scores):.4f} (± {np.std(cv_scores):.4f})")
print(f"- RMSLE Validação Final: {rmsle_final:.4f}")
print(f"- RMSE Validação Final: {rmse_final:.2f}")
print(f"- Melhor iteração: {model_final.best_iteration if hasattr(model_final, 'best_iteration') else 'N/A'}")
print(f"\n⚠️ IMPORTANTE: O Kaggle usa RMSLE como métrica!")
print(f"   Seu RMSLE local ({rmsle_final:.4f}) deve estar próximo ao score do Kaggle.")