import pandas as pd
from pandas import DataFrame
import os
import time

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def prepare_data(df: DataFrame) -> DataFrame:
    def _cleaned_data(df: DataFrame) -> DataFrame:
        df_cleaned = df.dropna()

        cols_keep = [
            col
            for col in df_cleaned.columns
            if pd.api.types.is_numeric_dtype(df_cleaned[col]) or col in ['Sex', 'Embarked']
            if col != 'PassengerId'
        ]

        return df_cleaned[cols_keep]

    def _standardization(df: DataFrame) -> DataFrame:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
        return df

    cleaned_df = _cleaned_data(df)
    return _standardization(cleaned_df)

matplotlib.use('tkagg')

path = os.path.join('..', 'data', 'diabetes.csv')
df = pd.read_csv(path)

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

def investigate_parameters(param_grid, X_train, y_train, X_test, y_test):
    results = []
    best_score = 0
    best_params = {}

    for max_depth in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for n in param_grid['n_estimators']:
                for subsample in param_grid['subsample']:
                    for colsample in param_grid['colsample_bytree']:
                        start_time = time.time()

                        model = XGBClassifier(
                            max_depth=max_depth,
                            learning_rate=lr,
                            n_estimators=n,
                            subsample=subsample,
                            colsample_bytree=colsample,
                            random_state=42,
                            eval_metric='logloss'
                        )

                        model.fit(X_train, y_train)
                        train_time = time.time() - start_time

                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]

                        metrics = {
                            'max_depth': max_depth,
                            'learning_rate': lr,
                            'n_estimators': n,
                            'subsample': subsample,
                            'colsample_bytree': colsample,
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Recall': recall_score(y_test, y_pred),
                            'F1': f1_score(y_test, y_pred),
                            'Time': train_time
                        }

                        results.append(metrics)

                        if metrics['F1'] > best_score:
                            best_score = metrics['F1']
                            best_params = {
                                'max_depth': max_depth,
                                'learning_rate': lr,
                                'n_estimators': n,
                                'subsample': subsample,
                                'colsample_bytree': colsample
                            }

    return pd.DataFrame(results), best_params, best_score

results_df, best_params, best_score = investigate_parameters(param_grid, X_train, y_train, X_test, y_test)

results_df.to_csv('xgb_parameter_search_results.csv', index=False)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
for lr in param_grid['learning_rate']:
    subset = results_df[results_df['learning_rate'] == lr]
    plt.plot(subset['n_estimators'], subset['F1'], marker='o', label=f'lr={lr}')
plt.xlabel('Количество деревьев')
plt.ylabel('F1-score')
plt.title('Влияние learning_rate и количества деревьев')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
for depth in param_grid['max_depth']:
    subset = results_df[results_df['max_depth'] == depth]
    plt.plot(subset['learning_rate'], subset['F1'], marker='o', label=f'depth={depth}')
plt.xlabel('Learning rate')
plt.ylabel('F1-score')
plt.title('Влияние глубины деревьев и learning_rate')
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
for subsample in param_grid['subsample']:
    subset = results_df[results_df['subsample'] == subsample]
    plt.plot(subset['colsample_bytree'], subset['F1'], marker='o', label=f'subsample={subsample}')
plt.xlabel('colsample_bytree')
plt.ylabel('F1-score')
plt.title('Влияние subsample и colsample_bytree')
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.scatter(results_df['n_estimators'], results_df['Time'], c=results_df['max_depth'], cmap='viridis')
plt.colorbar(label='max_depth')
plt.xlabel('Количество деревьев')
plt.ylabel('Время обучения (с)')
plt.title('Время обучения в зависимости от параметров')
plt.grid()

plt.tight_layout()
plt.show()

best_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nЛучшие параметры модели:")
print(best_params)
print(f"\nМетрики лучшей модели:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матрица ошибок')
plt.colorbar()
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.xticks([0, 1], ['Не диабет', 'Диабет'])
plt.yticks([0, 1], ['Не диабет', 'Диабет'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.show()

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Важность признака')
plt.title('Важность признаков в модели XGBoost')
plt.gca().invert_yaxis()
plt.show()
