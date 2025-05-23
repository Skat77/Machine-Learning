import pandas as pd
from pandas import DataFrame
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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

path = os.path.join('data/diabetes.csv')
df = pd.read_csv(path)


X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_proba_log = log_reg.predict_proba(X_test)[:, 1]

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
y_proba_tree = tree.predict_proba(X_test)[:, 1]

depths = range(1, 21)
recall_scores = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    recall_scores.append(recall)

plt.figure(figsize=(10, 6))
plt.plot(depths, recall_scores, marker='o', linestyle='-', color='r')
plt.title('Зависимость Recall от глубины дерева')
plt.xlabel('Глубина дерева')
plt.ylabel('Recall')
plt.grid(True)
plt.xticks(depths)
plt.show()
