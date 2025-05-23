import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

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




def logistic_reg(df: DataFrame):
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)


    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Точность модели: {accuracy:.2f}")

    return y_test, y_pred, model, X_train, X_test, y_train, scaler


def assessing_impact(y_test, y_pred, model, X_train, X_test, y_train, scaler):
    X_train_no_emb = X_train.drop('Embarked', axis=1)
    X_test_no_emb = X_test.drop('Embarked', axis=1)

    X_train_no_emb_scaled = scaler.fit_transform(X_train_no_emb)
    X_test_no_emb_scaled = scaler.transform(X_test_no_emb)


    model_no_emb = LogisticRegression(max_iter=1000)
    model_no_emb.fit(X_train_no_emb_scaled, y_train)

    y_pred_no_emb = model_no_emb.predict(X_test_no_emb_scaled)
    accuracy_no_emb = accuracy_score(y_test, y_pred_no_emb)

    print(f"\nТочность без Embarked: {accuracy_no_emb:.4f}")
    print(f"Разница в точности: {accuracy_score(y_test, y_pred) - accuracy_no_emb:.4f}")

    print("\nКоэффициенты исходной модели:")
    print(pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', ascending=False))


def main():
    path = os.path.join("data/Titanic.csv")
    df = pd.read_csv(path)
    df_final = prepare_data(df)

    print("Распределение значений Embarked:")
    print(df_final['Embarked'].value_counts())

    y_test, y_pred, model, X_train, X_test, y_train, scaler = logistic_reg(df_final)
    assessing_impact(y_test, y_pred, model, X_train, X_test, y_train, scaler)

main()
