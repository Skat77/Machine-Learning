import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
import seaborn as sns
import os


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

def logistic_reg(df: DataFrame):
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    return y_test, y_pred, y_proba, model, X_train, X_test, y_train, scaler


def plot_metrics(y_test, y_pred, y_proba):
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Precision (weighted): {precision:.3f}")
    print(f"Recall (weighted): {recall:.3f}")
    print(f"F1-score (weighted): {f1:.3f}\n")

    plt.figure(figsize=(15, 5))


    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Reds',
        xticklabels=['Not Survived', 'Survived'],
        yticklabels=['Not Survived', 'Survived']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.subplot(1, 3, 2)
    precision_vals, recall_vals, _ = precision_recall_curve(
        y_test, y_proba[:, 1]
    )
    plt.plot(recall_vals, precision_vals, lw=2, color='green')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color='blue',
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    path = os.path.join("data/Titanic.csv")
    df = pd.read_csv(path)
    df_final = prepare_data(df)

    y_test, y_pred, y_proba, model, X_train, X_test, y_train, scaler = logistic_reg(df_final)

    plot_metrics(y_test, y_pred, y_proba)

main()
