import matplotlib
import numpy as np
from typing import Any
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from tabulate import tabulate


class SWLinearRegression:
    def __init__(self, X: np.ndarray, Y:np.ndarray):
        self.len: int = len(X)
        self.x: np.ndarray = X.flatten() if X.ndim > 1 else X
        self.y: np.ndarray = Y.flatten() if Y.ndim > 1 else Y

    @property
    def mean_values(self) -> tuple[float, float]:
        sum_x = sum(self.x)
        sum_y = sum(self.y)
        return sum_x / self.len, sum_y / self.len

    @property
    def linear_regression_coeffs(self) -> tuple[float, float]:
        x_mean, y_mean = self.mean_values

        cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(self.x, self.y))
        ss_x = sum((xi - x_mean)**2 for xi in self.x)

        if ss_x == 0:
            raise ValueError("Регрессия невозможна")

        b_1 = cov / ss_x
        b_0 = y_mean - b_1 * x_mean

        return b_0, b_1

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        b0, b1 = self.linear_regression_coeffs
        return b0 + b1 * x_values

    def __repr__(self) -> str:
        b0, b1 = self.linear_regression_coeffs
        return f"SWLinearRegression(intercept={b0:.2f}, slope={b1:.2f})"


def get_data() -> tuple[np.ndarray, np.ndarray]:
    diabetes: dict[str, Any] = load_diabetes()

    data: np.ndarray = diabetes['data']

    target: np.ndarray = diabetes['target']
    bmi_data: np.ndarray = data[:, 2]

    X: np.ndarray = bmi_data.reshape(-1, 1)
    Y: np.ndarray = target

    return X, Y

def print_predictions_table(X: np.ndarray, Y: np.ndarray, model: LinearRegression,
                            num_rows: int = 10) -> None:
    predictions = model.predict(X)

    table_data = []
    for i in range(num_rows):
        table_data.append([
            f"{X[i][0]:.4f}",
            f"{Y[i]:.1f}",
            f"{predictions[i]:.1f}",
            f"{(Y[i] - predictions[i]):.1f}"
        ])

    headers = ["BMI", "Реальное значение", "Прогноз", "Ошибка"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

matplotlib.use('tkagg')


def linear_regression(X: np.ndarray, Y: np.ndarray) -> LinearRegression:
    model: LinearRegression = LinearRegression()
    model.fit(X, Y)
    return model


def sw_linear_regression(X: np.ndarray, Y: np.ndarray) -> SWLinearRegression:
    model = SWLinearRegression(X, Y)
    return model

def print_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, model_name: str) -> None:
    mae = mean_absolute_error(Y_true, Y_pred)
    r2 = r2_score(Y_true, Y_pred)
    mape = mean_absolute_percentage_error(Y_true, Y_pred)

    print(f"\nМетрики для модели {model_name}:")
    print(f"MAE (Средняя абсолютная ошибка): {mae:.2f}")
    print(f"R² (Коэффициент детерминации): {r2:.2f}")
    print(f"MAPE (Средняя абсолютная процентная ошибка): {mape:.2%}")

def main() -> None:
    X, Y = get_data()

    sklearn_model: LinearRegression = linear_regression(X, Y)

    sw_model: SWLinearRegression = sw_linear_regression(X, Y)

    models = [
        (sklearn_model, "Scikit-Learn LinearRegression"),
        (sw_model, "SW LinearRegression")
    ]

    for model, name in models:
        predictions = model.predict(X)
        print_metrics(Y, predictions, name)

main()
