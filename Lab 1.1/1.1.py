import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

class DataAnalyzer:
    def __init__(self, data: list[tuple[float, float]]):
        self.points: list[tuple[float, float]] = data
        self.len: int = len(self.points)
        self.x_col: int = 0
        self.y_col: int = 1
        self._column_selection()

    def _column_selection(self) -> None:
        while True:
            try:
                choice: int = int(input(f"Установить первый столбец как X? 1 - Да, 0 - Выбрать Y. (по умолчанию X -  {self.x_col})\n"))
                if choice in (0, 1):
                    self.x_col = int(choice)
                    self.y_col = 1 - self.x_col
                    break
            except ValueError:
                print("Номер столбца должен быть целым")

    @property
    def x_values(self) -> list[float]:
        return [point[self.x_col] for point in self.points]

    @property
    def y_values(self) -> list[float]:
        return [point[self.y_col] for point in self.points]

    @property
    def mean_values(self) -> tuple[float, float]:
        sum_x = sum(self.x_values)
        sum_y = sum(self.y_values)
        return sum_x / self.len, sum_y / self.len

    @property
    def min_values(self) -> tuple[float, float]:
        min_x = min(self.x_values)
        min_y = min(self.y_values)
        return min_x, min_y

    @property
    def max_values(self) -> tuple[float, float]:
        max_x = max(self.x_values)
        max_y = max(self.y_values)
        return max_x, max_y

    @property
    def linear_regression(self) -> tuple[float, float]:
        x = self.x_values
        y = self.y_values
        x_mean, y_mean = self.mean_values

        cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        ss_x = sum((xi - x_mean)**2 for xi in x)

        b_1 = cov / ss_x
        b_0 = y_mean - b_1 * x_mean

        return b_0, b_1

    def print_statistics(self) -> None:
        if not self.points:
            print('Нет данных')
            return

        av_x, av_y = self.mean_values
        min_x, min_y = self.min_values
        max_x, max_y = self.max_values

        print("\nСтатистика:")
        print(f"Количество точек: {self.len}")
        print(f"X: min={min_x:.2f}, max={max_x:.2f}, avg={av_x:.2f}")
        print(f"Y: min={min_y:.2f}, max={max_y:.2f}, avg={av_y:.2f}")

def get_points(file_path: str) -> list[tuple[float, float]]:
    data: list[tuple[float, float]] = []
    with open(file_path, "r") as f:
        for line in f:
            hours, score = line.strip().split(",")
            try:
                data.append((float(hours), float(score)))
            except ValueError:
                continue

    return data

matplotlib.use('tkagg')
plt.style.use('_mpl-gallery')

def plot_comparison(analyzer: DataAnalyzer) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    x_vals = np.array(analyzer.x_values)
    y_vals = np.array(analyzer.y_values)

    b0, b1 = analyzer.linear_regression
    y_pred = b1 * x_vals + b0


    ax1.scatter(x_vals, y_vals, color='green', label='Данные')
    ax1.set_title('Данные')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()


    ax2.scatter(x_vals, y_vals, color='green', label='Линейная регрессия')
    ax2.plot(x_vals, y_pred,
             '-',
             color='red',
             label=f'Регрессия: y = {b1:.2f}x + {b0:.2f}'
             )

    ax2.set_title('Данные с регрессией')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()


    ax3.scatter(x_vals, y_vals, color='green', label='Регрессия')
    ax3.plot(x_vals, y_pred, '-', color='red',
             label=f'Регрессия: y = {b1:.2f}x + {b0:.2f}')


    for xi, yi, yi_pred in zip(x_vals, y_vals, y_pred):
        error = abs(yi - yi_pred)
        square_size = abs(error)
        rect = Rectangle(
            (xi - square_size/2, min(yi,yi_pred)),
            width=square_size,
            height=square_size,
            color='black',
            alpha=0.3,
        )
        ax3.add_patch(rect)

    ax3.set_title('Квадраты ошибок')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.legend()

    plt.tight_layout()
    plt.show()

data: list[tuple[float, float]] = get_points(file_path=os.path.join("data/student_scores.csv"))
points: DataAnalyzer = DataAnalyzer(data)


points.print_statistics()
plot_comparison(points)
