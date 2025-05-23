import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import itertools
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')


class DataAnalyzePairplot:
    def __init__(self, iris_data):
        self.df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        self.df['target'] = iris_data.target
        self.df['species'] = self.df['target'].map(
            {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        )

        self.feature_names = [name.replace(" (cm)", "") for name in iris_data.feature_names]
        self.n_features = len(self.feature_names)


    def pairplot(self, save_path=None):
        plot = sns.pairplot(
            self.df,
            hue='species',
            palette='viridis',
            corner=True,
            diag_kind='kde',
            plot_kws={'alpha': 0.7, 's': 25},
            height=2.5
        )


        for i in range(self.n_features):
            for j in range(self.n_features):
                if i >= j:
                    ax = plot.axes[i, j]
                    if i == j:
                        ax.set_ylabel('')
                    else:
                        if j == 0:
                            ax.set_ylabel(self.feature_names[i])
                        if i == self.n_features - 1:
                            ax.set_xlabel(self.feature_names[j])

        plt.suptitle('Pairplot Iris Dataset', y=1.02)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

class DataAnalyzePyplot:
    def __init__(self, data_iris):
        self.iris = data_iris
        self.df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        self.df['target']: list[str] = self.iris.target
        self.feature_names = [name.replace(" (cm)", "") for name in self.iris.feature_names]

        self.X = self.iris.data
        self.y = self.iris.target
        self.target_names = self.iris.target_names

    def _get_feature_pairs(self) -> list[tuple[int, int, str, str]]:
        pairs = []
        for i, j in itertools.combinations(range(len(self.feature_names)), 2):
            pairs.append((i, j, self.feature_names[i], self.feature_names[j]))
        return pairs

    def plot(self) -> None:
        feature_pairs = self._get_feature_pairs()
        n_pairs = len(feature_pairs)

        n_rows = (n_pairs + 1) // 2
        plt.figure(figsize=(12, 6))

        for plot_num, (x_idx, y_idx, x_label, y_label) in enumerate(feature_pairs, 1):
            plt.subplot(n_rows, 2, plot_num)

            for target in np.unique(self.y):
                mask = self.y == target
                plt.scatter(self.X[mask, x_idx],
                            self.X[mask, y_idx],
                            label=self.target_names[target])

            plt.xlabel(f'{x_label} (cm)')
            plt.ylabel(f'{y_label} (cm)')
            plt.legend()
            plt.title(f'{x_label} - {y_label}')

        plt.tight_layout()
        plt.show()

def get_iris_subsets(iris) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    df['species'] = [iris.target_names[i] for i in iris.target]

    df_set_vers = df[df['species'].isin(['setosa', 'versicolor'])].copy()
    df_vers_virg = df[df['species'].isin(['versicolor', 'virginica'])].copy()

    return df_set_vers, df_vers_virg

matplotlib.use('TkAgg')


def train_and_evaluate(X: np.ndarray, y: np.ndarray, title) -> None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(random_state=0)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{title} - Точность модели: {accuracy:.4f}")


def iris_classification():
    df_set_vers, df_vers_virg = get_iris_subsets(load_iris())

    X1 = df_set_vers.iloc[:, :4].values
    y1 = (df_set_vers['species'] == 'versicolor').astype(int).values
    train_and_evaluate(X1, y1, "Setosa vs Versicolor")


    X2 = df_vers_virg.iloc[:, :4].values
    y2 = (df_vers_virg['species'] == 'virginica').astype(int).values
    train_and_evaluate(X2, y2, "Versicolor vs Virginica")


def synthetic_classification():

    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma', alpha=0.7)
    plt.title("Сгенерированный датасет для бинарной классификации")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.colorbar()
    plt.show()

    train_and_evaluate(X, y, "Синтетический датасет")

iris_classification()

synthetic_classification()
