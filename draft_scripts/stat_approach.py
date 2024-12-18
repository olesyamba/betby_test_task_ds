import numpy as np
import pandas as pd
from scipy.stats import binom, geom

# Загрузка данных
file_paths = {
    "first_pair": "path_to/test_data_first_pair.csv",
    "second_pair": "path_to/test_data_second_pair.csv",
    "third_pair": "path_to/test_data_third_pair.csv",
}

data_frames = {name: pd.read_csv(path).assign(pair=name) for name, path in file_paths.items()}
full_data = pd.concat(data_frames.values(), ignore_index=True)

# Целевая переменная для подсчета эмпирических вероятностей
full_data['target'] = full_data['home_score'].astype(int).astype(str) + ":" + full_data['away_score'].astype(
    int).astype(str)


## 1. Биномиальное распределение
def compute_binomial_prob(n_home, p_home, n_away, p_away):
    """
    Оценка вероятности по биномиальному распределению
    n_home, n_away: максимальное количество возможных попыток (голов).
    p_home, p_away: вероятность забить гол.
    """
    # Биномиальное распределение для домашних и гостевых команд
    home_distribution = binom.pmf(range(n_home + 1), n_home, p_home)  # Вероятности от 0 до n_home голов
    away_distribution = binom.pmf(range(n_away + 1), n_away, p_away)  # Вероятности от 0 до n_away голов

    # Вероятностная матрица - комбинаторные перемножения
    matrix = np.outer(home_distribution, away_distribution)

    # Нормализация
    matrix /= matrix.sum()
    return matrix


## 2. Геометрическое распределение
def compute_geometric_prob(p_home, p_away, max_goals_home, max_goals_away):
    """
    Геометрическое распределение для оценки вероятности первого гола.
    p_home, p_away: вероятность забить первый гол (по геометрическому распределению).
    max_goals_home, max_goals_away: диапазон голов.
    """
    home_distribution = np.array([geom.pmf(k, p_home) for k in range(max_goals_home)])
    away_distribution = np.array([geom.pmf(k, p_away) for k in range(max_goals_away)])

    # Перемножение вероятностей для каждой комбинации
    matrix = np.outer(home_distribution, away_distribution)

    # Нормализация
    matrix /= matrix.sum()
    return matrix


## 3. Эмпирические вероятности
empirical_counts = full_data['target'].value_counts()
total_games = len(full_data)
empirical_probabilities = empirical_counts / total_games

empirical_prob_df = pd.DataFrame({
    "observed_combination": empirical_probabilities.index,
    "probability": empirical_probabilities.values
})

print("\nЭмпирические вероятности:")
print(empirical_prob_df)

empirical_prob_df.to_csv("empirical_probabilities.csv", index=False)

## 1. Биномиальные вероятности
n_home, n_away = 7, 6  # Максимальное число голов в домашней и гостевой команде
p_home, p_away = 0.3, 0.2  # Вероятность гола

binomial_matrix = compute_binomial_prob(n_home, p_home, n_away, p_away)

# Сохранение биномиальных вероятностей
binomial_df = pd.DataFrame(
    binomial_matrix,
    index=range(n_home + 1),
    columns=range(n_away + 1)
)
binomial_df.to_csv("binomial_probabilities.csv", index=False)
print("\nБиномиальные вероятности:")
print(binomial_df)

## 2. Геометрические вероятности
p_home_geom, p_away_geom = 0.2, 0.1
max_goals_home, max_goals_away = 7, 6

geometric_matrix = compute_geometric_prob(p_home_geom, p_away_geom, max_goals_home, max_goals_away)

# Сохранение геометрических вероятностей
geometric_df = pd.DataFrame(
    geometric_matrix,
    index=range(max_goals_home),
    columns=range(max_goals_away)
)

geometric_df.to_csv("geometric_probabilities.csv", index=False)

print("\nГеометрические вероятности:")
print(geometric_df)
