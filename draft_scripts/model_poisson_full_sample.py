import math

import numpy as np
import pandas as pd

# Загрузка данных
file_paths = {
    "first_pair": "data/test_data_first_pair.csv",
    "second_pair": "data/test_data_second_pair.csv",
    "third_pair": "data/test_data_third_pair.csv",
}

# Чтение файлов с добавлением идентификатора пары
data_frames = {name: pd.read_csv(path).assign(pair=name) for name, path in file_paths.items()}

# Объединение всех данных
full_data = pd.concat(data_frames.values(), ignore_index=True)
del data_frames, file_paths

# Указанные значения средних голов
home_mean = full_data['home_score'].mean()  # Среднее количество голов для домашней команды
away_mean = full_data['away_score'].mean()  # Среднее количество голов для гостевой команды

# Диапазоны значений голов
home_score_range = range(full_data['home_score'].min(), full_data['home_score'].max())  # От 0 до 7
away_score_range = range(full_data['away_score'].min(), full_data['away_score'].max())  # От 0 до 6

# Функция для вычисления вероятности Пуассона
def poisson_probability(lmbda, k):
    """
    Вычисляет вероятность Пуассона для k событий при среднем lmbda.
    """
    return (np.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)

# Создание матрицы вероятностей
prob_matrix = np.zeros((len(home_score_range), len(away_score_range)))

for i, home_score in enumerate(home_score_range):
    for j, away_score in enumerate(away_score_range):
        home_prob = poisson_probability(home_mean, home_score)
        away_prob = poisson_probability(away_mean, away_score)
        prob_matrix[i, j] = home_prob * away_prob  # Независимые вероятности

# Преобразование в DataFrame для удобства сохранения
prob_df = pd.DataFrame(
    prob_matrix,
    index=[f"Home {i}" for i in home_score_range],
    columns=[f"Away {j}" for j in away_score_range],
)

# Сохранение в CSV в требуемом формате
output_csv_path = "scores.csv"  # Укажите нужный путь
prob_df.to_csv(output_csv_path)

print(f"Матрица вероятностей успешно сохранена в файл {output_csv_path}")

# Учет преимущества домашней команды
home_std = full_data['home_score'].std()
away_std = full_data['away_score'].std()
home_advantage = home_std - away_std
adjusted_home_mean = home_mean + home_advantage

# Функция для вычисления вероятности Пуассона
def poisson_probability(lmbda, k):
    """
    Вычисляет вероятность Пуассона для k событий при среднем lmbda.
    """
    return (np.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)

# Создание матрицы вероятностей с учетом корректировки
adjusted_prob_matrix = np.zeros((len(home_score_range), len(away_score_range)))

for i, home_score in enumerate(home_score_range):
    for j, away_score in enumerate(away_score_range):
        home_prob = poisson_probability(adjusted_home_mean, home_score)
        away_prob = poisson_probability(away_mean, away_score)
        adjusted_prob_matrix[i, j] = home_prob * away_prob

# Преобразование в DataFrame для удобства сохранения
adjusted_prob_df = pd.DataFrame(
    adjusted_prob_matrix,
    index=[f"Home {i}" for i in home_score_range],
    columns=[f"Away {j}" for j in away_score_range],
)

# Сохранение в CSV
adjusted_csv_path = "adjusted_scores.csv"  # Укажите путь для сохранения файла
adjusted_prob_df.to_csv(adjusted_csv_path)

print(f"Матрица вероятностей с учетом домашнего преимущества сохранена в файл {adjusted_csv_path}")

