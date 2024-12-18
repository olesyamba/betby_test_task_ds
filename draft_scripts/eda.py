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

# Нахождение минимального и максимального счета
min_home_score = full_data['home_score'].min()
max_home_score = full_data['home_score'].max()
min_away_score = full_data['away_score'].min()
max_away_score = full_data['away_score'].max()

# Среднее и стандартное отклонение
home_score_mean = full_data['home_score'].mean()
home_score_std = full_data['home_score'].std()
away_score_mean = full_data['away_score'].mean()
away_score_std = full_data['away_score'].std()

print("Home Score - Min:", min_home_score, "Max:", max_home_score,
      "Mean:", home_score_mean, "Std Dev:", home_score_std)
print("Away Score - Min:", min_away_score, "Max:", max_away_score,
      "Mean:", away_score_mean, "Std Dev:", away_score_std)
