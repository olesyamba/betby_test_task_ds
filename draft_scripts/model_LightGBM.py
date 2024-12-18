import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
# xgoals
# Загрузка данных
file_paths = {
    "first_pair": "data/test_data_first_pair.csv",
    "second_pair": "data/test_data_second_pair.csv",
    "third_pair": "data/test_data_third_pair.csv",
}

data_frames = {name: pd.read_csv(path).assign(pair=name) for name, path in file_paths.items()}
full_data = pd.concat(data_frames.values(), ignore_index=True)

# Создание целевой переменной с только существующими классами
full_data['target'] = full_data['home_score'] * 10 + full_data['away_score']

# Оставляем только те классы, в которых есть 2 и более примеров
class_counts = full_data['target'].value_counts()
valid_classes = class_counts[class_counts >= 2].index  # Фильтруем классы с >= 2 примерами
full_data = full_data[full_data['target'].isin(valid_classes)]


# # Фильтруем только те классы, которые присутствуют в данных
# unique_classes = full_data['target'].unique()
# full_data = full_data[full_data['target'].isin(unique_classes)]

# Признаки и целевая переменная
features = ['home_score', 'away_score']
X = full_data[features]
y = full_data['target']

# Кодирование меток (LabelEncoder)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Перекодируем метки в диапазон [0, num_classes)

# Подсчёт количества классов
num_classes = len(label_encoder.classes_)

# Разделение данных на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.unique(y))

# Параметры для обучения LightGBM
params = {
    "objective": "multiclass",
    "num_class": num_classes,  # Количество классов соответствует уникальным значениям с учетом фильтрации по количеству участников в классе
    "boosting_type": "gbdt",
    "metric": "multi_error",
    "max_depth": 10,
    "num_leaves": 31,
    "min_data_in_leaf": 20,  # Минимальное количество данных на лист
    "learning_rate": 0.05,
}

# Создание и обучение модели
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_test,
                  callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(10)])  #

# Предсказания вероятностей
y_pred_prob = model.predict(X_test)

# Определяем предсказанные метки
y_pred = np.argmax(y_pred_prob, axis=1)

# Оценка точности и Cross-Entropy Loss
accuracy = accuracy_score(y_test, y_pred)
cross_entropy_loss = log_loss(y_test, y_pred_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Cross-Entropy Loss: {cross_entropy_loss:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm")
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.title('Confusion Matrix')
plt.show()
