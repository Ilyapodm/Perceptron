from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

SAMPLES = 500
SEED = 42
TRAIN_SIZE = 0.7 # 70%
VAL_SIZE = 0.66 # 20% из 30%, оставшихся после разбиения на train

# создание датасета
X, y = make_classification(
    n_samples=SAMPLES,  # количество примеров
    n_features=2,  # количество признаков
    n_informative=2,  # количество признаков, от которых реально зависит y
    n_redundant=0,  # количество признаков, выражающихся через другие
    random_state=SEED,  # seed для воспроизводимости
    n_clusters_per_class=1  # один класс на кластер -> линейно разделимые классы
)

# первое стратифицированное разбиение
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    stratify=y
)

# второе стратифицированное разбиение
X_val, X_test, y_val, y_test = train_test_split(
    X_test, 
    y_test,
    train_size=VAL_SIZE,
    random_state=SEED,
    stratify=y_test
)

# нормировка данных по train_set!!
for col in range(2):
    mean = X_train[:, col].mean()  # среднее 
    std = X_train[:, col].std()  # стандартное отклонение
    X_train[:, col] = (X_train[:, col] - mean) / std
    X_val[:, col] = (X_val[:, col] - mean) / std
    X_test[:, col] = (X_test[:, col] - mean) / std

# print(np.bincount(y_train))
# print(np.bincount(y_val))
# print(np.bincount(y_test))

