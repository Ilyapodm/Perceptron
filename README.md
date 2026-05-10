# Однослойный перцептрон

Реализация однослойного перцептрона с нуля на NumPy для задачи бинарной классификации.

## Архитектура

Перцептрон реализует логистическую регрессию:

$$\hat{y} = \sigma(w^T x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

**Функция потерь** — бинарная кросс-энтропия:

$$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log\hat{y}^{(i)} + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]$$

**Градиенты** (выведены аналитически через chain rule):

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{m} X^T (\hat{y} - y), \quad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m}\sum(\hat{y} - y)$$

Обновление весов через мини-батчевый SGD:

$$w \leftarrow w - \eta \cdot \nabla_w \mathcal{L}$$

## Данные

Синтетический датасет (`make_classification`): 500 примеров, 2 признака, линейно разделимые классы (`n_clusters_per_class=1`).

Разбиение с стратификацией (сохранение пропорции классов):

| Выборка | Размер | Назначение |
|---------|--------|------------|
| Train   | 70%    | Обучение, обновление весов |
| Val     | 20%    | Мониторинг loss во время обучения |
| Test    | 10%    | Финальная честная оценка |

Нормировка — Z-score по **train**, параметры применяются к val и test.

## Структура проекта
```
├── src/
│    ├── perceptron.py          ← Класс перцептрона
│    └── data_preprocessing.py  ← Загрузка и подготовка данных
├── helpers/
│    └── helpers.py             ← Утилиты вывода метрик
├── experiments/
│    ├── exp_lr.py              ← Влияние learning rate
│    ├── exp_batch_size.py      ← Влияние размера батча
│    └── exp_weights_init.py    ← Влияние инициализации весов
├── main.py                     ← Обучение + метрики + графики
├── report.md                   ← Анализ экспериментов
└── README.md                   ← Описание архитектуры
```


## Запуск

```bash
uv run main.py                                # основное обучение
uv run experiments/exp_lr.py                  # эксперимент: learning rate
uv run experiments/exp_batch_size.py          # эксперимент: batch size
uv run experiments/exp_weights_init.py        # эксперимент: инициализация
```

## Метрики

На тестовой выборке вычисляются: Accuracy, Precision, Recall, F1, ROC-AUC.