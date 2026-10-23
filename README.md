# ML Experiments

Пять учебных проектов: регрессия, ансамбли, кластеризация и классификация изображений. В ноутбуках сохранены код, метрики и графики.

**Стек:** Python, pandas, NumPy, scikit-learn, SciPy, PyTorch, torchvision, matplotlib, seaborn.

## Classic ML

### Медицинские расходы: OLS и Lasso

[Ноутбук](notebooks/insurance_regression.ipynb) · [Метрики](results/insurance_regression_metrics.csv)

- EDA набора из 1338 строк, агрегации и объединение таблиц в pandas.
- `Pipeline` с масштабированием и one-hot encoding, подбор Lasso через `GridSearchCV`, сравнение фильтрации выбросов.
- OLS на test из 268 строк: **R² 0,7836, RMSE 5796,28, MAE 4181,19**. Lasso не улучшил результат; агрессивная фильтрация target снизила качество.

### Цены автомобилей: деревья и ансамбли

[Ноутбук](notebooks/used_cars_ensembles.ipynb) · [Метрики](results/used_cars_metrics.csv)

- Обработка пропусков и редких категорий; linear regression, kNN, decision tree, bagging, blending и stacking.
- Подбор бэггинга: **360 конфигураций × 5 folds**. Для blending выделена meta-выборка, для stacking используются OOF-предсказания.
- Базовый bagging: **RMSE 1791,49, MAPE 23,35%** на test 20%, seed 42. Подбор параметров не улучшил test-метрики.
- Отдельная реализация random forest на `DecisionTreeClassifier` с bootstrap и усреднением вероятностей.

Пропуски и частоты категорий рассчитаны до train/test split. Для строгой оценки эту обработку нужно перенести внутрь folds.

### Кластеризация: k-means, DBSCAN и PCA

[Ноутбук](notebooks/insurance_clustering.ipynb)

Собственный k-means и сравнение с `KMeans`, `AgglomerativeClustering`, `DBSCAN`. Масштабирование, PCA и визуализация кластеров. ARI относительно регионов близок к нулю: **0,0019** у собственного k-means, **0,0025** у sklearn KMeans после scaling и PCA. Найденные кластеры почти не совпадают с региональной разметкой.

[Методы, результаты и запуск Classic ML](docs/classic-ml.md)

## Computer Vision

### ResNet-50: transfer learning

[Ноутбук](notebooks/resnet50_transfer_learning.ipynb) · [Результаты четырёх разбиений](results/resnet50_four_splits.csv)

450 изображений ImageNet, девять классов объединены в три метакласса. Сравнены замороженный backbone, частичный и полный fine-tuning, линейная и MLP-головы: **8 конфигураций × 4 разбиения = 32 обучения**.

Выбранная по validation конфигурация на seed 42: **67/68 верных ответов на test, accuracy 98,53%, macro-F1 0,9852**. Выборка небольшая и относится к домену предобучения. Test-метрики показаны для каждой конфигурации.

### Спортивные фотографии: residual CNN с нуля

[Ноутбук](notebooks/sports_residual_cnn.ipynb)

30 видов спорта, 4002 обучающих изображения. Собственные Dataset и четыре residual-блока, аугментации, AdamW, CyclicLR и сохранение checkpoint. На последней эпохе **accuracy 82% на 150 отложенных изображениях**. Эта часть используется как validation для выбора checkpoint; отдельного финального test нет.

[Архитектуры, графики и запуск Computer Vision](docs/computer-vision.md)

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Открывайте каждый ноутбук в новом kernel и выполняйте сверху вниз. Classic ML не требует GPU; sports CNN использует CUDA. Полный grid search ансамблей может занять заметное время. Точные версии исходных сред не зафиксированы.

Данные и checkpoint скачиваются отдельно. Пути описаны в документации, ссылки на источники собраны в [PROVENANCE.md](PROVENANCE.md).
