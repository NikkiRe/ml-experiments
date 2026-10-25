# ML Experiments

Шесть учебных проектов: регрессия, ансамбли, градиентный бустинг, кластеризация и классификация изображений. В ноутбуках сохранены код, метрики и графики.

Лабораторные работы выполнены совместно с Кириллом Слепцовым; ноутбук с градиентным бустингом сделан отдельно, после курса. Исходные Colab и детали — в [PROVENANCE.md](PROVENANCE.md).

**Стек:** Python, pandas, Polars, NumPy, scikit-learn, CatBoost, LightGBM, XGBoost, Optuna, SciPy, PyTorch, torchvision, matplotlib, seaborn.

## Classic ML

### Медицинские расходы: OLS и Lasso

[Ноутбук](notebooks/insurance_regression.ipynb) · [Метрики](results/insurance_regression_metrics.csv)

- EDA набора из 1338 строк, агрегации и объединение таблиц в pandas.
- `Pipeline` с масштабированием и one-hot encoding, подбор `alpha` для Lasso через `GridSearchCV`, сравнение двух вариантов фильтрации выбросов.
- OLS на test из 268 строк: **R² 0,7836, RMSE 5796,28, MAE 4181,19**. Lasso не улучшил результат; фильтрация выбросов с учётом target снизила качество.

### Цены автомобилей: деревья и ансамбли

[Ноутбук](notebooks/used_cars_ensembles.ipynb) · [Метрики](results/used_cars_metrics.csv)

- Обработка пропусков и редких категорий; linear regression, kNN, decision tree, bagging, blending и stacking.
- Подбор бэггинга: **360 конфигураций × 5 folds**. Для blending выделена meta-выборка, в stacking метамодель обучается на OOF-предсказаниях.
- Базовый bagging: **RMSE 1791,49, MAPE 23,35%** на test 20%, seed 42. Подбор параметров не улучшил test-метрики.
- Собственный random forest на `DecisionTreeClassifier` (bootstrap, усреднение вероятностей) для классификации региона в insurance.

Пропуски и частоты категорий рассчитаны до train/test split. Для строгой оценки эту обработку нужно перенести внутрь folds.

### Медицинские расходы: градиентный бустинг

[Ноутбук](notebooks/gradient_boosting.ipynb) · [Регрессия](results/boosting_regression_metrics.csv) · [Классификация](results/boosting_classification_metrics.csv)

- Тот же split, что у OLS. Число деревьев, гиперпараметры CatBoost и порог классификации подбираются по 5-fold CV на train; test используется только для финального сравнения.
- CatBoost, LightGBM, XGBoost и `HistGradientBoostingRegressor`. Early stopping внутри folds, подбор CatBoost через Optuna.
- SHAP показал взаимодействие smoker × BMI ≥ 30. Три признака (`obese`, `smoker_obese`, `age2`) подняли R² OLS на test с **0,7836 до 0,8773**. У CatBoost после Optuna **R² 0,8823, RMSE 4276**.
- Классификация дорогих клиентов (`charges` не ниже 80-го перцентиля train): порог вероятности максимизирует F1 на OOF-предсказаниях. На test F1 **0,863** у CatBoost и **0,860** у логистической регрессии с теми же тремя признаками.

### Кластеризация: k-means, DBSCAN и PCA

[Ноутбук](notebooks/insurance_clustering.ipynb)

Собственный k-means и сравнение с `KMeans`, `AgglomerativeClustering`, `DBSCAN`. Масштабирование, PCA и визуализация кластеров. ARI относительно регионов близок к нулю: **0,0019** у собственного k-means, **0,0025** у sklearn KMeans после scaling и PCA. Найденные кластеры почти не совпадают с региональной разметкой.

[Методы, результаты и запуск Classic ML](docs/classic-ml.md)

## Computer Vision

### ResNet-50: transfer learning

[Ноутбук](notebooks/resnet50_transfer_learning.ipynb) · [Результаты четырёх разбиений](results/resnet50_four_splits.csv)

450 изображений ImageNet, девять классов объединены в три метакласса. Сравнены замороженный backbone, частичный и полный fine-tuning, линейная и MLP-головы: **8 конфигураций × 4 разбиения = 32 обучения**.

Выбранная по validation конфигурация на seed 42 (`G_frozen_lr1e-2`, замороженный backbone): **67/68 верных ответов на test, accuracy 98,53%, macro-F1 0,9852**. Выборка небольшая и относится к домену предобучения. Test-метрики показаны для каждой конфигурации.

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

Открывайте каждый ноутбук в новом kernel и выполняйте сверху вниз. Classic ML не требует GPU; sports CNN рассчитан на CUDA. Полный grid search ансамблей может занять заметное время. Версии библиотек для `gradient_boosting.ipynb` указаны в [PROVENANCE.md](PROVENANCE.md); для лабораторных работ точные версии окружений не сохранились.

Данные и веса в репозиторий не включены. Датасеты скачиваются по ссылкам из [PROVENANCE.md](PROVENANCE.md), пути к ним описаны в [docs/classic-ml.md](docs/classic-ml.md) и [docs/computer-vision.md](docs/computer-vision.md). Checkpoint создаются при обучении.
