# Classic ML

## Insurance: регрессия

[Ноутбук](../notebooks/insurance_regression.ipynb) · [Метрики](../results/insurance_regression_metrics.csv)

Задача: предсказать `charges` по `age`, `sex`, `bmi`, `children`, `smoker`, `region`. В наборе 1338 строк; split 80/20 с seed 42 даёт 1070 train и 268 test. CSV загружается по URL из ноутбука.

Числовые признаки масштабируются через StandardScaler, категории кодируются OneHotEncoder. Преобразования и регрессор объединены в Pipeline. Для Lasso подбираются четыре значения `alpha` через GridSearchCV (`cv=5`, scoring R²).

- OLS: **RMSE 5796,28, MAE 4181,19, R² 0,7836** на test.
- Lasso с мягкой IQR-фильтрацией: удалена одна train-строка, **RMSE 5806,32, R² 0,7828**.
- Lasso с фильтрацией по target: удалено 116 train-строк, **RMSE 7458,81, R² 0,6416**.

![Распределение target на train](../results/insurance_train_distribution.png)

![Предсказания OLS на test](../results/insurance_ols_actual_predicted.png)

Lasso не улучшил OLS. Удаление дорогих страховок ухудшило качество на исходном test. IQR-пороги вычислены до внутренней CV, а один test используется для сравнения нескольких сценариев. Ранний EDA охватывает весь набор; EDA перед обучением выполнен на train. Подвыборка для Shapiro-Wilk не имеет фиксированного seed.

## Used cars: ансамбли

[Ноутбук](../notebooks/used_cars_ensembles.ipynb) · [Метрики](../results/used_cars_metrics.csv)

Скачайте `used_cars_UK.csv` по ссылке из [источников](../PROVENANCE.md) и поместите в `data/used_cars_UK.csv` либо задайте `USED_CARS_CSV` до запуска Jupyter.

Подготовка включает заполнение пропусков, объединение редких категорий в `Other` и one-hot encoding. Train/test split: 80/20, seed 42. Для blending train дополнительно разделён на 2358 base-строк и 590 meta-строк, 37 признаков.

Результаты на test, RMSE / MAPE:

- Linear regression: **2258,92 / 35,71%**.
- kNN (k=5): **2291,32 / 27,81%**.
- Decision tree: **2269,76 / 31,83%**.
- Bagging: **1791,49 / 23,35%**.
- Bagging после подбора: **1794,78 / 25,19%**. Проверено 360 конфигураций на пяти folds; лучший CV RMSE **1497,26**.
- Blending: **1979,44 / 28,63%**.
- Stacking: **1984,19 / 26,61%**.
- Stacking с настроенным bagging: **1767,68 / 23,69%**.

Также выполнены подбор decision tree и сравнение метамоделей LinearRegression, RandomForestRegressor и GradientBoostingRegressor.

Импутация и частоты категорий рассчитаны до split. Для строгого сравнения нужна предобработка внутри folds и отдельная финальная выборка: текущий test использован для нескольких сравнений.

### Random forest для классификации региона

Реализация использует `DecisionTreeClassifier`, bootstrap и усреднение вероятностей. После удаления дубликата в insurance осталось 1337 строк; стратифицированное разбиение с seed 42: 1069 train и 268 test.

- sklearn RF: accuracy **0,3843**, macro-F1 **0,3787**.
- Собственный RF: accuracy **0,3955**, macro-F1 **0,3867**.

IQR-клиппинг выполнен до split. Разница на одном test мала; серия повторов и обработка редких классов в bootstrap не проверены.

## Insurance: кластеризация

[Ноутбук](../notebooks/insurance_clustering.ipynb)

1338 строк, шесть признаков: age, bmi, children, charges, sex и smoker. `region` исключён из признаков и используется для расчёта ARI. Кластеризация выполняется на всём наборе без train/test.

Собственный k-means сошёлся за 13 итераций. ARI относительно региона:

- Собственный k-means: **0,0019**.
- sklearn KMeans: **0,0019**.
- AgglomerativeClustering: **0,0027**.
- DBSCAN: **−0,0013**.
- KMeans после StandardScaler и PCA до двух компонент: **0,0025**.

![Регионы и найденные кластеры после PCA](../results/insurance_clustering_pca.png)

Кластеры почти не совпадают с регионами. Масштабирование и PCA применены вместе, их отдельный вклад не измерен. DBSCAN после scaling не запускался. Собственный k-means использует одну инициализацию и не обрабатывает пустые кластеры.

## Среда

Установите зависимости из `requirements.txt`. GPU не нужен. Grid search использует `n_jobs=-1`; точные версии исходных сред не зафиксированы. Данные и исходные Colab перечислены в [PROVENANCE.md](../PROVENANCE.md).
