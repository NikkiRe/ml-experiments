# Источники

Авторы исходных учебных работ: **Кирилл Слепцов и Никита Евстигнеев**. Решения выполнены совместно по учебным заданиям и шаблонам.

## Исходные работы

- [ЛР1: Pandas, EDA и линейная регрессия](https://colab.research.google.com/drive/1KybocoecK4DJ0N-ZgGAp4nyXhCvruxU8): `notebooks/insurance_regression.ipynb`. SHA-256 исходника: `3ba4f4af7cc30042369ec0834d30d51adf860a62187baee7dce7155ff39ed61f`.
- [ЛР5: ансамбли](https://colab.research.google.com/drive/1Ktvfu91n8q_e3CxQqLNBXuJEsw2d_al1): `notebooks/used_cars_ensembles.ipynb`. SHA-256: `7493da669447e2441e4aa25b7a2ee8253718f5f28a8596b2b2bc37e6df67b21b`.
- ЛР5, задача 6 (градиентный бустинг): в исходном Colab осталась черновиком без запусков. `notebooks/gradient_boosting.ipynb` выполнен отдельно после курса по формулировке задания и в совместную сдачу не входит.
- [ЛР6: кластеризация](https://colab.research.google.com/drive/1YfAy4jqcFla5DTf3BxSmhb62ZVj3IZZc): `notebooks/insurance_clustering.ipynb`. SHA-256: `051993a8631d16b3415d9eae1c958c04b6ac8d73a927ff66ce139ab9115551e0`.
- ЛР3: transfer learning с ResNet-50, `notebooks/resnet50_transfer_learning.ipynb`. SHA-256 исходника: `22db9742267cf57d2797aa20672818d410c2a3dcff6245279a09d706f8e51aa0`.
- ЛР2: residual CNN, исходный файл `TNC_Lab2_с_подробными_пояснениями.ipynb`, `notebooks/sports_residual_cnn.ipynb`. SHA-256: `b335d443dddc3170b002289b84caf206ebda6b26cf847d135321f58b59f7b441`.

В репозитории оставлены выполненные разделы. Удалены требования сдачи, пустые ответы, незавершённые эксперименты, служебные Colab-выводы, встроенные фотографии и две схемы из шаблона. Пути адаптированы для локального запуска, пояснения сокращены. Числовые результаты и графики относятся к исходным запускам; обучение при оформлении репозитория не повторялось.

## Данные и модели

- Insurance: [CSV из stedy/Machine-Learning-with-R-datasets](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv); [описание на Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset/data).
- Used cars: [описание на Kaggle](https://www.kaggle.com/datasets/muhammadawaistayyab/used-cars-prices-in-uk); [CSV из учебного задания](https://drive.google.com/uc?id=1Cdek-lsS3-XiDFtD5PzgXNsy3gvd_Rd-).
- ImageNet: [ILSVRC/imagenet-1k на Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k), девять классов из `validation`, по 50 изображений.
- ResNet-50: веса `torchvision.models.ResNet50_Weights.IMAGENET1K_V2` из официального хранилища PyTorch.
- Sports: [архив sports.zip из учебного задания](https://drive.google.com/uc?id=1biITy7cv0-lo-bwZrFkyDCGsEOWCBx5W). Первоначальный автор и лицензия набора в исходнике не указаны.

Датасеты, фотографии и checkpoint не включены. При скачивании действуют условия источников. Для смешанного содержимого из решений и учебных шаблонов единая лицензия не назначена.

## Сохранённые результаты

CSV извлечены из исходных outputs без пересчёта, PNG сохранены без редактирования. Индексы ячеек ниже начинаются с нуля.

- `results/resnet50_seed42.csv`: ResNet-50, ячейка 27.
- `results/resnet50_four_splits.csv`: ResNet-50, ячейка 28.
- `results/insurance_regression_metrics.csv`: ЛР1, ячейки 26 и 29; повторные константы OLS не продублированы.
- `results/used_cars_metrics.csv`: ЛР5, ячейки 13, 30 и 33.
- `results/resnet50_training.png`: ResNet-50, ячейка 29.
- `results/sports_training.png`: sports CNN, ячейка 62.
- `results/insurance_train_distribution.png`: первый PNG ЛР1, ячейка 23.
- `results/insurance_ols_actual_predicted.png`: ЛР1, ячейка 26.
- `results/insurance_clustering_pca.png`: ЛР6, ячейка 10.

`results/boosting_regression_metrics.csv`, `results/boosting_classification_metrics.csv` и `results/boosting_shap_bmi.png` записывает `gradient_boosting.ipynb` при запуске. Сохранённый запуск: Python 3.12, scikit-learn 1.9.1, CatBoost 1.2.10, LightGBM 4.7.0, XGBoost 3.4.1, Optuna 5.0.0, Polars 2.1.0.
