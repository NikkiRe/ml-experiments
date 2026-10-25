# Источники

Авторы исходных учебных работ: **Кирилл Слепцов и Никита Евстигнеев**. Решения выполнены совместно по учебным заданиям и шаблонам.

## Исходные работы

- [ЛР1: Pandas, EDA и линейная регрессия](https://colab.research.google.com/drive/1KybocoecK4DJ0N-ZgGAp4nyXhCvruxU8): `notebooks/insurance_regression.ipynb`.
- [ЛР5: ансамбли](https://colab.research.google.com/drive/1Ktvfu91n8q_e3CxQqLNBXuJEsw2d_al1): `notebooks/used_cars_ensembles.ipynb`.
- ЛР5, задача 6 (градиентный бустинг): в исходном Colab осталась черновиком без запусков. `notebooks/gradient_boosting.ipynb` выполнен отдельно после курса по формулировке задания и в совместную сдачу не входит.
- [ЛР6: кластеризация](https://colab.research.google.com/drive/1YfAy4jqcFla5DTf3BxSmhb62ZVj3IZZc): `notebooks/insurance_clustering.ipynb`.
- ЛР3: transfer learning с ResNet-50: `notebooks/resnet50_transfer_learning.ipynb`.
- ЛР2: residual CNN: `notebooks/sports_residual_cnn.ipynb`.

В репозитории оставлены выполненные разделы. Удалены требования сдачи, пустые ответы, незавершённые эксперименты, служебные Colab-выводы, встроенные фотографии и две схемы из шаблона. Пути адаптированы для локального запуска, пояснения сокращены. Код почищен без изменения вычислений: в `used_cars_ensembles.ipynb` импорты собраны в одной ячейке, убраны повторные определения функций и запасные ветки `try/except`; в `insurance_regression.ipynb` и sports-ноутбуке убраны лишние `print`, в sports-ноутбуке также неиспользуемая первая версия `calc_accuracy` и итоговая проверка из шаблона, класс сети переименован в `ResidualCNN`; метрики OLS в ячейке с Lasso печатаются из переменных. Строки, которые печатал удалённый код (в том числе `None` после `print(df.info())`), убраны и из сохранённых outputs. Числовые результаты и графики относятся к исходным запускам; обучение при оформлении репозитория не повторялось.

## Данные и модели

- Insurance: [CSV из stedy/Machine-Learning-with-R-datasets](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv); [описание на Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset/data).
- Used cars: [описание на Kaggle](https://www.kaggle.com/datasets/muhammadawaistayyab/used-cars-prices-in-uk); [CSV из учебного задания](https://drive.google.com/uc?id=1Cdek-lsS3-XiDFtD5PzgXNsy3gvd_Rd-).
- ImageNet: [ILSVRC/imagenet-1k на Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k), девять классов из `validation`, по 50 изображений.
- ResNet-50: веса `torchvision.models.ResNet50_Weights.IMAGENET1K_V2` из официального хранилища PyTorch.
- Sports: [архив sports.zip из учебного задания](https://drive.google.com/uc?id=1biITy7cv0-lo-bwZrFkyDCGsEOWCBx5W). Первоначальный автор и лицензия набора в исходнике не указаны.

Датасеты, фотографии и checkpoint не включены. При скачивании действуют условия источников. Для смешанного содержимого из решений и учебных шаблонов единая лицензия не назначена.

## Сохранённые результаты

CSV перенесены из сохранённых outputs без пересчёта, PNG — без редактирования. Номера ячеек — в ноутбуках этого репозитория, с нуля.

- `results/resnet50_seed42.csv`: ResNet-50, ячейка 27.
- `results/resnet50_four_splits.csv`: ResNet-50, ячейка 28.
- `results/insurance_regression_metrics.csv`: ЛР1, ячейки 17 и 20.
- `results/used_cars_metrics.csv`: ЛР5, ячейки 9, 12 и 14.
- `results/resnet50_training.png`: ResNet-50, ячейка 29.
- `results/sports_training.png`: sports CNN, ячейка 34.
- `results/insurance_train_distribution.png`: первый PNG ЛР1, ячейка 14.
- `results/insurance_ols_actual_predicted.png`: ЛР1, ячейка 17.
- `results/insurance_clustering_pca.png`: ЛР6, ячейка 8.

`results/boosting_regression_metrics.csv`, `results/boosting_classification_metrics.csv` и `results/boosting_shap_bmi.png` записывает `gradient_boosting.ipynb` при запуске. Сохранённый запуск: Python 3.12, scikit-learn 1.9.1, CatBoost 1.2.10, LightGBM 4.7.0, XGBoost 3.4.1, Optuna 5.0.0, Polars 2.1.0.
