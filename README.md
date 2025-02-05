# Few-Shot Logo Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Система для обнаружения и классификации логотипов с использованием Few-Shot Learning подхода.

## Особенности
- Детекция логотипов с помощью YOLOv5
- Классификация через CLIP с прототипами классов
- Двтвсет LogoDet-3K


## Установка
1. Клонировать репозиторий:
```bash
git clone https://github.com/yourusername/logo-recognition.git
cd logo-recognition
```

2. Установить зависимости:
```bash
pip install -r requirements.txt
```

3. Скачать веса YOLO:
```bash
wget -P models https://github.com/toxamontsg/yolo_models/raw/refs/heads/main/yolo5s_logo.pt
```

4. Скачать dataset:
```bash
wget -P data/raw 123.57.42.89/Dataset_ict/LogoDet-3K.zip
```

## Использование

### Подготовка данных

```python
from src.data_processing import DataSplit

processor = DataSplit(dataset_type='PL2K')
processor.load_data(
    input_path='data/raw/PL2K',
    output_path='data/processed'
)
```

### Подготовка данных

```python
from src.data_processing import DataSplit
from src.model import FewShotLogoRecognizer
from src.dataset import FewShotLogoDataset

# разделение данных на train, val, test 
data_split = DataSplit(dataset_type='LogoDet-3K')
data_split.load_data(
    input_path='.data/raw/LogoDet-3K',
    output_path='data/processed'
)


# Загрузка данных
dataset = FewShotLogoDataset(
    images_dir='data/processed/images/val',
    annotations_dir='data/processed/annotations/val',
    support_samples=5
)

# Инициализация модели
support_set = dataset.get_support_set()
query_set = dataset.get_query_set()


brand_examples = {}
for img_path, brand in support_set:
    if brand not in brand_examples:
        brand_examples[brand] = []
    brand_examples[brand].append(img_path)

recognizer = FewShotLogoRecognizer(
    brand_examples=brand_examples,
    model_name="ViT-B/32",
    detector_path="../models/yolo5s_logo.pt"
    
)

```

## Описание


### Цель системы

Реализация end-to-end решения для обнаружения и классификации логотипов брендов на изображениях с использованием few-shot обучения. Система должна:

1. Детектировать логотипы на изображениях.
2. Классифицировать их по брендам, даже при ограниченном количестве примеров.


### Архитектура системы

#### 1. Детекция логотипов

**Модель**: YOLO (например, YOLOv5) для обнаружения bounding box'ов.

**Особенности**:
- Использует предобученную модель для локализации логотипов.
- Настраиваемые параметры: `conf` (порог уверенности), `iou` (порог пересечения).
- Обработка ошибок: проверка на пустые предсказания, фильтрация некорректных обрезков.

#### 2. Классификация логотипов

**Модель**: CLIP (Contrastive Language-Image Pretraining) от OpenAI.

**Few-shot подход**:
- **Прототипы брендов**: Усредненные эмбеддинги изображений.
- **Классификация**: Сравнение эмбеддинга обнаруженного логотипа с прототипами через Евклидово расстояние.

**Особенности**:
- **Нормализация эмбеддингов**: Для стабильности вычислений.
- **Коэффициент и порог**: Настройка чувствительности сигмоидной функции для преобразования расстояния в вероятность.
- **Дообучение**: Возможность дообучения модели CLIP на специфическом наборе данных логотипов.
- **В production**: Использование векторной базы данных для хранения эмбеддингов изображений прототипов.
- **Few-shot обучение**: Для классификации требуется всего 5-6 примеров на бренд.
- **Эмбеддинги CLIP**: Позволяют работать с малым количеством данных.
- Для хорошего результата можно сделать дообучение модели CLIP.

#### 3. Валидация системы

**Метрики детекции**: mAP (mean Average Precision), IoU.

**Метрики классификации**: Accuracy, Precision, Recall, F1, ROC-AUC.

**Бинарная классификация**: Возможность проверки принадлежности к конкретному бренду с настраиваемым порогом.