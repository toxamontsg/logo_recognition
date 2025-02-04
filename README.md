# Few-Shot Logo Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Система для обнаружения и классификации логотипов с использованием Few-Shot Learning подхода.

## Особенности
- Детекция логотипов с помощью YOLOv8
- Классификация через CLIP с прототипами классов
- Поддержка форматов PL2K и LogoDet-3K
- Оценка качества (mAP, точность классификации)
- Гибкая настройка через конфигурационные файлы

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
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -P models/
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
from src.model import FewShotLogoRecognizer
from src.dataset import FewShotLogoDataset
from src.evaluator import LogoEvaluator

# Инициализация модели
recognizer = FewShotLogoRecognizer(
    brand_examples={'brand1': ['path1.jpg', ...]},
    detector_path='models/yolov8n.pt'
)

# Загрузка данных
dataset = FewShotLogoDataset(
    images_dir='data/processed/images/val',
    annotations_dir='data/processed/annotations/val',
    support_samples=5
)

# Оценка модели
evaluator = LogoEvaluator(recognizer, dataset)
detection_metrics = evaluator.evaluate_detection()
classification_acc = evaluator.evaluate_classification()
```

### Добавление новых брендов

```python
recognizer.add_new_brand(
    brand_name='new_brand',
    example_paths=['examples/new1.jpg', 'examples/new2.jpg']
)
```

## Результаты