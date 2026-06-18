# LENTA Pricetag Detection & OCR Pipeline

Детекция и распознавание ценников в магазине «ЛЕНТА» с каскадным пайплайном:
цветовая детекция → трекинг (Kalman + Hungarian) → SAHI/мульти-скейл YOLO refine → OCR → CSV.

## Архитектура

```
4K frame (3840×2160)
  │
  ├─ Stage 1: Цветовая детекция (utils/candidate_detector.py)
  │   HSV → inRange → морфология → контуры
  │   White + Red + Yellow маски → слияние красной+белой частей
  │   Адаптивная HSV-калибровка (auto_calibrate_hsv)
  │
  ├─ Stage 2: Трекер (utils/pricetag_tracker.py)
  │   Kalman filter + Hungarian matching (scipy)
  │   NEW → Stage 3 (YOLO)
  │   TRACKED → reuse cached OCR (быстро)
  │
  ├─ Stage 3: YOLO refine (utils/tiled_inference.py)
  │   SAHI (Slicing Aided Hyper Inference) — тайлы 640×640, stride 320
  │   Мульти-скейл детекция (0.75×, 1.0×, 1.25×)
  │   WBF (Weighted Boxes Fusion) слияние дублей
  │   Адаптивный порог confidence (adaptive_conf)
  │   infer/infer_yolo_rknn.py → RKNN int8 на RK3588
  │
  ├─ Stage 4: OCR + barcode + QR
  │   infer/infer_easyocr.py → EasyOCR
  │   utils/ocr_enhancer.py → CLAHE + sharpen + upscale
  │   Мульти-pass OCR (otsu, adaptive, denoise, deskew)
  │   pyzbar → EAN-13 штрихкоды
  │   utils/qr_parser.py → QR-коды (JSON / key=value / pipe)
  │   utils/product_db.py → поиск товара по штрихкоду из CSV-каталога
  │
  └─ Stage 5: CSV export (utils/csv_writer.py)
      Буферизация, дедупликация, валидация цен/штрихкодов/bbox
      → results/pricetags_YYYYMMDD.csv
```

## Структура проекта

```
├── config.yaml                     # Единый конфиг (augmentations, sahi, adaptive_*)
│
├── infer/
│   ├── infer_easyocr.py            # EasyOCR распознавание
│   └── infer_yolo_rknn.py          # YOLOv12n RKNN инференс
│
├── utils/
│   ├── __init__.py
│   ├── candidate_detector.py       # Цветовая детекция ценников + адаптивная HSV
│   ├── cascade_pipeline.py         # Каскад: детекция → трекер → YOLO → OCR
│   ├── pricetag_tracker.py         # Kalman + Hungarian трекинг
│   ├── tiled_inference.py          # SAHI + мульти-скейл + WBF слияние
│   ├── wbf.py                      # Weighted Boxes Fusion
│   ├── iou.py                      # IoU вычисление
│   ├── csv_writer.py               # Экспорт в CSV
│   ├── ocr_enhancer.py             # CLAHE + sharpen + мульти-pass OCR
│   ├── qr_parser.py                # Парсинг QR-кода
│   ├── product_db.py               # Поиск товара по штрихкоду
│   └── config.py                   # Загрузчик config.yaml
│
├── train/
│   ├── run_train_yolo.py           # Обучение YOLO (полный цикл с аугментациями)
│   └── eval_trained_model.py       # Визуальная оценка обученной модели
│
├── data/yolov12n/
│   ├── data.yaml                   # Конфиг YOLO (1 класс: pricetag)
│   ├── data_2class.yaml             # Конфиг YOLO (2 класса)
│   ├── data_full.yaml              # Конфиг YOLO (7 классов, внутренние элементы)
│   ├── merge_tiled.py              # Слияние tiled-данных
│   ├── merge_datasets.py          # Слияние нескольких датасетов
│   ├── filter_small_boxes.py       # Фильтрация маленьких bbox
│   ├── cleanup_dataset.py         # Очистка датасета
│   └── stats_tiled.py             # Статистика tiled-данных
│
├── eval_compare.py                 # Сравнение моделей на тестовом кадре
├── eval_visualize.py              # Визуализация результатов детекции
├── run_test_10s.py                # Быстрый тест пайплайна (10 секунд)
│
├── models/yolov12n/
│   ├── yolov12n.pt                # Обученная YOLOv12n
│   └── yolov12n_retrained.pt      # Дообученная на tiled-данных
│
├── onnx2rknn/
│   └── yolov12n_rknn.py           # ONNX → RKNN int8 конвертация
│
└── tests/
    ├── conftest.py                 # Mock config.yaml для тестов
    ├── test_csv_writer.py         # 53 теста — CSV-парсинг
    ├── test_tracker.py             # 25 тестов — трекинг
    ├── test_candidate_detector.py  # 10 тестов — геометрия кропов + IoU
    └── test_qr_parser.py           # 9 тестов — QR-парсинг
```

## Установка

```bash
pip install -r requirements.txt
```

## Обучение YOLO

```bash
# Обучение на tiled-данных с аугментациями (настроены в config.yaml)
python train/run_train_yolo.py

# Визуальная оценка на тестовом кадре
python train/eval_trained_model.py
```

### Аугментации (config.yaml → augmentations)

| Параметр | Значение | Описание |
|----------|:--------:|----------|
| mosaic | 1.0 | Mosaic 4-изображений (вкл) |
| mixup | 0.2 | Mixup-блендинг |
| copy_paste | 0.3 | Копипаст объектов |
| erasing | 0.4 | Random erasing |
| fliplr | 0.5 | Горизонтальный флип |
| scale | 0.5 | Масштабирование |
| close_mosaic | 10 | Выкл mosaic за 10 эпох до конца |

## Запуск

```bash
# Быстрый тест пайплайна (10 секунд видео)
python run_test_10s.py

# Сравнение моделей
python eval_compare.py
```

## Конфигурация

Ключевые секции `config.yaml`:

```yaml
augmentations:        # Параметры аугментации YOLO
sahi:                 # SAHI — тайлинг (tile_size: 640, stride: 320)
multiscale:           # Мульти-скейл детекция (0.75×, 1.0×, 1.25×)
adaptive_hsv:         # Авто-калибровка HSV порогов по статистике кадра
adaptive_conf:        # Адаптивный порог confidence по плотности детекций
temporal_boost:       # Временное усиление confidence для отслеженных объектов
cascade:
  adaptive_yolo: true           # Пропуск YOLO при уверенном трекинге
  yolo_skip_frames: 5           # Пропуск YOLO при трекинге
  yolo_force_interval: 15       # Принудительный YOLO каждые N кадров
tracker:
  center_dist_thresh: 10.0      # Порог расстояния центров
  max_frames_missed: 15         # Макс пропущенных кадров до удаления
```

## Модели (Google Drive)

| Файл | Размер | Описание |
|------|:------:|----------|
| `yolo12n.pt` | 5.3 MB | Pretrained COCO (скачивается автоматически) |
| `models/yolov12n/yolov12n.pt` | 5.2 MB | Обученная модель |
| `models/yolov12n/yolov12n_retrained.pt` | 5.2 MB | Дообученная на tiled-данных |

**Ссылка на Google Drive:** <!-- TODO: добавить ссылку -->

## Требования к железу

| Компонент | Dev | Production (RK3588) |
|-----------|:---:|:------------------:|
| GPU | RTX 3060 12GB | NPU (RKNN int8) |
| YOLO | PyTorch FP16 | RKNN int8 (~30ms) |
| OCR | EasyOCR (CUDA) | PaddleOCR-RKNN (~50ms) |
| RAM | 16GB+ | 8GB+ |

## Тесты

```bash
# Запуск всех 97 тестов
pytest tests/ -v
```
