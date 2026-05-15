# LENTA Pricetag Detection & OCR Pipeline

Детекция и распознавание ценников в магазине «ЛЕНТА» с каскадным пайплайном: цветовая детекция → трекинг → YOLO refine (опционально) → OCR → CSV.

## Архитектура

```
4K frame (3840×2160)
  │
  ├─ Stage 1: Цветовая детекция (utils/candidate_detector.py)
  │   HSV → inRange → морфология → контуры
  │   White + Red + Yellow маски → слияние красной+белой частей
  │   → 60-112 candidates
  │
  ├─ Stage 2: Трекер (utils/pricetag_tracker.py)
  │   Center-distance Hungarian matching
  │   NEW → Stage 3 (OCR)
  │   TRACKED → reuse cached OCR (быстро)
  │
  ├─ Stage 3: YOLO refine (опционально, при наличии RKNN)
  │   infer/infer_yolo_rknn.py → точный bbox
  │
  ├─ Stage 4: OCR + barcode + QR
  │   infer/infer_paddleocr.py → PaddleOCR (fallback EasyOCR)
  │   utils/ocr_enhancer.py → CLAHE + sharpen + upscale
  │   pyzbar → EAN-13 штрихкоды
  │   OpenCV QR → QR-коды
  │
  └─ Stage 5: CSV export (utils/csv_writer.py)
      Дедупликация spatial-кластеризацией
      → results/pricetags_YYYYMMDD.csv
```

## Структура проекта

```
├── config.yaml                    # Единый конфиг
├── main_pipeline_local.py         # Основной пайплайн
│
├── infer/
│   ├── infer_paddleocr.py         # OCR (PaddleOCR + EasyOCR fallback)
│   ├── infer_easyocr.py           # EasyOCR fallback
│   └── infer_yolo_rknn.py         # YOLOv12n RKNN инференс
│
├── utils/
│   ├── candidate_detector.py      # Цветовая детекция ценников
│   ├── cascade_pipeline.py        # Каскад: детекция → трекер → OCR
│   ├── pricetag_tracker.py        # Трекинг между кадрами
│   ├── csv_writer.py              # Экспорт в CSV (28 колонок)
│   ├── ocr_enhancer.py            # Препроцессинг для OCR
│   ├── config.py                  # Загрузчик config.yaml
│   ├── qr_parser.py               # Парсинг QR-кода
│   └── video_processor.py         # Пакетная обработка видео
│
├── models/yolov12n/
│   ├── yolov12n.pt                # Обученная YOLOv12n (mAP=0.994)
│   └── yolov12n_qat.pt            # QAT (mAP=0.995, для RKNN)
│
├── onnx2rknn/
│   └── yolov12n_rknn.py           # ONNX → RKNN int8 конвертация
│
├── train_yolo.py                  # Обучение YOLOv12n
├── train_qat.py                   # QAT дообучение
├── export_yolo.py                 # Экспорт в ONNX
├── prepare_data.py                # Подготовка данных (split + class remap)
│
└── data/yolov12n/
    ├── data.yaml                  # Конфиг для YOLO (2 класса)
    └── data_2class.yaml           # Альтернативный конфиг
```

## Установка

### Основное окружение (Python 3.13)

```bash
pip install -r requirements.txt
```

### PaddleOCR окружение (Python 3.12, опционально для RK3588)

```bash
# Создать отдельное окружение
python3.12 -m venv .venv_paddle
.venv_paddle\Scripts\activate   # Windows
# или source .venv_paddle/bin/activate  # Linux

pip install paddlepaddle-gpu paddleocr
```

## Обучение YOLO

```bash
# 1. Подготовка данных (train/val split + class remap)
python prepare_data.py

# 2. Обучение
python train_yolo.py                 # основное обучение
python train_qat.py --epochs 5       # QAT дообучение (для RKNN)

# 3. Экспорт
python export_yolo.py                 # → ONNX
# На RK3588: python onnx2rknn/yolov12n_rknn.py  # → RKNN int8
```

## Запуск пайплайна

```bash
# Камера
python main_pipeline_local.py

# Видео
python main_pipeline_local.py --video 26_12-20.mp4 --skip 3

# Пакетная обработка
python utils/video_processor.py . --skip 3
```

## Конфигурация

Все параметры в `config.yaml`:

```yaml
ocr:
  engine: paddleocr    # paddleocr | easyocr | auto (fallback)

tracker:
  center_dist_thresh: 2.0   # макс смещение центра (кратность размеру bbox)
  size_ratio_thresh: 0.25   # мин соотношение размеров bbox

cluster:
  dy_threshold: 80          # высота ряда полки
  dx_threshold: 600         # макс смещение ценника между кадрами
```

## Модели (Google Drive)

| Файл | Размер | Описание |
|------|:------:|----------|
| `yolo12n.pt` | 5.3 MB | Pretrained COCO (скачивается автоматически) |
| `models/yolov12n/yolov12n.pt` | 5.2 MB | Обученная модель (mAP=0.994) |
| `models/yolov12n/yolov12n_qat.pt` | 5.2 MB | QAT модель (mAP=0.995) |

**Ссылка на Google Drive:** <!-- TODO: добавить ссылку -->

## Требования к железу

| Компонент | Dev | Production (RK3588) |
|-----------|:---:|:------------------:|
| GPU | RTX 3060 12GB | NPU (RKNN int8) |
| YOLO | PyTorch FP16 | RKNN int8 (~30ms) |
| OCR | EasyOCR (CUDA) | PaddleOCR-RKNN (~50ms) |
| RAM | 16GB+ | 8GB+ |
