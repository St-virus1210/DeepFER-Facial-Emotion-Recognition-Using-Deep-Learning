# DeepFER — Facial Emotion Recognition Using Deep Learning

DeepFER is an end-to-end facial-expression classification project for seven expression categories:

`angry, disgust, fear, happy, neutral, sad, surprise`

It includes:
- Dataset loading from directory structure
- Data augmentation
- Custom CNN training
- Transfer learning with MobileNetV2
- Evaluation with accuracy, precision, recall, F1-score and confusion matrix
- Single-image prediction
- Real-time webcam detection with OpenCV
- Streamlit web interface

> Important: the model classifies visible facial-expression categories. It should not be treated as a definitive measurement of a person's internal emotional or mental state.

## 1. Environment

Python 3.10–3.12 is recommended.

Create an environment:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Dataset

Place your dataset in:

```text
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── validation/
│   └── same seven folders
└── test/
    └── same seven folders
```

The loader also accepts `val` instead of `validation` when using the command-line scripts.

Recommended public datasets include FER-style datasets with the same seven categories. Make sure your dataset license permits your intended use.

## 3. Train the custom CNN

```bash
python -m src.train_cnn
```

Optional:

```bash
python -m src.train_cnn --epochs 30 --batch-size 64
```

The model is saved as:

```text
models/deepfer_cnn.keras
```

## 4. Train MobileNetV2 transfer-learning model

```bash
python -m src.train_transfer
```

The model is saved as:

```text
models/deepfer_mobilenetv2.keras
```

## 5. Evaluate a model

```bash
python -m src.evaluate --model models/deepfer_mobilenetv2.keras --split test
```

Outputs are placed in `results/`.

## 6. Predict an image

```bash
python -m src.predict path/to/face.jpg
```

Or specify a model:

```bash
python -m src.predict path/to/face.jpg --model models/deepfer_mobilenetv2.keras
```

## 7. Real-time webcam

```bash
python -m src.realtime --model models/deepfer_mobilenetv2.keras
```

Press `q` to quit.

## 8. Streamlit application

```bash
streamlit run app/app.py
```

The application supports image upload and webcam input where supported by the Streamlit installation.

## Project workflow

```text
Image / Video
     ↓
Face Detection
     ↓
Face Crop
     ↓
Resize + Normalization
     ↓
CNN / MobileNetV2
     ↓
7-class Softmax
     ↓
Expression + Confidence
```

## Notes

- Training requires a dataset; this package intentionally does not redistribute third-party image datasets.
- If the dataset is small, transfer learning is generally useful, but performance depends strongly on dataset quality, class balance, subject diversity and evaluation methodology.
- For a fair academic comparison, keep the test set fixed and do not tune hyperparameters against it.
