# DeepFER: Facial Emotion Recognition Using Deep Learning

## Project Overview
DeepFER is a seven-class facial emotion recognition system using deep learning.
The project uses image augmentation, class weighting, a custom CNN baseline,
and transfer learning with MobileNetV2. It also includes evaluation metrics,
confusion matrices, training curves, model saving, single-image prediction,
and a real-time webcam application.

## Dataset
The supplied dataset contains:
- 35,887 JPEG images
- Training: 28,821
- Validation: 7,066
- Classes: angry, disgust, fear, happy, neutral, sad, surprise

Training class counts:
- angry: 3,993
- disgust: 436
- fear: 4,103
- happy: 7,164
- neutral: 4,982
- sad: 4,938
- surprise: 3,205

Validation class counts:
- angry: 960
- disgust: 111
- fear: 1,018
- happy: 1,825
- neutral: 1,216
- sad: 1,139
- surprise: 797

There is no separate test directory in the supplied ZIP. The notebook therefore
uses the supplied validation directory for validation/evaluation and explicitly
documents this limitation. For a publication-quality result, a held-out test
set should be created before final training.

## Models
1. Custom CNN baseline
2. MobileNetV2 transfer-learning model
3. Optional fine-tuning of the last layers of MobileNetV2

## Evaluation
Accuracy, precision, recall, F1-score, confusion matrix, per-class performance,
and training/validation curves are generated.

## Real-Time Demo
`app.py` uses OpenCV Haar-cascade face detection and the trained Keras model.
It displays the predicted emotion and confidence for detected faces.

## Colab
Upload the supplied ZIP to Google Colab, run the notebook cells in order, and
set DATASET_ZIP to the uploaded ZIP path.

## Important Interpretation Note
Facial-expression classifiers predict a dataset label from visual patterns.
They should not be treated as reliable measurements of a person's internal
mental state, mental health, intent, or personality.
