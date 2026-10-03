# DeepFER — Project Report

## 1. Abstract
DeepFER is a deep-learning-based facial expression classification system for seven
categories: angry, disgust, fear, happy, neutral, sad, and surprise. The system
uses image augmentation, class weighting, a custom CNN baseline, and MobileNetV2
transfer learning with fine-tuning. Evaluation includes accuracy, precision,
recall, F1-score and a confusion matrix.

## 2. Problem Statement
Manual interpretation of facial expressions is subjective and difficult to scale.
DeepFER investigates whether a convolutional neural network can learn useful
visual patterns for classifying facial-expression labels from images.

## 3. Dataset
The supplied dataset contains 35,887 JPEG images:
- 28,821 training images
- 7,066 validation images

The training distribution is:
angry 3993, disgust 436, fear 4103, happy 7164, neutral 4982,
sad 4938, surprise 3205.

The validation distribution is:
angry 960, disgust 111, fear 1018, happy 1825, neutral 1216,
sad 1139, surprise 797.

The ZIP does not contain a separate test split.

## 4. Methodology
1. Extract and inspect the dataset.
2. Resize images to 96×96 RGB.
3. Apply horizontal flipping, rotation, zoom and translation to training images.
4. Calculate balanced class weights.
5. Train a custom CNN baseline.
6. Train a frozen MobileNetV2 transfer-learning model.
7. Fine-tune the final backbone layers using a low learning rate.
8. Evaluate on the supplied validation set.
9. Export the final Keras model.
10. Run real-time inference with OpenCV.

## 5. Evaluation
The notebook calculates accuracy and a full per-class classification report.
Because the dataset is imbalanced, macro F1 and class-level recall are important
alongside accuracy. Exact performance numbers should be taken from the executed
notebook rather than assumed in advance.

## 6. Deployment
The final model is saved as `deepfer_mobilenetv2.keras`. The supplied `app.py`
uses a webcam and OpenCV Haar-cascade face detection for a simple real-time demo.

## 7. Limitations
Facial expression labels do not directly reveal a person's internal emotional
state. Predictions may vary because of pose, lighting, occlusion, camera quality,
dataset bias and class imbalance. The current evaluation does not use an
independent test set.

## 8. Future Scope
- Add a truly independent test set.
- Compare EfficientNet, ResNet and other backbones.
- Add face alignment and stronger face detection.
- Perform systematic hyperparameter tuning.
- Quantize the model for edge deployment.
- Add calibration and confidence thresholds.
- Conduct subgroup/fairness analysis where appropriate.
- Build a Streamlit or web interface.
