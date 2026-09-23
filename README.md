# YOLOv11 Vehicle Object Detection

A computer vision project for detecting and localizing different types of vehicles in images using **YOLOv11**. The project covers model training, evaluation, inference, and a simple Streamlit interface for interactive predictions.

## Project Overview

The goal is to build an object detection model capable of identifying and localizing five vehicle classes:

* Bicycle
* Bus
* Car
* Motorbike
* Truck

The project uses a pretrained **YOLO11s** model and fine-tunes it on a custom vehicle detection dataset.

## Dataset

The dataset contains annotated vehicle images divided into training, validation, and test sets.

**Classes:**

* Bicycle
* Bus
* Car
* Motorbike
* Truck

The dataset is used in YOLO format with bounding-box annotations.

## Approach

The workflow includes:

1. Loading and inspecting the vehicle detection dataset.
2. Using a pretrained YOLO11s model.
3. Fine-tuning the model on the custom dataset.
4. Evaluating the trained model on the validation set.
5. Running inference on unseen test images.
6. Visualizing detected objects with bounding boxes and confidence scores.
7. Building a Streamlit application for interactive image detection.

## Model Training

The YOLO11s model was trained with:

* **Epochs:** 100
* **Image size:** 640 × 640
* **Batch size:** 16
* **Early stopping patience:** 10
* **Pretrained weights:** YOLO11s

Training was performed using GPU acceleration.

## Evaluation

The trained model was evaluated using standard object detection metrics.

| Metric    | Score |
| --------- | ----: |
| Precision | 0.906 |
| Recall    | 0.684 |
| mAP@50    | 0.779 |
| mAP@50:95 | 0.528 |

The validation set contained **74 images and 670 annotated vehicle instances**.

The model achieved its strongest detection performance on cars, while performance varied across the different vehicle classes.

## Inference

The trained model was also tested on unseen images to verify its practical detection behavior.

For each image, the model produces:

* Detected vehicle class
* Bounding box coordinates
* Confidence score

Example predictions are visualized directly in the notebook.

## Streamlit Application

A simple Streamlit interface is included for interactive inference.

The application allows users to:

* Upload an image.
* Use a demo image.
* Adjust the confidence threshold.
* Run YOLOv11 detection.
* View the original and annotated images.
* Inspect detected objects and their confidence scores.

### Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run streamlit_app.py
```

The application expects the trained model at:

```text
model/best.pt
```

## Project Structure

```text
Object-Detection-Project/
│
├── assets/
│   └── demo.png
│
├── model/
│   ├── best.pt
│   └── labels.txt
│
├── utils/
│   ├── detector.py
│   └── visualization.py
│
├── Object_Detection_YOLOV11 (1).ipynb
├── app.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Technologies

* Python
* YOLOv11
* Ultralytics
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Streamlit

## Key Takeaways

This project demonstrates an end-to-end object detection workflow, including:

* Transfer learning with a pretrained YOLO model.
* Object detection model training.
* Evaluation using Precision, Recall, and mAP.
* Visual analysis of predictions.
* Model inference.
* Basic deployment through Streamlit.

## Limitations

The model's performance varies between vehicle classes, and the validation set is relatively small. Further improvement could be achieved through a larger and more diverse dataset, additional data augmentation, and further hyperparameter tuning.

## Future Improvements

Possible improvements include:

* Expanding the training dataset.
* Testing different YOLO model sizes.
* Performing more extensive hyperparameter tuning.
* Evaluating performance on a more diverse set of real-world images.
* Adding video or webcam inference to the Streamlit application.
