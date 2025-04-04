# YOLOv8n Traffic Light Detection (LISA Dataset) - BOROSA Hackathon

## 1. Project Overview

This project was developed as part of the **Bosch Road Safety Hackathon (BOROSA) 2025 - Prelim Hackathon**. The specific problem statement addressed was **"Code Red | Traffic Violation Detection System"**, focusing on developing an intelligent system to recognize Red Signal Jumps.

The core requirement was to create a system capable of detecting traffic lights and their states (Red, Yellow, Green) using computer vision, suitable for deployment on an onboard component (like a microcontroller-based ECU, e.g., Raspberry Pi or Jetson Nano).

This repository documents the first crucial step: **training a lightweight, state-of-the-art object detection model (YOLOv8n) specifically for traffic light detection and color recognition using the LISA dataset on the Kaggle platform.**

## 2. Dataset

*   **Dataset Used:** [LISA (Laboratory for Intelligent & Safe Automobiles) Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)
*   **Source:** Downloaded via Kaggle Datasets (`mbornoe/lisa-traffic-light-dataset`).
*   **Content:** Contains video sequences and extracted frames from driving scenarios in the US, annotated with traffic light bounding boxes and states.
*   **Annotation Format:** The dataset version used provides annotations in **CSV format** (`frameAnnotationsBOX.csv`), nested within subdirectories corresponding to different sequences (e.g., `daySequence1`, `nightSequence1`). This differs from some other formats (like COCO JSON or a single top-level CSV) and required specific parsing logic.

## 3. Environment & Setup

*   **Platform:** Kaggle Notebooks
*   **Hardware:** 2 x NVIDIA T4 GPUs (Initially attempted, final successful training used **1 x T4 GPU** due to DDP issues). *Note: Final validation run shown in output used a P100.*
*   **Key Software:**
    *   Python 3.10
    *   PyTorch 2.5.1+cu121
    *   Ultralytics YOLO 8.3.101
    *   Pandas
    *   OpenCV (opencv-python-headless)
    *   Scikit-learn (for data splitting)
    *   PyYAML

## 4. Data Preparation

This was a critical phase involving several steps and challenges:

1.  **Dataset Download & Extraction:** The dataset was downloaded and unzipped within the Kaggle environment.
2.  **Annotation File Discovery:**
    *   **Challenge:** Initial assumption of a single `allAnnotations.csv` file was incorrect. The annotations were found to be split into multiple `frameAnnotationsBOX.csv` files, nested like `/kaggle/working/lisa_traffic_light/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv`.
    *   **Solution:** Used Python's `glob` module to recursively find all relevant `frameAnnotationsBOX.csv` files within the nested structure.
3.  **Annotation Loading & Combination:** Loaded each discovered CSV into a Pandas DataFrame and concatenated them into a single DataFrame for unified processing. The sequence name (e.g., `daySequence1`) was extracted from the CSV file path and stored for later use.
4.  **Image Path Construction:**
    *   **Challenge:** The `Filename` column in the CSVs (e.g., `dayTest/daySequence1--00111.jpg`) did not directly map to the actual image file locations (e.g., `/kaggle/working/lisa_traffic_light/daySequence1/daySequence1/frames/daySequence1--00111.jpg`).
    *   **Solution:** Developed logic to extract the base image filename from the CSV and construct the correct, full image path using the known directory structure (`BASE_DIR/sequence_name/sequence_name/frames/actual_image_filename`).
5.  **Annotation Format Conversion (CSV to YOLO .txt):**
    *   Iterated through the combined annotations.
    *   For each annotation, read the corresponding image *once* to get its dimensions (width, height) for normalization.
    *   Mapped the annotation tag (e.g., 'go', 'stopLeft') to predefined class IDs using the `CLASS_MAPPING` dictionary (see below).
    *   Converted the bounding box coordinates (Upper Left X/Y, Lower Right X/Y) from the CSV into the YOLO format: `<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`.
    *   Saved the YOLO-formatted annotations into individual `.txt` files (one per image) in a temporary `raw_labels` directory.
6.  **Class Mapping Used:**
    ```python
    CLASS_MAPPING = {
        'stop': 0,
        'stopleft': 0,      # Mapped directional stop to base stop
        'warning': 1,
        'warningleft': 1,   # Mapped directional warning to base warning
        'go': 2,
        'goleft': 2,        # Mapped directional go to base go
        'goforward': 2,     # Mapped forward go to base go
    }
    # Corresponds to 3 classes: 0: stop, 1: warning, 2: go
    ```
7.  **Train/Validation Split:**
    *   Used `sklearn.model_selection.train_test_split` to split the list of processed image basenames (those with valid images and annotations) into training (80%) and validation (20%) sets.
    *   Copied the corresponding image files (from their original locations) and the generated `.txt` label files (from `raw_labels`) into the final YOLO directory structure:
        ```
        lisa_yolo_format/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml
        ```
8.  **`data.yaml` Creation:** Generated the `data.yaml` file required by YOLOv8, specifying the paths to the train/val image directories and the class names in the correct order based on the `CLASS_MAPPING`.

## 5. Training Process

1.  **Model Selection:** Chose **YOLOv8n** (nano version) due to its balance of speed and accuracy, making it suitable for the target edge devices. Pretrained weights from COCO were used for transfer learning.
2.  **Training Library:** Utilized the `ultralytics` Python library.
3.  **Hyperparameters:**
    *   Epochs: 75
    *   Batch Size: 32
    *   Image Size: 640x640
    *   Optimizer: AdamW
    *   Learning Rate: 0.001 (initial)
    *   Patience (Early Stopping): 15 epochs
    *   Workers (Dataloader): 2
4.  **Multi-GPU Challenge:**
    *   **Issue:** Attempting to train using both available T4 GPUs (`device='0,1'`) resulted in the training process hanging indefinitely after the Distributed Data Parallel (DDP) initialization logs were printed. No epoch progress was shown for hours.
    *   **Troubleshooting:** Stopped the multi-GPU attempt and forced training onto a single GPU (`device='0'`).
    *   **Resolution:** Single-GPU training proceeded correctly and showed consistent epoch progress. The decision was made to continue with the reliable single-GPU training to ensure a completed model, despite the longer training time.
5.  **Training Duration:** The successful single-GPU training run for 75 epochs completed in approximately **3.973 hours**.
6.  **Monitoring:** Training progress (loss, mAP, etc.) was monitored via the standard output printed by the `ultralytics` library in the Kaggle notebook cell.

## 6. Results & Evaluation

The model was evaluated on the validation set after each epoch. The final metrics reported after 75 epochs (using the `best.pt` weights) were:

| Metric         | Value  | Description                                                                 |
| -------------- | ------ | --------------------------------------------------------------------------- |
| **mAP50(B)**   | 0.936  | Mean Average Precision @ IoU=0.50 (Good overlap threshold)                  |
| **mAP50-95(B)**| 0.640  | Mean Average Precision @ IoU=0.50:0.95 (COCO primary metric, stricter)      |
| Precision(B)   | 0.929  | Of all detected lights, ~93% were correctly classified.                     |
| Recall(B)      | 0.875  | The model found ~87.5% of all actual traffic lights in the validation set. |

*(Note: (B) indicates metrics calculated for the bounding box detection task).*

The training process also generated visualization plots saved in the results directory: `

## Visualization of Results

The following plots were generated by the `ultralytics` library during training and saved in the results directory (`/kaggle/working/LISA_Traffic_Light_Detection/yolov8n_e75_bs32_20250404_100024/`).

**a) Training & Validation Metrics (`![results](https://github.com/user-attachments/assets/cf9e2426-5984-4472-a25c-65687cad01a8)
`)**

This plot visualizes the trends mentioned above, showing the progression of losses and mAP metrics over the 75 epochs. It helps confirm convergence and identify potential overfitting (which appears minimal here).

b) Confusion Matrix ![confusion_matrix](https://github.com/user-attachments/assets/69d3f871-0696-4fe0-8e39-42d5726a1f6b)

c) Precision-Recall Curve ![PR_curve](https://github.com/user-attachments/assets/48d5ffce-fa58-4cd4-9a98-a61a2e5b3eaf)

d) P_curve.png: Precision vs. Confidence threshold. ![P_curve](https://github.com/user-attachments/assets/bef32414-58f3-4325-922d-54f71f2fb3ff)

e) R_curve.png: Recall vs. Confidence threshold. ![R_curve](https://github.com/user-attachments/assets/20bbc411-0215-4ac9-acb8-84167adeef17)

f) F1_curve.png: F1 Score vs. Confidence threshold. ![F1_curve](https://github.com/user-attachments/assets/9dc83e77-9efd-4f13-9486-aa1ebf327a3c)

g) Sample Validation Predictions : These images provide a qualitative look at the model's performance. They show a batch of images from the validation set with the ground truth labels and the model's predictions overlaid. This helps visualize how well the model detects lights and how accurate the bounding boxes are in practice.
(val_batch*_labels.jpg ![val_batch0_labels](https://github.com/user-attachments/assets/c559eea2-cc0d-41dc-b325-0ccba9bb0204)
 & val_batch*_pred.jpg![val_batch0_pred](https://github.com/user-attachments/assets/1af77f78-66df-4ee2-8a4e-ed28392e51f4)) 
 
(val_batch1_labels.jpg ![val_batch1_labels](https://github.com/user-attachments/assets/814eda49-8892-483c-9ca4-6c2057ef2634)
, val_batch1_pred.jpg ![val_batch1_pred](https://github.com/user-attachments/assets/127886fa-266a-446b-8b02-4ef2577de45f))


## 7. Analysis & Interpretation

*   **Overall Performance:** The achieved metrics, particularly **mAP50 of 0.936**, indicate a **very good** performance for detecting and classifying the basic states (stop, warning, go) of traffic lights under typical conditions at a reasonable bounding box overlap.
*   **Localization Accuracy:** The **mAP50-95 of 0.640** is **decent/good** and suggests that while the model generally finds the lights, the tightness of the predicted bounding boxes might vary (which is expected for a small, fast model like YOLOv8n).
*   **Precision vs. Recall:** The high precision (0.929) is excellent, meaning the model rarely predicts a traffic light where there isn't one or misclassifies its state badly. The good recall (0.875) means it finds most of the lights, though it might miss some, potentially smaller, more distant, or partially occluded ones.
*   **Suitability for Edge Devices:** The YOLOv8n model, combined with these strong performance metrics, demonstrates its potential suitability for deployment on resource-constrained devices like Jetson Nano or Raspberry Pi, aligning with the hackathon's requirements. The small file size (6.2MB stripped) is also advantageous.
*   **Class Mapping Impact:** The decision to map directional signals (e.g., `stopLeft`) to their base classes (e.g., `stop`) simplified the problem to 3 main classes. If directional distinction were critical, a different mapping and potentially more data/training would be needed.

## 8. Usage & Next Steps

1.  **Trained Weights:** The best performing model weights are saved in the `weights/` subfolder of the results directory:
    `/kaggle/working/LISA_Traffic_Light_Detection/yolov8n_e75_bs32_20250404_100024/weights/best.pt`
    *(Note: The timestamp in the path might differ slightly on re-runs).*
2.  **Inference:** This `best.pt` file can be loaded using the `ultralytics` library for running inference on new images or video streams:
    ```python
    from ultralytics import YOLO

    # Load the trained model
    model = YOLO('/path/to/your/best.pt')

    # Run inference
    results = model.predict(source='your_image.jpg' or 'your_video.mp4', save=True)
    # Process results (bounding boxes, classes, confidences)
    # print(results[0].boxes)
    ```
3.  **Deployment:** For optimal performance on edge devices (Jetson/Pi):
    *   **Export:** Convert the PyTorch `.pt` model to an optimized format like ONNX or TensorRT Engine using the `model.export()` function from `ultralytics`.
    *   **Runtime:** Use appropriate inference runtimes (ONNX Runtime, TensorFlow Lite, NVIDIA TensorRT) on the target device.
4.  **Hackathon Integration:** This trained model provides the core traffic light detection capability. Further steps for the "Code Red" system would involve:
    *   Integrating inference into a real-time video processing pipeline.
    *   Tracking detected lights across frames.
    *   Implementing logic to determine if the vehicle crossed the stop line *while* the relevant traffic light was red (Red Signal Jump detection).

## 9. Future Work & Improvements

*   **Data Augmentation:** Apply more aggressive augmentation techniques during training to improve robustness to varied lighting, weather, and partial occlusions.
*   **Hyperparameter Tuning:** Experiment with learning rate schedules, optimizer settings, batch size (if memory allows), and augmentation parameters.
*   **Model Size:** If edge device constraints permit, explore slightly larger models (e.g., YOLOv8s) for potentially higher mAP50-95 (better localization).
*   **Dataset Expansion:** Incorporate more diverse data, especially challenging scenarios (heavy rain, fog, unusual light configurations, more night scenes).
*   **Investigate DDP:** If faster multi-GPU training is desired later, further investigate the DDP hanging issue on Kaggle (try different PyTorch/Ultralytics versions, environment variables, etc.).

## 10. Acknowledgements

*   The creators of the LISA Traffic Light Dataset.
*   The Ultralytics team for the YOLOv8 implementation and library.
*   Kaggle for the cloud computing platform.
*   Bosch Global Software Technologies for organizing the BOROSA Hackathon.

---

*(Optional: Add a License section, e.g., MIT License)*
