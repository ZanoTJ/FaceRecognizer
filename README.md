
# Real-Time Face Detection and Recognition

A deep learning project that detects faces using **YOLOv8** and recognizes identities using a **transfer-learning CNN based on InceptionV3**. The notebooks are designed for **Google Colab** and support recognition from images, webcam snapshots, and a live webcam stream with labeled bounding boxes.

## Features

- Detect multiple faces in an image or webcam frame using YOLOv8 face detection.
- Recognize trained identities with an InceptionV3-based convolutional neural network.
- Display bounding boxes, predicted names, and confidence scores during live webcam detection.
- Predict `unknown` for low-confidence recognition results (the live stream uses a confidence threshold of `0.5`).
- Train using image augmentation, early stopping, learning-rate reduction, and model checkpoints.
- Evaluate predictions using accuracy, weighted F1, classification reports, and a confusion matrix.

## Tech Stack

**Language/environment:** Python, Jupyter Notebook, Google Colab, Google Drive

**Deep learning:** TensorFlow / Keras, InceptionV3 (ImageNet pretrained weights), Ultralytics YOLOv8

**Image processing and data analysis:** OpenCV, NumPy, pandas, scikit-learn, Pillow, Matplotlib, Supervision, Hugging Face Hub

## Files

```text
FaceRecognizer/
├── FinalProject_Face_Recognizer_training.ipynb
├── FinalProject_Face_Recognizer_real_implement.ipynb
├── data/                       # Add your training face images here
└── README.md

Google Drive (created during training):
MyDrive/
├── datasets/
│   └── lfw-funneled.tgz        # Background/unknown-face dataset
└── saved_model/
    ├── labels.npy              # Encoded identity labels
    └── test_model_new.h5       # Trained Keras recognition model
```

> **IMPORTANT — Run the training notebook first.**
>
> You **must run `FinalProject_Face_Recognizer_training.ipynb` before `FinalProject_Face_Recognizer_real_implement.ipynb`**. Training produces **both `labels.npy` and `test_model_new.h5`**, and the implementation notebook requires them. Without these files, it cannot load the recognition model or map predictions to names.

## Getting Started

### 1. Open the training notebook in Google Colab

Clone the repository or open the notebooks directly from GitHub in Google Colab:

```bash
git clone https://github.com/ZanoTJ/FaceRecognizer.git
```

The training notebook already includes a Colab cell that clones this repository into `/content/FaceRecognizer`. If you've already cloned it, avoid cloning it again into an unexpected location.

### 2. Prepare the datasets

- Place face images of the people you want the model to recognize in the repository's **`data/`** directory. The notebook reads images from `/content/FaceRecognizer/data`.
- The training notebook extracts an LFW archive from **`/content/drive/MyDrive/datasets/lfw-funneled.tgz`** and samples faces as examples of the **`unknown`** class. Download/provide this dataset yourself and update the archive path if you store it somewhere else.
- The notebook derives an identity label using the **first whitespace-separated part of each image filename** (`os.path.basename(path).split()[0]`). Name your files accordingly, or adapt the label-extraction code.
- Mount Google Drive when prompted and create the **`MyDrive/saved_model/`** folder if it does not already exist.

> **Notebook setup note:** The training notebook contains Colab-specific paths and cells that may need minor adjustment. In particular, ensure `cv2`, `numpy`, and `Detections` are imported before functions that use them, and ensure the `dnn/` checkpoint directory exists before training. The model's input shape is `(200, 200, 3)`, so verify that every training image is consistently resized to 200 × 200 pixels.

### 3. Run the training notebook **FIRST**

Open and execute:

**`FinalProject_Face_Recognizer_training.ipynb`**

The training workflow:

1. Loads known-face images and samples unknown-face images from LFW.
2. Detects/crops selected faces and prepares images for training.
3. Encodes identity labels and saves **`labels.npy`**.
4. Splits the data into training and testing sets.
5. Builds a CNN with a pretrained **InceptionV3** backbone and dense classification layers.
6. Trains with augmentation, custom focal loss, Adam optimizer, early stopping, checkpointing, and learning-rate scheduling.
7. Evaluates the model and saves **`test_model_new.h5`**.

**Expected training outputs:**

```text
/content/drive/MyDrive/saved_model/labels.npy
/content/drive/MyDrive/saved_model/test_model_new.h5
```

The notebook saves the identity labels with `np.save(...)` and the recognition model with `model.save(...)`. **Confirm that both files exist before moving to the next notebook.**

### 4. Run the recognition notebook **SECOND**

Open and execute:

**`FinalProject_Face_Recognizer_real_implement.ipynb`**

The notebook mounts Google Drive and loads both artifacts:

```python
import numpy as np
from tensorflow.keras.models import load_model

label_train = np.load(
    '/content/drive/MyDrive/saved_model/labels.npy',
    allow_pickle=True
)

model = load_model(
    '/content/drive/MyDrive/saved_model/test_model_new.h5',
    custom_objects={'focal_loss_fixed': focal_loss}
)
```

> `focal_loss` is a custom function defined in the notebook. Run its definition cell **before** loading the model. If you change the training loss or Keras version, adjust loading as necessary.

The notebook separately downloads a pretrained **YOLOv8 face detection model** via Hugging Face Hub:

```python
model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt"
)
facial_model = YOLO(model_path)
```

You can then:

- **Test an image:** supply a photo at `/content/testing.jpg` (or change the path) and execute the image-testing cells.
- **Capture a webcam photo:** grant browser camera permission when Colab asks.
- **Run live detection:** execute the webcam-stream cells near the end of the notebook to draw boxes and identity labels on detected faces. Click the stream/video control to stop the demonstration.

## How It Works

```text
Image / Webcam Frame
         |
         v
YOLOv8 Face Detection
         |
         v
Crop Detected Faces
         |
         v
Resize to 200 x 200
         |
         v
InceptionV3-Based CNN
         |
         v
Predicted Class + Confidence
         |
         v
Map Class to labels.npy
         |
         v
Bounding Box + Name / "unknown"
```

**Important distinction:** The YOLOv8 model handles **face detection**, while the trained Keras/InceptionV3 model handles **identity recognition**. `labels.npy` preserves the ordering of names corresponding to the classifier outputs; replacing it with labels from another training run can produce incorrect names.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `labels.npy` or `test_model_new.h5` not found | Run the **training notebook first**, verify both generated files are in `MyDrive/saved_model/`, and mount the correct Google Drive account. |
| Model loading fails | Execute the `focal_loss` definition before `load_model(...)`; verify TensorFlow/Keras compatibility with the model used for training. |
| Missing data or empty dataset | Confirm images are in `/content/FaceRecognizer/data` and the LFW `.tgz` archive path matches your Drive. |
| Training checkpoint save fails | Create the `dnn/` directory before the checkpoint callback writes `dnn/face_recognizer.weights.h5`. |
| Tensor/image shape errors | Use consistently sized, valid **200 × 200 RGB/BGR three-channel face images** for the recognition CNN; review preprocessing cells if array shapes differ. |
| Webcam is unavailable | Use Google Colab in a supported browser, allow camera access, and run the notebook's JavaScript/webcam setup cells. |
| Prediction displays the wrong name | Ensure `labels.npy` and `test_model_new.h5` came from the **same training run**. |

## Notes

- These notebooks are written for **Google Colab** and use Colab-specific imports, JavaScript camera access, and absolute Google Drive paths; local Jupyter execution requires modifications.
- The recognition model and `labels.npy` are **not assumed to be pretrained downloads from this repository**: produce them yourself by running the training notebook.
- The separate pretrained YOLOv8 detector is downloaded automatically by the notebook, subject to network access.
- Face recognition is probabilistic; confidence thresholds do **not** guarantee a correct identification. Obtain consent before collecting or using people's facial images and do not treat this demo as a security-grade identity verifier.

## Project Workflow Summary

**Training notebook → `labels.npy` + `test_model_new.h5` → Real-time implementation notebook → Face detection and recognition**
