# ASL Sign Recognition — Deep Learning with MediaPipe & BiLSTM

**An experimental American Sign Language (ASL) single-sign classifier built with PyTorch, MediaPipe, and OpenCV.** The system recognizes a short, isolated sign from a prerecorded video or a live webcam recording and predicts its corresponding English word.

> **Scope:** This is a university deep learning project and a **three-class sign recognition prototype**, not a general-purpose ASL interpreter or sentence translator. Its results are preliminary and should not be used as a substitute for human interpretation or accessibility services.

**Project team:** Ella Bar, Naomi Green, and Daniella Simonovsky  
**Original collaborative repository:** [ellahazelbar/DeepWSSubmission](https://github.com/ellahazelbar/DeepWSSubmission)

## Highlights

- **Self-enriched dataset:** supplemented publicly sourced sign videos with team-recorded examples; approximately 40 recordings per target word before mirroring.
- **Landmark-based features:** MediaPipe Holistic extracts hand and pose landmarks, plus a compact face bounding box, reducing dependence on raw video pixels.
- **Temporal classification:** a PyTorch bidirectional LSTM (BiLSTM) models how gestures evolve across frames.
- **Experimental comparison:** evaluated multiple architectures and training configurations, including CNN–LSTM, ResNet50–BiLSTM, and attention-based models.
- **Two inference modes:** recognize a sign from an existing video or record and classify a sign using a webcam.
- **Best reported validation accuracy:** **81.4%** on the project's restricted three-word dataset (80/20 training/validation split). This is **not** an independent test-set or cross-signer generalization result.

## Problem and approach

ASL recognition is challenging because signing varies with hand shape, movement, viewpoint, and individual style. Learning directly from videos also requires handling lighting, clothing, background, and signer position, which is expensive when only a small amount of training data is available.

We investigated whether a more compact representation of motion could improve learning efficiency on a small dataset. Our final pipeline converts each video into sequences of body and hand coordinates, then uses a recurrent network to classify the sign.

```text
Video or webcam
       │
       ▼
OpenCV video capture / frame handling
       │
       ▼
MediaPipe Holistic landmark extraction
       │
       ▼
Per-frame normalization + temporal feature sequence
       │
       ▼
PyTorch bidirectional LSTM
       │
       ▼
Classification logits → predicted sign
```

### Feature extraction

The landmark pipeline represents each frame using **77 three-dimensional landmarks (231 numerical features)**:

| Source | Landmarks |
| --- | ---: |
| Left hand | 21 |
| Right hand | 21 |
| Body pose | 33 |
| Face bounding box (minimum and maximum corners) | 2 |
| **Total** | **77** |

Instead of using all 468 facial landmark points, the pipeline retains a bounding box around the face. This preserves coarse spatial information about the face relative to the hands without allowing facial features to dominate the representation. The landmark coordinates are standardized on a per-frame basis, and mirrored variants are generated for augmentation.

The checked-in preprocessing code uses **139 frames** as its nominal sequence length and encodes each frame as **231 features**. It stores processed training samples in `.piped` files.

### Model and experiments

The final approach uses a **two-layer BiLSTM** with a classification head. Its bidirectional processing captures temporal information from both directions of the recorded sequence. Implementation settings are configurable in [`src/models/keypoint_bilstm.py`](src/models/keypoint_bilstm.py); the checked-in model defaults should be treated as the authority when loading the included weights.

Before choosing the landmark-based approach, we also experimented with more computationally expensive raw-video models, including:

- A convolutional LSTM baseline.
- A ResNet50 + BiLSTM architecture.
- An attention-based model.

We investigated different training choices and regularization strategies, including optimizers, learning-rate scheduling, and dropout. The landmark pipeline reduced input dimensionality and made further experiments more practical.

## Data and evaluation

The initial collection of online ASL videos proved smaller and less consistent than expected: some source links were unavailable, while signs under the same English label were not always visually equivalent. We therefore reviewed the signs manually, recorded additional examples, and augmented the available clips through mirroring.

The **released demo is restricted to the following vocabulary** (defined in [`demo/predict.py`](demo/predict.py)):

| Class label |
| --- |
| `change` |
| `cousin` |
| `trade` |

**Results reported in the final assignment:**

| Measure / setting | Reported value |
| --- | --- |
| Best validation accuracy | **81.4%** |
| Dataset split | 80% training / 20% validation |
| Final-model training described in the report | 100 epochs |
| Hardware used for the reported training | Google Colab T4 GPU |
| Typical late training accuracy | Approached 100% |
| Raw-video baselines' reported validation range | Approximately 60–80% |

These results demonstrate better-than-chance performance on a limited classification task, but they also reveal **substantial overfitting**. The small vocabulary and small number of examples limit the conclusions that can be drawn about unseen signers, recording conditions, or new ASL vocabulary. The reported figure is a validation result, not a benchmark of real-world ASL translation performance.

**Reproducibility note:** The final report describes a 100-epoch experiment; the checked-in [`src/training/train.py`](src/training/train.py) currently specifies 50 epochs and local data/output paths. The repository contains saved model weights, but the original training videos and processed training samples are **not included**. Consequently, reproducing the reported result requires reconstructing the dataset and matching the original experimental configuration.

## Repository structure

```text
DeepWSSubmission/
├── README.md                    # Project overview and usage
├── requirements.txt             # Core Python dependencies
├── ProcessData.py               # Auxiliary video augmentation script
├── demo/
│   ├── predict.py               # Prediction from an existing video
│   └── predict_realtime.py      # Webcam-based prediction
├── models/
│   ├── best_model_gd.pth        # Saved model checkpoint
│   └── best_model_sgd.pth       # Saved model checkpoint used by default demo
└── src/
    ├── data/
    │   ├── piper.py             # MediaPipe extraction and video handling
    │   ├── process_wlasl.py     # Earlier dataset organization utility
    │   └── Readme.txt          # Notes that the training data was removed
    ├── models/
    │   ├── keypoint_bilstm.py   # Landmark-based BiLSTM classifier
    │   ├── cnn_lstm.py          # CNN–LSTM experiment
    │   ├── resnet50_bilstm.py   # ResNet50–BiLSTM experiment
    │   └── attention_resnet50.py # Attention-based architecture
    └── training/
        └── train.py             # Training script (requires local data/path setup)
```

## Getting started

### Requirements

- Python 3 and the packages in `requirements.txt`.
- An accessible video file for offline prediction, or a working webcam for live prediction.
- A compatible PyTorch/MediaPipe environment. A GPU is helpful for model training, but the demo chooses CUDA only when available.

Clone **your fork** using its URL (or download the source from GitHub), then from the repository root run:

```bash
python -m pip install -r requirements.txt
```

> **Compatibility:** This is an academic project, not a maintained pip package. The dependency file specifies minimum versions rather than a fully pinned environment, so some newer library releases may require adjustments. The optional `ProcessData.py` augmentation utility also imports MoviePy, which is not listed among the core requirements.

### 1. Predict a sign from a video

```bash
python demo/predict.py path/to/sign_video.mp4
```

The script loads `models/best_model_sgd.pth` by default and prints a predicted class, for example:

```text
Predicted sign: cousin
```

You can supply a different checkpoint:

```bash
python demo/predict.py path/to/sign_video.mp4 --model_path models/best_model_gd.pth
```

The video should contain **one isolated sign**, not a continuous ASL sentence. The sample output above is illustrative; the predicted label depends on the video.

### 2. Predict a sign from a webcam

```bash
python demo/predict_realtime.py
```

A webcam window opens and records frames until a key is pressed; the script then processes the captured sequence and prints the predicted label. The capture uses the default camera (`0`) and requires a graphical environment.

### 3. Training on a custom dataset (advanced)

The training and preprocessing scripts are included as research code, but **training is not plug-and-play**. In particular, the scripts currently use local paths originally configured for the project team and expect class-organized `.piped` training samples.

Before retraining, you would need to obtain/record appropriately permitted sign videos, prepare the data, configure the paths in [`src/data/piper.py`](src/data/piper.py) and [`src/training/train.py`](src/training/train.py), and decide on a reproducible train/validation protocol. The code should also be reviewed for model/checkpoint configuration consistency before comparing results to the submitted report.

## Limitations and future work

The key limitations are the **three-word vocabulary**, limited training data, and overfitting. Furthermore, sign-level classification does not address the segmentation, grammatical structure, and translation challenges of full ASL sentences.

Potential extensions include collecting a larger, more diverse dataset; evaluating generalization to unseen signers; improving data splitting and augmentation protocols; testing alternative temporal architectures; and expanding the vocabulary before attempting continuous-sign recognition.

## Contributors and attribution

This project was developed collaboratively by **Ella Bar, Naomi Green, and Daniella Simonovsky** as a university deep learning assignment. The repository is derived from the [original shared project](https://github.com/ellahazelbar/DeepWSSubmission). Please preserve the original contribution history and credit when forking or adapting it.

The data and third-party frameworks have their own usage terms. The original repository does not specify an open-source license; do not assume that the code, dataset, or recorded videos are available for unrestricted redistribution.

## References

- Lugaresi et al., *MediaPipe: A Framework for Building Perception Pipelines* (2019). https://arxiv.org/abs/1906.08172
- Hochreiter and Schmidhuber, *Long Short-Term Memory* (1997).
- The project's final submission report and source code in the [original repository](https://github.com/ellahazelbar/DeepWSSubmission).
