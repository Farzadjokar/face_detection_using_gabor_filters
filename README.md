# Gabor Face Detector

A desktop app (Python + **tkinter**) that loads an image and detects faces using **Gabor filter features**, a **PCA** projection and a **Probabilistic Neural Network (PNN)** classifier, with no deep learning and no pre-built face detectors.

It is a from-scratch implementation of the classification-based approach covered in my Computer Vision course slides on Gabor filters (see [References](#references)).

<p align="center">
  <img src="docs/example_astronaut.png" width="32%" alt="Detection on a portrait">
  <img src="docs/example_composite.png" width="32%" alt="Faces of different sizes">
  <img src="docs/gabor_bank.png" width="32%" alt="Gabor filter bank">
</p>

## Features

- Open any image (PNG, JPG, BMP, TIFF, WebP) and detect faces at multiple scales
- Live sliders: confidence threshold, minimum overlapping hits, NMS overlap and minimum face size. Moving a slider re-filters the cached scan instantly
- Detection runs in a background thread, so the GUI never freezes (progress bar included)
- **"Look inside" windows:** view the Gabor filter bank (4 scales × 6 orientations) and the Gabor magnitude responses of your own image
- Optionally show every raw candidate window to see how non-maximum suppression works
- Save the annotated result as PNG/JPG
- Retrain on your own faces / backgrounds with one command

## How it works

The pipeline mirrors the "Application 2: face detection" slides:

| Step | What happens | Where |
|---|---|---|
| 1. Scan | A **20×20 window** slides over an **image pyramid** (scale step 1.2) so faces of any size are examined | `GaborFaceDetector.candidates` |
| 2. Gabor filtering | **Four Gabor filters** (0°, 45°, 90°, 135°; wavelength 6 px, σ = 2) → magnitude of the complex response | `face_filters`, `magnitude_maps` |
| 3. Feature vector | 10×10 response per filter, **8 corner pixels removed** → 4 × 92 = **368 features**, log-compressed and L2-normalised for illumination/contrast invariance | `scan_features` |
| 4. PCA | 368 → 60 dimensions | `PCAPNN` |
| 5. PNN | Parzen-window classifier with Gaussian kernels gives a face likelihood (confidence = sigmoid of the log-likelihood ratio) | `PCAPNN.predict_proba` |
| 6. Merge | Overlapping detections within/across scales: **keep the candidate with the highest face likelihood**; boxes lying inside a larger detection (eyes, cheeks) are dropped | `nms`, `GaborFaceDetector.filter` |

Design notes:

- **Valid convolution.** The Gabor kernels are 11×11, and only responses whose kernel lies fully inside the 20×20 window are used, which gives exactly the 10×10 map per filter. A window's features therefore depend only on its own pixels, so training patches and scanned windows are processed identically.
- **Localisation-aware negatives.** Besides random background crops, the training set contains zoomed-in face *parts*, half-shifted and zoomed-out views of real faces as non-faces. This teaches the PNN that only a *tight* face window is a face.
- **Hard-negative mining.** The detector is run on face-free pictures; its false positives are added to the non-face set and the model is refit (3 rounds).
- **Fast PNN.** Training samples are compressed into weighted k-means prototypes (500 face / 1200 non-face), so a 512×512 image is scanned in a few seconds on a laptop CPU.
- `gabor_face.py` also contains the **Manjunath & Ma Gabor wavelet bank** (slides 8–16: `a`, `σu`, `σv`, `θn = nπ/K`, with `Uh=0.4, Ul=0.05, S=4, K=6`), used for the filter-bank visualisation.

## Installation

```bash
git clone https://github.com/<your-username>/gabor-face-detector.git
cd gabor-face-detector
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`tkinter` ships with the standard Python installers for Windows and macOS. On Debian/Ubuntu: `sudo apt install python3-tk`.

## Usage

```bash
python app.py
```

1. **Open image…**
2. **Detect faces** (the first scan of a photo takes a few seconds)
3. Tune the sliders. If real faces are missed, lower *Confidence threshold*; if you see false boxes, raise it or increase *Min. overlapping hits*
4. **Save result…**

A pre-trained `model.npz` is included. To retrain (or add your own data):

```bash
python train.py                                    # uses the 100 faces / 100 non-faces bundled with scikit-image
python train.py --faces my_faces/ --nonfaces my_backgrounds/
python make_figures.py                             # regenerate docs/*.png
```

Use tight head crops (forehead to chin) for `--faces`, and photos without faces for `--nonfaces`.

You can also use the engine without the GUI:

```python
import cv2, numpy as np
from gabor_face import PCAPNN, GaborFaceDetector, face_filters

det = GaborFaceDetector(PCAPNN.load("model.npz"), face_filters())
gray = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
boxes, _ = det.detect(gray, threshold=0.8)          # [(x, y, w, h, confidence), ...]
```

## Project structure

```
app.py            tkinter GUI
gabor_face.py     Gabor bank, features, PCA + PNN, multi-scale detector, NMS
viz.py            drawing helpers (boxes, filter bank, response maps)
train.py          training script (augmentation, hard-negative mining)
make_figures.py   regenerates the images in docs/
model.npz         pre-trained classifier
```

## Results and honest limitations

- On a held-out split the patch classifier reaches an AUC of about 0.96. That split is small (≈125 face / 500 non-face patches), so treat the number as indicative only.
- Qualitatively it localises frontal faces well in clean portraits and in scenes with several faces of different sizes (see the images above).
- **This is an educational project, not a production detector.** The bundled training set contains only 100 faces, so expect missed faces and false positives, especially on:
  - small faces (default minimum is 40 px) and strongly textured backgrounds (grass, fabric, foliage)
  - non-frontal, rotated or heavily occluded faces
- Quality improves substantially with more training faces (`--faces`) and more varied background images (`--nonfaces`). For real applications use a modern CNN-based detector.

## Ideas for extension

- Multi-scale Gabor bank (more than 4 filters) or learnable Gabor layers (e.g. GaborNet)
- Replace the PNN with an SVM or a small CNN on the same features
- Webcam / video mode, eye and mouth landmark localisation from the Gabor responses

## References

- B. S. Manjunath and W. Y. Ma, *Texture Features for Browsing and Retrieval of Image Data*, IEEE TPAMI, vol. 18, no. 8, 1996
- L.-L. Huang, A. Shimizu and H. Kobatake, *Classification-Based Face Detection Using Gabor Filter Features*
- A. Alekseev and A. Bobe, *GaborNet: Gabor filters with learnable parameters in deep convolutional neural networks*

## License

MIT, see `LICENSE` (add the file with your name before publishing).