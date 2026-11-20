# Camera-Only Vehicle Speed Estimation

Estimating a car's speed (km/h) from dash-cam video alone, using a ConvNeXt-Tiny CNN followed by a bidirectional GRU, trained on the Comma2k19 driving dataset.

**Test RMSE: 1.54 km/h** on 9,600 unseen frames (8 drives). For comparison, always guessing the average speed gives 35.86 km/h.

![Predicted vs real speed on the test set](images/test_speed_and_rmse.png)

*Top: predicted vs real speed across all 8 test drives. Bottom: error per frame.*

---

## Results

| Model | Validation RMSE (km/h) | Test RMSE (km/h) |
|---|---|---|
| Baseline: always predict mean speed | – | 35.86 |
| Single CNN | ~3.3 | – |
| Ensemble of 3 CNNs | 2.65 | – |
| **3 CNNs + 30 GRUs (final)** | **1.18** | **1.54** |

Per-drive test RMSE ranges from 0.98 to 1.87 km/h.

Speed and other sensor data are **never used as model inputs**. Speed is used only as the training and evaluation label. All settings were chosen on the validation set; the test set was used once, at the end.

---

## How it works

The model runs in two stages.

**Stage 1: CNN, frame by frame**

1. Each frame is resized to 320×240 and cropped to the road area (160×320).
2. The input is 9 channels:
   - the current frame,
   - the difference between frames *t* and *t−4*,
   - the difference between frames *t−4* and *t−8*.

   Frame differences show the network *motion*, not just a single picture.
3. A ConvNeXt-Tiny pretrained on ImageNet is used, with its first layer widened from 3 to 9 input channels. It predicts speed and also produces a 768-number feature vector per frame.

**Stage 2: GRU, whole drive**

4. The CNN features for a whole drive are fed into a 2-layer bidirectional GRU, which looks at the frames before and after each moment and smooths out single-frame mistakes.
5. Final prediction = average of 3 CNNs (different random seeds) × 10 GRUs each = 30 models, plus mirror-image (flip) test-time augmentation and light smoothing.

### Key idea: time-scale augmentation

Taking frames further apart in time looks like driving faster. So during training, the frame gap is randomly scaled (×0.25 to ×2.0) **and the speed label is scaled by the same amount**. This creates realistic new training examples at many speeds and cut the error by about 40% in a single change (3.84 → 2.33 km/h test RMSE during development).

Other augmentations: "still" samples (the same frame repeated, labelled 0 km/h), random zoom and shift, brightness and contrast changes, blur, and horizontal flips.

---

## Training details

| | CNN (ConvNeXt-Tiny) | GRU |
|---|---|---|
| Parameters | 27.8 M | 593 K |
| Optimizer | AdamW (lr 1e-4, weight decay 0.05) | AdamW (lr 5e-4) |
| Schedule | OneCycle, 20 epochs, early stopping | Cosine, up to 6,000 steps, early stopping |
| Loss | Smooth L1 | MSE |
| Other | EMA weights, bf16 mixed precision, batch 64 | 400-frame windows, feature noise + dropout |

Training time: about 65–80 minutes per CNN on one GPU; each GRU trains in about 1 minute because CNN features are cached.

### Engineering

- All 91,200 frames are cached once as memory-mapped NumPy arrays, so training doesn't re-decode images.
- A custom threaded batch loader replaces PyTorch's DataLoader, which had problems on Windows.
- Channels-last memory format and bf16 mixed precision speed up training.

---

## Dataset

[Comma2k19](https://github.com/commaai/comma2k19), using a version with extracted frames:

| Split | Drives | Frames |
|---|---|---|
| Train | 60 | 72,000 |
| Validation | 8 | 9,600 |
| Test | 8 | 9,600 |

The dataset is **not included** in this repository because of its size. Expected folder layout:

```
data/Comma_2k19_Driving_Dataset/
├── Training/<segment>/labels.csv  (+ frame images)
├── Validation/<segment>/...
└── Testing/<segment>/...
```

Each `labels.csv` has the columns `frame`, `filename`, and `car_speed_km_h`.

---

## How to run

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Put the dataset in `data/` as shown above.
3. Run `speed_estimation.ipynb`. A CUDA GPU is required.

The first run builds the frame cache in `cache_160x320/`. Outputs:
- `test_predictions.csv`: real and predicted speed for every test frame
- `test_speed_and_rmse.png`: the chart above

---

## What I learned

- **Motion matters more than appearance.** Frame differences gave the network the information it needed; a single frame can't show speed.
- **Good augmentation can beat a bigger model.** Time-scale augmentation helped more than larger backbones, which overfit.
- **Ensembles stabilise noisy training.** Individual GRUs varied from run to run; averaging many removed most of that variance.
- **Things that didn't help:** optical flow (OpenCV Farneback), larger models, and sensor fusion were tried and dropped.

## How I built it

This project was developed with the help of an AI assistant (Gemini). I directed the experiments, decided what to try next, kept a log of what helped and what didn't, and checked every change against validation results before keeping it. The test set was kept separate and used only for the final evaluation.

---

*Deep Learning course project · KMITL Robotics & AI Engineering · 2026*
