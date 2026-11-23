# TinyML-Based Industrial Fault Detection

A software-only, reproducible TinyML workflow. It simulates three-axis vibration, temperature, and motor current; extracts 18 features from 128-sample windows; trains a five-state 1D CNN; converts it to a full-integer INT8 TensorFlow Lite model; and runs a streaming protective-trip simulation locally.

No sensor, microcontroller, or cloud account is required. The default data source is the deterministic synthetic generator. Optional local CSV and CWRU-style MATLAB files can be placed in `data/real/`.

## Health-state labels

| ID | State |
|---:|---|
| 0 | Normal |
| 1 | Bearing Wear |
| 2 | Imbalance |
| 3 | Misalignment |
| 4 | Overheating |

## VS Code setup (Windows PowerShell)

Use Python 3.10–3.12 (Python 3.11 is recommended; TensorFlow wheels are not available for every Python release). Open this repository as a VS Code folder, then run in its integrated PowerShell terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. In VS Code, select `.venv` with **Python: Select Interpreter**.

## Train, quantize, test, and simulate

Run these commands from the repository root with `.venv` activated:

```powershell
python -m models.train
python -m quantize.convert
python -m unittest discover -s tests -v
python -m edge_sim.run --scenario cycle --windows 30
```

`models.train` saves the Keras model and feature scaler in `models/artifacts/`, and reports float-model accuracy, macro F1, confusion matrix, model size, and per-window latency. `quantize.convert` calibrates a full INT8 model using representative training features, writes float32 and full-INT8 `.tflite` models plus the TFLite-Micro-style `model_data.h` into that directory, then reports comparable float32/INT8 accuracy, macro F1, confusion matrix, footprint, and latency.

To simulate a particular state, use (for example) `python -m edge_sim.run --scenario bearing_wear --windows 12`. `cycle` alternates normal and fault conditions. The edge loop latches the relay/buzzer/LED only when the same non-normal class is predicted above 0.85 confidence for two consecutive windows. Three consecutive high-confidence Normal windows simulate recovery and reset the actuators.

Optional controls:

```powershell
python -m models.train --dataset synthetic --recordings-per-class 60
python -m models.train --dataset auto --data-dir data/real
python -m edge_sim.run --scenario overheating --windows 15 --interval 0.05
```

`auto` uses recognized local data when available and falls back to synthetic data if no usable dataset is found. If real data omits or has too few windows for one or more of the five required states, synthetic examples are added for those states. CSV files should contain a label column and vibration columns (`x`, `y`, `z` or common acceleration aliases); temperature and current columns are optional. CWRU `.mat` files are read when SciPy is installed, using `DE_time` (or another recognized acceleration signal) and a state inferred from the file/folder name. Headerless whitespace/tab-delimited `.txt` or `.dat` samples are also supported when their parent folder identifies the state, which accommodates exported NASA bearing test runs. Supported state-name keywords include `normal`, `bearing`/`inner`/`outer`/`ball`, `imbalance`/`unbalance`, `misalignment`, and `overheat`/`thermal`. Data not in these formats can be converted to the documented CSV columns.

## Design and outputs

- Sampling is 1 kHz equivalent; windows contain 128 samples with a 64-sample (50% overlap) hop.
- Each window maps to 18 values: five vibration statistics/spectral features per axis (RMS, excess kurtosis, peak-to-peak, dominant FFT frequency, and spectral centroid), plus temperature mean, temperature maximum, and current mean.
- `data_sim/` creates reproducible sensor records and loads supported local datasets. `features/` owns windowing and feature extraction. `models/` trains/evaluates the Keras classifier. `quantize/` creates full INT8 TFLite and a C array header. `edge_sim/` handles streaming inference, alert logging, and simulated actuator/recovery logic.
- The terminal edge view displays sensor readings, state/confidence, inference latency, and actuator status. Training and quantization reports include the confusion matrix and model size.
- Generated model files and alert logs are ignored by Git and can be recreated with the commands above.

## Tests

The unittest suite covers 50% window overlap, exactly 18 extracted features, protective trip/recovery behavior, and a small train-and-full-INT8-convert smoke test. Run it with `python -m unittest discover -s tests -v`.
