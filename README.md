# OPEN-SET INDUSTRIAL LOAD RECOGNITION USING SPHERICAL REPRESENTATION AND CLASS-ADAPTIVE KNN REJECTION

This directory is the clean implementation of the selected open-set NILM method.

## Pipeline

Closed-set branch:

```text
current i + instantaneous power p
-> MiniROCKET per stream
-> concatenate features
-> MinMaxScaler
-> RidgeClassifier
-> predicted known class
```

Rejection branch:

```text
same raw MiniROCKET features
-> independent MinMaxScaler
-> SpHOR Linear128 encoder: h = GELU(Wf + b)
-> raw h
-> Euclidean 1-NN distance
-> class-level percentile threshold from training ground-truth labels
```

The held-out unknown class is never used for MiniROCKET fitting, MinMaxScaler
fitting, Ridge training, SpHOR training, or threshold calibration.

## Modules

- `datasets.py`: load LILAC, PLAID-other, and WHITED-other as `i+p` streams.
- `splits.py`: leave-one-class-out known/unknown random split.
- `minirocket_features.py`: fit MiniROCKET on training streams and transform train/test streams.
- `closed_set.py`: MinMaxScaler + RidgeClassifier closed-set branch.
- `sphor.py`: Linear128 SpHOR representation learning.
- `rejection.py`: raw-h Euclidean 1-NN scores, ground-truth class thresholds, open-set metrics.
- `experiment.py`: one-fold and full-dataset orchestration.
- `../run_clean_open_nilm_method.py`: command-line runner.

## Default Hyperparameters

```text
num_features_per_stream = 672
input streams = i + p
closed classifier = RidgeClassifier(class_weight="balanced")
feature scaler = MinMaxScaler
SpHOR encoder = Linear(D,128) + GELU
SpHOR epochs = 100
SpHOR lr = 5e-4
weight_decay = 1e-4
batch_size = 32
temperature = 0.10
ortho_weight = 0.1
label_smoothing = 0.1
mixup_alpha = 1.0
grad_clip = 5.0
reject space = raw h
reject distance = Euclidean 1-NN
threshold group = training ground-truth class
threshold percentiles = 90,91,92,93,94,95,96,97,98,99,100
split = random, train_fraction=0.8
```

## Test-Time kNN Reference

For SpHOR rejection, each fold writes:

```text
knn_reference_seed{seed}_unknown_{label}.npz
```

The file stores only the training reference vectors and their ground-truth class
IDs for kNN rejection:

```text
train_vectors_128
y_train_groundtruth
train_sample_index
known_classes
```

Threshold calibration uses `y_train_groundtruth`.

## Example Commands

LILAC:

```powershell
D:\anaconda3\python.exe -u run_clean_open_nilm_method.py `
  --dataset lilac `
  --dataset-dir ".\lilac\agr" `
  --label-file labels_corrected.npy `
  --output-dir outputs_lilac_clean_final_method
```


