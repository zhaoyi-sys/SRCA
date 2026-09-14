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
-> Linear128 encoder
-> raw h
-> Euclidean 3-NN distance
-> class-level percentile threshold from training ground-truth labels
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


