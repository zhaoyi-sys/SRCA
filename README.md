# SRCA: Open-Set NILM Load Recognition

This repository contains a clean engineering implementation of SRCA for
open-set non-intrusive load monitoring experiments on LILAC and ILSED.

## Core Protocol

For each experiment, one device class is held out as unknown. Known classes are
randomly split into training and testing sets. The unknown class is never used
for MiniROCKET fitting, Ridge closed-set classifier fitting, SpHOR training, or
threshold calibration.

The kNN reference library used at test time stores only:

- `h_train_128`: 128-D SpHOR embeddings of known training samples.
- `y_train_groundtruth`: ground-truth labels of those training samples.
- `train_sample_index`: source sample indices.
- `thresholds`: class-adaptive thresholds calibrated from training leave-one-out
  kNN distances grouped by ground-truth training class.

This version does not re-predict training classes to calibrate thresholds.

## Main Files

- `train_srca_open_set.py`: train and save deployable artifacts.
- `test_srca_open_set.py`: load an artifact and evaluate the test split.
- `run_lilac_ilsed_srca_open_set.py`: train/test LILAC and/or ILSED in one run.
- `srca_open_set_engineering/pipeline.py`: reusable implementation.
- `nilm_clean_open/`: dataset loading, split, MiniROCKET, SpHOR, and rejection utilities.
- `minirocket.py`: MiniROCKET CPU/Numba implementation.

## Example

Train LILAC with `fluorescent-lamp` as the unknown class:

```powershell
python train_srca_open_set.py `
  --dataset lilac `
  --unknown-label fluorescent-lamp `
  --output-dir outputs_srca_open_set_engineering `
  --seed 42 `
  --reject-k 3 `
  --threshold-percentile 93
```

Test the saved artifact:

```powershell
python test_srca_open_set.py `
  --artifact-dir outputs_srca_open_set_engineering\lilac_unknown_fluorescent-lamp_seed42
```

Train and test all leave-one-unknown tasks for both datasets:

```powershell
python run_lilac_ilsed_srca_open_set.py `
  --datasets lilac,ilsed `
  --output-dir outputs_srca_open_set_engineering `
  --seed 42 `
  --reject-k 3 `
  --threshold-percentile 93
```

