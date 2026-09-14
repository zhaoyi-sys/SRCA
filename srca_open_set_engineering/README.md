# SRCA Open-Set NILM Engineering Pipeline

This folder contains a clean train/test pipeline for LILAC and ILSED open-set
load recognition.

## Key Policy

The test-time kNN reference library stores only:

- `h_train_128`: the 128-D SpHOR embeddings of known training samples.
- `y_train_groundtruth`: the ground-truth known-class IDs of these training samples.
- `train_sample_index`: original sample indices.
- `thresholds`: class-adaptive rejection thresholds calibrated from training
  leave-one-out kNN distances grouped by ground-truth class.

The training set is not re-predicted to group thresholds. In other words, this
pipeline does not use `closed.train_pred` or predicted training labels for the
class-adaptive threshold.

## Pipeline

Training:

```text
known training waveforms
-> MiniROCKET fit/transform
-> Ridge closed-set classifier fit
-> SpHOR Linear128 training
-> encode training samples as h_train_128
-> leave-one-out kNN distance on h_train_128
-> true-class percentile thresholds from y_train_groundtruth
-> save deployable artifacts
```

Testing:

```text
test waveforms
-> saved MiniROCKET transform
-> Ridge predicted known class
-> saved SpHOR encoder -> h_eval_128
-> kNN distance to saved h_train_128
-> threshold selected by predicted known class
-> known / unknown decision
```

## Files

- `../train_srca_open_set.py`: trains one or all leave-one-unknown artifacts.
- `../test_srca_open_set.py`: tests one trained artifact.
- `../run_lilac_ilsed_srca_open_set.py`: trains and tests LILAC and/or ILSED.
- `pipeline.py`: reusable implementation.

## Example Commands

LILAC, one unknown class:

```powershell
python train_srca_open_set.py `
  --dataset lilac `
  --unknown-label fluorescent-lamp `
  --output-dir outputs_srca_open_set_engineering `
  --seed 42 `
  --reject-k 3 `
  --threshold-percentile 93
```

Then test it:

```powershell
python test_srca_open_set.py `
  --artifact-dir outputs_srca_open_set_engineering\lilac_unknown_fluorescent-lamp_seed42
```

ILSED, one unknown class:

```powershell
python train_srca_open_set.py `
  --dataset ilsed `
  --unknown-label 加热管 `
  --output-dir outputs_srca_open_set_engineering `
  --seed 42 `
  --reject-k 3 `
  --threshold-percentile 93
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

