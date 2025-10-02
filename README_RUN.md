## IndoML 2025 Datathon — Baseline (TF-IDF + LinearSVC)

### Setup

1. Create a virtual environment (recommended) and install deps:

```bash
pip install -r IndoML_Datathon/requirements.txt
```

### Train

```bash
python IndoML_Datathon/baseline/train_and_predict.py train \
  --train_path IndoML_Datathon/data/trainset.json \
  --model_dir IndoML_Datathon/baseline/models
```

### Predict

Generate predictions for dev-test:

```bash
python IndoML_Datathon/baseline/train_and_predict.py predict \
  --input_path IndoML_Datathon/data/dev_testset.json \
  --output_path IndoML_Datathon/sample_submission_file/predictions.json/predictions.json \
  --model_dir IndoML_Datathon/baseline/models
```

Generate predictions for test:

```bash
python IndoML_Datathon/baseline/train_and_predict.py predict \
  --input_path IndoML_Datathon/data/testset.json \
  --output_path IndoML_Datathon/sample_submission_file/predictions.json/predictions.json \
  --model_dir IndoML_Datathon/baseline/models
```

Notes:
- The script predicts both `Mistake_Identification` and `Providing_Guidance` labels and writes them under `tutor_responses -> <tutor> -> annotation`.
- Output structure mirrors the input schema expected by Codabench (conversation id/history preserved).

## Advanced Transformer Pipeline (DeBERTa v3, k-fold ensemble)

### Install extra deps

```bash
pip install -r IndoML_Datathon/requirements.txt
```

### Train k-fold models (per track)


Track 1 — Mistake Identification:
```bash
python -m advanced_Task2.train_kfold_track `
  --track mi `
  --model_name models/deberta-v3-large `
  --train_path data/trainset.json `
  --out_dir advanced_Task2/models `
  --folds 2 --epochs 4 --batch_size 2 --lr 3e-5 --max_len 128
```

Track 2 — Providing Guidance:
```bash
python -m advanced_Task2.train_kfold_track `
  --track pg `
  --model_name models/deberta-v3-large `
  --train_path data/trainset.json `
  --out_dir advanced_Task2/models `
  --folds 2 --epochs 4 --batch_size 2 --lr 3e-5 --max_len 128
```

### Predict with ensemble (per track)


Dev/Test for Track 1 (MI):
```bash
python -m advanced_Task2.predict_ensemble_track `
  --track mi `
  --model_name models/deberta-v3-large `
  --models_dir advanced_Task2/models `
  --input_path data/dev_testset.json `
  --output_path advanced_Task2/preds_mi.json
```

Dev/Test for Track 2 (PG):
```bash
python -m advanced_Task2.predict_ensemble_track `
  --track pg `
  --model_name models/deberta-v3-large `
  --models_dir advanced_Task2/models `
  --input_path data/dev_testset.json `
  --output_path advanced_Task2/preds_pg.json
```

To predict for test, change `--input_path` accordingly.

### Merge per-track predictions into final submission

```bash
python -m advanced_Task2.merge_tracks `
  --mi_path advanced_Task2/preds_mi.json `
  --pg_path advanced_Task2/preds_pg.json `
  --output_path sample_submission_file/predictions.json/predictions.json
```

### What each file in `advanced_Task2/` does

- `modeling_singlehead.py`: DeBERTa-based encoder with a single classification head (3 classes). Used for one track at a time.
- `train_kfold_track.py`: Trains k-fold models for a chosen track (`--track mi` or `--track pg`). Uses (context, response) pair encoding, class-weighted loss, AMP, and linear warmup.
- `predict_ensemble_track.py`: Loads all fold checkpoints for the chosen track, averages logits (softmax voting), and writes a per-track JSON with only that track’s labels filled.
- `merge_tracks.py`: Combines the two per-track JSONs (MI and PG) into the final `predictions.json` in the required Codabench schema.

