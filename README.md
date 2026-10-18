# MetaboNet Multimodal Glucose Forecasting

Training and inference code for type 1 diabetes glucose forecasting on the
[MetaboNet](https://metabo-net.org/) corpus
([leaderboard](https://metabonetglucose-leaderboard.hf.space/)), predicting CGM at
+30/+60/+90/+120 minutes and scored by DTS error-grid zone A (the share of predictions within
±20% of truth).

The repository is built as a controlled benchmark: the temporal backbones (TCN, PatchTST,
Mamba-2, LSTM, ModernTCN, xLSTM, Transformer++, MambaFormer, plus native TFT / TiDE /
PatchTST) share one window pipeline, one fusion front end, one multi-horizon head, one
training loop and one evaluation stack, so a measured difference is the backbone and nothing
else.

> Research code. Not a medical device and not validated for insulin dosing. No treatment or
> safety claim follows from retrospective forecasting accuracy.

## Setup

Python 3.9+, PyTorch 2.1+. There is no package install; every entry point runs as
`python -m metabonet_forecasting.<module>` from the repository root.

```bash
pip install -r requirements.txt
# optional: official Mamba-2 CUDA kernels (Ampere+); a pure-PyTorch scan is used otherwise
# pip install mamba-ssm causal-conv1d
```

Data is not redistributed here.

1. Request the MetaboNet corpus from its maintainers and place it at
   `data/metabonet_public.parquet`.
2. Clone the organisers' submission toolkit (templates and live-board targets) into `toolkit/`:

   ```bash
   git lfs install
   git clone https://github.com/replicahealth/glucose-prediction-submission-toolkit.git toolkit
   ```

## Pipeline

```
data/prepare.py   corpus parquet -> sequence store (memmapped lattice + anchors), built once
data/dataset.py   YAML `data:` block -> splits, train-only standardizer, lazy windows
models/           batch dict -> ForecastOutput (encoders -> fusion -> backbone -> shared head)
training/         Trainer -> best.pt, standardizer.json, train_log.csv, predictions_*.parquet
evaluation/       metrics.json and results tables, recomputed from saved predictions
submission/       competition parquets (predict, predict_annual, recalibrate)
```

Design points that the numbers depend on:

- **Causal by construction.** Segments are cut at every gap; windows only index
  `[t-L+1 : t]`; labels come from the raw CGM column, never an interpolated slot.
- **Split by subject, never by anchor.** Adjacent CGM readings correlate at ~0.99, so an
  anchor-level split leaks almost completely.
- **The standardizer is fitted on training anchors only** and frozen into each run directory;
  inference reuses it and never refits.
- **Config-driven.** `configs/base.yaml` documents every key and why its default is what it
  is. Configs compose with `include:`, and any key can be overridden with `--set a.b=value`.
  The fully resolved config is written into each run directory.

## Usage

```bash
scripts/00_setup_env.sh                     # check imports, GPUs, corpus
scripts/01_prepare_data.sh                  # build the sequence store (~30-60 min, ~40 GB RAM)
scripts/01_prepare_data.sh --smoke          # small store for a plumbing check
scripts/02_smoke_test.sh                    # every architecture end to end on the smoke store
```

Train and evaluate one model:

```bash
python -m metabonet_forecasting.training.train \
    --config configs/experiment/e2_backbones.yaml --name mamba2_ctx288_s1 \
    --set model.backbone.name=mamba2 data.context=288 seed=1

python -m metabonet_forecasting.evaluation.evaluate runs/mamba2_ctx288_s1
python -m metabonet_forecasting.evaluation.aggregate runs --out results
```

Stage scripts `10_`–`60_` run the benchmark grids (baselines, backbones, fusion, modalities,
context length, generalisation). The competition models were trained with:

| Script | What it trains |
|---|---|
| `scripts/run_ctx1152.sh` | `STAGE=store` builds the 96-hour-context store the models below use |
| `scripts/run_c96t.sh` | Transformer++ / Mamba-2 / MambaFormer, 96 h context, expected-zone-A (`zone_t`) loss |
| `scripts/run_c96a.sh` | annual specialists: the c96t models warm-started on an IOBP2-heavy mix |
| `scripts/run_final.sh` | the same three backbones trained on all data (live board) |

The headers of these scripts state the hypothesis, the settings and the measured effects.

## Inference

Build a submission store from the template, so test windows go through exactly the code that
built the training data, then predict with a run directory (`best.pt`, `config.resolved.yaml`,
`standardizer.json`):

```bash
# live board
COMPETITION=live scripts/90_make_submission.sh RUN_NAME

# annual: use predict_annual, which harmonises the patient statics to the
# pattern the models were trained on (see the module docstring for why)
python -m metabonet_forecasting.submission.predict_annual \
    --run runs/c96a_transformerpp \
    --store artifacts/store/submission_annual \
    --template toolkit/data/annual_competition/template.parquet \
    --out artifacts/submission/annual_c96a_transformerpp.parquet
```

Ensemble several members and apply the per-horizon zone-A recalibration, fitted on the live
board's public answer key and transferred to the annual set:

```bash
python -m metabonet_forecasting.submission.recalibrate \
    --live live_a.parquet live_b.parquet live_c.parquet \
    --annual annual_a.parquet annual_b.parquet annual_c.parquet \
    --targets toolkit/data/live_leaderboard/targets.parquet \
    --fit-weights --affine \
    --out-live live_ens.parquet --out-annual annual_ens.parquet
```

## License

MIT, see [LICENSE](LICENSE). The MetaboNet data and the organisers' toolkit are covered by
their own terms.
