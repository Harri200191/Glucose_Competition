# MetaboNet Multimodal Glucose Forecasting

A controlled benchmark and architecture study for type 1 diabetes glucose forecasting:
how should CGM, insulin, carbohydrate, activity and static information be **encoded and
fused** so that sequence models learn transferable glucose dynamics rather than
short-term autocorrelation or dataset-specific shortcuts?

Five temporal backbones (TCN, TiDE, TFT, PatchTST, Mamba-2) and four fusion strategies
(raw concatenation, modality encoders, gated, CGM-centred FiLM) over one common window
pipeline, one training loop and one evaluation stack.

**Everything — the layout, every command, the experiment stages, what has been verified
and what has not — is in [RUNBOOK.md](RUNBOOK.md).**

```bash
scripts/00_setup_env.sh              # check the environment
scripts/01_prepare_data.sh --smoke   # small dataset build
scripts/03_run_tests.sh              # 65 tests: leakage, alignment, splits, metrics
scripts/02_smoke_test.sh             # all architectures, end to end, ~1 minute
```

This is a research forecasting benchmark, not a validated insulin-dosing system. No
treatment or safety claim follows from retrospective accuracy alone.
