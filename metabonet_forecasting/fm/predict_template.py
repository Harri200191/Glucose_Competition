"""Chronos-2 predictions for a competition template (submission store).

    python -m metabonet_forecasting.fm.predict_template \
        --model-id runs/c2ft_lora/model --gpu 6 \
        --store artifacts/store/submission_live \
        --template artifacts/submission/live_template.parquet \
        --out artifacts/submission/live_c2ft.parquet

Same contract as ``submission.predict``: every template row gets a prediction
in template order, written with the template's exact schema. Raw CGM in mg/dL
straight from the submission store's channel memmap; insulin (basal+bolus) as
past + future-known covariates, future block capped at the anchor's segment
end. Quantile side table written next to the output.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

GRID_MIN = 5
PRED_STEPS = 24
HSTEPS = {30: 6, 60: 12, 90: 18, 120: 24}
COV = ["ins_basal", "ins_bolus"]
QLEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--context", type=int, default=288)
    a = ap.parse_args(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import pandas as pd
    import pyarrow.parquet as pq
    import pyarrow as pa
    import torch
    from chronos import Chronos2Pipeline

    from ..data.store import SequenceStore

    store = SequenceStore(a.store)
    anchors = store.anchors
    if "template_row" not in anchors.columns:
        raise SystemExit(f"{a.store} is not a submission store")
    template_tbl = pq.read_table(a.template)
    template = template_tbl.to_pandas()

    rows = anchors["row"].to_numpy()
    template_row = anchors["template_row"].to_numpy()
    seg_end = store.segments.set_index("seg")["end"].loc[
        anchors["seg"].to_numpy()].to_numpy()
    cgm = int(store.channel_ids(["cgm"])[0])
    cov_ids = {n: int(store.channel_ids([n])[0]) for n in COV}
    seg_start = store.segments.set_index("seg")["start"].loc[
        anchors["seg"].to_numpy()].to_numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(a.model_id, device_map=device)
    print(f"{len(anchors):,} anchors, model {a.model_id} on {device}")

    n = len(anchors)
    pred = np.full((n, 4), np.nan, dtype=np.float32)
    quant = np.full((n, 4, len(QLEVELS)), np.nan, dtype=np.float32)
    t0 = time.time()
    for lo in range(0, n, a.batch):
        hi = min(lo + a.batch, n)
        inputs = []
        for i in range(lo, hi):
            r = int(rows[i])
            s = max(r - a.context + 1, int(seg_start[i]))
            hist = np.asarray(store.channels[s: r + 1, cgm], dtype=np.float32)
            past = {nm: np.nan_to_num(np.asarray(
                store.channels[s: r + 1, c], dtype=np.float32))
                for nm, c in cov_ids.items()}
            fut = {}
            usable = int(min(PRED_STEPS, max(int(seg_end[i]) - r - 1, 0)))
            for nm, c in cov_ids.items():
                f = np.zeros(PRED_STEPS, dtype=np.float32)
                if usable > 0:
                    f[:usable] = np.nan_to_num(np.asarray(
                        store.channels[r + 1: r + 1 + usable, c],
                        dtype=np.float32))
                fut[nm] = f
            inputs.append({"target": hist, "past_covariates": past,
                           "future_covariates": fut})
        q, mean = pipe.predict_quantiles(
            inputs, prediction_length=PRED_STEPS, quantile_levels=QLEVELS)
        steps = [HSTEPS[h] - 1 for h in (30, 60, 90, 120)]
        pred[lo:hi] = np.stack(
            [np.asarray(m.float().cpu())[0][steps] for m in mean])
        quant[lo:hi] = np.stack(
            [np.asarray(t.float().cpu())[0][steps][:, :] for t in q])
        if (lo // a.batch) % 20 == 0:
            print(f"  {hi:,}/{n:,}  {hi / max(time.time() - t0, 1e-9):,.0f}/s",
                  flush=True)

    # into template order, exact schema
    out_tbl = template_tbl
    side = pd.DataFrame({"id": template["id"].to_numpy()})
    for j, h in enumerate((30, 60, 90, 120)):
        col = f"pred_{h}"
        values = np.full(len(template), np.nan)
        values[template_row] = np.clip(pred[:, j], 40.0, 400.0)
        idx = out_tbl.schema.get_field_index(col)
        out_tbl = out_tbl.set_column(
            idx, out_tbl.schema.field(idx),
            pa.array(values, type=out_tbl.schema.field(idx).type))
        for k, lv in enumerate(QLEVELS):
            qv = np.full(len(template), np.nan)
            qv[template_row] = np.clip(quant[:, j, k], 40.0, 400.0)
            side[f"q{int(round(lv * 100)):02d}_{h}"] = qv
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_tbl, a.out)
    side.to_parquet(a.out.with_suffix("").with_name(
        a.out.stem + ".quantiles.parquet"), index=False)
    print(f"wrote {a.out} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
