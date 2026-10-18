"""Annual-competition prediction: ``submission.predict`` plus one statics harmonisation step.

    python -m metabonet_forecasting.submission.predict_annual \
        --run runs/c96a_transformerpp \
        --store artifacts/store/submission_annual \
        --template toolkit/data/annual_competition/template.parquet \
        --out artifacts/submission/annual_c96a_transformerpp.parquet

Same arguments and guarantees as ``predict.py``. The one change: after the categorical codes
are aligned, each annual patient's statics are put in the pattern every IOBP2 patient had in
training, because IOBP2 is the only source that matches the annual set:

  gender            -> unknown (-1)        IOBP2 ships no gender; all 332 trained as unknown
  diabetes_duration -> imputed from age    the annual template withholds it (100% null)
  age_of_diagnosis  -> age - imputed duration

Why it is needed. The annual template ships gender, but in training "gender unknown" is close
to an IOBP2 fingerprint (IOBP2 is ~66% of all unknown-gender patients), and models learn it
even with ``use_source_identity=false``. Feeding the annual gender as-is moved a patient off
the pattern the IOBP2-specialised models were trained on and cost 3.95-12.08 zone-A points on
the IOBP2 ruler; withholding the diagnosis fields cost ~0. Harmonised, the 3-model ensemble
projected ~95.7 vs ~91.1 unharmonised.

The imputation curve is the band medians of the 270 IOBP2 patients that are not on the live
leaderboard, so nothing here is fitted on a scored patient.
"""
from __future__ import annotations

import numpy as np

from . import predict as P

AGE_PTS = [8, 12, 16, 20, 35, 50, 64]
DUR_PTS = [5, 5, 9, 12, 20, 32, 36]

_align_categoricals = P.align_categoricals


def align_and_harmonize(sub_store, train_store) -> None:
    _align_categoricals(sub_store, train_store)
    st = np.array(sub_store.statics, dtype=np.float32)
    names = list(sub_store.manifest["static_numeric"]) + list(sub_store.manifest["static_categorical"])
    ia, iad, idd, ig = (names.index(k) for k in ("age", "age_of_diagnosis", "diabetes_duration", "gender"))
    age = st[:, ia]
    dur = np.minimum(np.interp(age, AGE_PTS, DUR_PTS), np.maximum(age - 1.0, 0.0))
    if not (np.isnan(st[:, idd]).all() and np.isnan(st[:, iad]).all()):
        raise ValueError("expected the annual template to withhold diabetes_duration and "
                         "age_of_diagnosis; this harmonisation is only valid for that set")
    st[:, idd] = dur
    st[:, iad] = age - dur
    st[:, ig] = -1.0
    sub_store.statics = st
    P.LOG.info(f"harmonized statics for {len(st)} subjects: gender->unknown, duration imputed "
               f"(median {np.median(dur):.1f} y)")


def main(argv=None) -> int:
    # predict_template looks align_categoricals up as a module global at call time, so
    # rebinding it here is the whole change -- the inference path is otherwise byte-identical.
    P.align_categoricals = align_and_harmonize
    try:
        return P.main(argv)
    finally:
        P.align_categoricals = _align_categoricals


if __name__ == "__main__":
    raise SystemExit(main())
