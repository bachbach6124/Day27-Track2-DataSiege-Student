"""
Tiered defense: tightened baseline bounds, structural lineage checks, and
per-metric sliding-window z-score alerts (single-metric batch trigger).
"""
from api import Verdict

EXPECTED_UPSTREAM = ["raw.orders", "raw.customers"]
ROLLING_MIN_SAMPLES = 7
BATCH_COMBO_THRESHOLD = 1

# Per-metric z-score cutoffs (tuned via stream self-history)
Z = {
    "batch_row": 3.25,
    "batch_mean": 2.22,
    "batch_null": 2.62,
    "batch_stale": 2.89,
    "batch_std": 1.12,
    "embed_cent": 1.3,
    "embed_age": 3.5,
}


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


class _RunningStats:
    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def zscore(self, value):
        if self.n < ROLLING_MIN_SAMPLES:
            return None
        var = self.m2 / self.n
        if var <= 1e-12:
            return None
        return abs(value - self.mean) / (var ** 0.5)


def _stats(ctx, key):
    bucket = ctx.state.setdefault("rolling", {})
    if key not in bucket:
        bucket[key] = _RunningStats()
    return bucket[key]


def _record_clean(ctx, key, value):
    _stats(ctx, key).update(value)


def _z_spike(ctx, key, value, threshold=None):
    z = _stats(ctx, key).zscore(value)
    limit = Z[key] if threshold is None else threshold
    return z is not None and z > limit


def _mean_amount_upper(bl):
    span = bl["mean_amount_max"] - bl["mean_amount_min"]
    return bl["mean_amount_min"] + span * 0.91


def _feature_sigma_threshold(bl):
    return max(bl["feature_mean_shift_sigma_max"] * 1.2, 0.5)


def _embedding_centroid_threshold(bl):
    return bl["embedding_centroid_shift_max"] * 0.897


def _corpus_age_threshold(bl):
    return bl["corpus_avg_doc_age_days_max"] * 0.60


def _lineage_duration_threshold(bl):
    return bl["lineage_duration_ms_max"] * 0.87


def _staleness_threshold(bl):
    return bl["staleness_min_max"] * 0.93


def _rolling_data_batch(profile, ctx):
    metrics = (
        ("batch_row", profile["row_count"]),
        ("batch_mean", profile["mean_amount"]),
        ("batch_null", profile["null_rate"].get("customer_id", 0)),
        ("batch_stale", profile["staleness_min"]),
        ("batch_std", profile["std_amount"]),
    )
    spikes = sum(1 for key, value in metrics if _z_spike(ctx, key, value))
    return spikes >= BATCH_COMBO_THRESHOLD


def _update_data_batch_stats(profile, ctx):
    _record_clean(ctx, "batch_row", profile["row_count"])
    _record_clean(ctx, "batch_mean", profile["mean_amount"])
    _record_clean(ctx, "batch_null", profile["null_rate"].get("customer_id", 0))
    _record_clean(ctx, "batch_stale", profile["staleness_min"])
    _record_clean(ctx, "batch_std", profile["std_amount"])


def check_data_batch(payload, ctx):
    bl = ctx.baseline
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if profile.get("error"):
        return Verdict(alert=False, pillar="checks", reason="profile unavailable")

    reasons = []
    if profile["row_count"] < bl["row_count_min"] or profile["row_count"] > bl["row_count_max"]:
        reasons.append("row_count")
    null_rate = profile["null_rate"].get("customer_id", 0)
    if null_rate > bl["null_rate_max"]:
        reasons.append("null_rate")
    mean_hi = _mean_amount_upper(bl)
    if profile["mean_amount"] < bl["mean_amount_min"] or profile["mean_amount"] > mean_hi:
        reasons.append("mean_amount")
    if profile["staleness_min"] > _staleness_threshold(bl):
        reasons.append("staleness")

    if not reasons and _rolling_data_batch(profile, ctx):
        reasons.append("rolling_anomaly")

    if not reasons:
        _update_data_batch_stats(profile, ctx)

    return Verdict(alert=bool(reasons), pillar="checks", reason=",".join(reasons))


def check_contract_checkpoint(payload, ctx):
    bl = ctx.baseline
    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if diff.get("error"):
        return Verdict(alert=False, pillar="contracts", reason="diff unavailable")

    reasons = list(diff.get("violations", []))
    if diff["freshness_delay_min"] > bl["freshness_delay_max_min"]:
        reasons.append("freshness_delay")

    return Verdict(alert=bool(reasons), pillar="contracts", reason=",".join(reasons))


def check_lineage_run(payload, ctx):
    bl = ctx.baseline
    graph = ctx.tools.lineage_graph_slice(payload["run_id"])
    if graph.get("error"):
        return Verdict(alert=False, pillar="lineage", reason="graph unavailable")

    reasons = []
    if graph["duration_ms"] > _lineage_duration_threshold(bl):
        reasons.append("duration")
    if list(graph["actual_upstream"]) != EXPECTED_UPSTREAM:
        reasons.append("upstream")
    if graph["actual_downstream_count"] == 0:
        reasons.append("orphan_output")

    return Verdict(alert=bool(reasons), pillar="lineage", reason=",".join(reasons))


def check_feature_materialization(payload, ctx):
    bl = ctx.baseline
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if drift.get("error"):
        return Verdict(alert=False, pillar="ai_infra", reason="drift unavailable")

    if drift["mean_shift_sigma"] > _feature_sigma_threshold(bl):
        return Verdict(alert=True, pillar="ai_infra", reason="feature_skew")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    bl = ctx.baseline
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if drift.get("error"):
        return Verdict(alert=False, pillar="ai_infra", reason="drift unavailable")

    corpus = payload["corpus"]
    cent_key = f"embed_cent:{corpus}"
    age_key = f"embed_age:{corpus}"

    reasons = []
    if drift["centroid_shift"] > _embedding_centroid_threshold(bl):
        reasons.append("centroid_shift")
    if drift["avg_doc_age_days"] > _corpus_age_threshold(bl):
        reasons.append("corpus_staleness")

    if not reasons:
        if _z_spike(ctx, cent_key, drift["centroid_shift"], Z["embed_cent"]):
            reasons.append("centroid_shift")
        elif _z_spike(ctx, age_key, drift["avg_doc_age_days"], Z["embed_age"]):
            reasons.append("corpus_staleness")

    if not reasons:
        _record_clean(ctx, cent_key, drift["centroid_shift"])
        _record_clean(ctx, age_key, drift["avg_doc_age_days"])

    return Verdict(alert=bool(reasons), pillar="ai_infra", reason=",".join(reasons))
