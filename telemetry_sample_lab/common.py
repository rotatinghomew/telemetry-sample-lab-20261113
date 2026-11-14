"""Deterministic helpers shared by the planner, executor and replay checks."""

import hashlib
import json
import math

import numpy as np
from scipy.stats import beta, chi2


class PlanError(ValueError):
    """Raised when a plan, an input or an output cannot be used as written."""


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def plan_fingerprint(plan):
    """Fingerprint of every plan field except the fingerprint itself."""
    payload = {key: value for key, value in plan.items() if key != 'plan_fingerprint'}
    return sha256_text(canonical_json(payload))


DESIGN_KEYS = ('seed', 'baseline_probability', 'population_sizes', 'budget', 'metrics', 'pilot')


def design_input_digest(document):
    """Hash only design-time facts.

    ``replay_population`` (the future results) is deliberately excluded so a
    saved plan can be checked without, and independently of, that key.
    """
    missing = [key for key in DESIGN_KEYS if key not in document]
    if missing:
        raise PlanError('设计输入缺少字段：' + ', '.join(missing))
    subset = {key: document[key] for key in DESIGN_KEYS}
    return sha256_text(canonical_json(subset))


def round_half_up(value):
    return int(math.floor(value + 0.5))


def domain_mask(row, domain):
    return all(row.get(key) == value for key, value in domain.items())


def device_inclusion_probability(selected, frame_devices):
    if frame_devices <= 0:
        return None
    return selected / frame_devices


def binomial_variance_floor(a_mean, domain_records, events, confidence=0.90):
    """Positive planning variance when the pilot cannot show variation.

    A stratum with zero observed events is *not* evidence of zero variance.
    The event rate is bounded with a one-sided Clopper-Pearson bound and the
    bound is translated to the variance of a device-level total
    ``t_d = sum(field)`` over the device's domain rows.

    Returns ``(floor, kind, bounds)``; ``bounds`` are on the event-rate scale.
    """
    if domain_records <= 0 or a_mean <= 0:
        return 0.0, 'no_domain_records', {'lower': None, 'upper': None}
    events = int(events)
    domain_records = int(domain_records)
    if events <= 0:
        upper = float(beta.ppf(confidence, 1, domain_records + 1))
        return a_mean * a_mean * upper * (1.0 - upper), 'zero_events_floor', {
            'lower': 0.0, 'upper': upper}
    if events >= domain_records:
        lower = float(beta.ppf(1.0 - confidence, domain_records, 1))
        return a_mean * a_mean * lower * (1.0 - lower), 'all_events_floor', {
            'lower': lower, 'upper': 1.0}
    lower = float(beta.ppf(1.0 - confidence, events, domain_records - events + 1))
    upper = float(beta.ppf(confidence, events + 1, domain_records - events + 1))
    return (a_mean * a_mean * max(upper * (1.0 - upper), lower * (1.0 - lower)),
            'binomial_floor', {'lower': lower, 'upper': upper})


def adjusted_variance(values, confidence=0.90, ddof=1):
    """Upper chi-square bound for a pilot variance.

    Returns ``(adjusted, raw, factor, df)``.  With too few clusters the pilot
    cannot estimate variance and ``adjusted`` is zero, so callers must add the
    binomial floor.
    """
    array = np.asarray(values, dtype=float)
    n = array.size
    if n <= ddof:
        return 0.0, 0.0, 1.0, max(0, n - ddof)
    raw = float(np.var(array, ddof=ddof))
    df = n - ddof
    factor = df / float(chi2.ppf(1.0 - confidence, df))
    return raw * factor, raw, factor, df


def satterthwaite_df(parts):
    """Satterthwaite df for ``sum v_h`` with component df ``d_h``."""
    if not parts:
        return 1.0
    numerator = sum(value for value, _ in parts) ** 2
    denominator = sum(value * value / df for value, df in parts if value > 0 and df > 0)
    if denominator <= 0:
        return 1.0
    return max(1.0, numerator / denominator)


def t_quantile(probability, df):
    from scipy.stats import t as student_t
    return float(student_t.ppf(probability, max(df, 1.0)))
