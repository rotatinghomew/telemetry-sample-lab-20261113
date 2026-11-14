"""Repeated-replay validation of a plan against local replay material.

``replay_population`` may only be used here, after the plan is saved.  This
module never feeds replay results back into the design: it takes a *saved*
plan file, re-reads it, and for each seed compares the achieved estimate with
the known replay value under the design-based interval.

The comparison against the uniform-1% baseline uses the same replay rows and
the same ratio estimator, with independent-record Bernoulli sampling at
``baseline_probability``.  Because that design samples rows independently, its
interval is the simple-random-sampling interval for a proportion (rows are the
selection unit), while the clustered plan uses the device-cluster interval.
This is exactly the point: applying the SRS formula to the cluster design (or
the reverse) would misstate coverage.
"""

import math

import numpy as np

from .design import verify_plan
from .estimate import estimate
from .execute import sample
from .sampling import uniform_draw


def true_metric_value(rows, metric):
    domain = metric['domain']
    field = metric['field']
    selected = [row for row in rows
                if all(row.get(key) == value for key, value in domain.items())]
    if not selected:
        return None
    return sum(float(row[field]) for row in selected) / len(selected)


def baseline_sample(rows, probability, seed):
    selected = [row for row in rows if uniform_draw(seed, row['row_id']) < probability]
    return selected


def baseline_estimate(rows, metric, probability, seed):
    """Row-level Bernoulli sample with an appropriate SRS interval.

    The baseline scheme selects rows independently with probability ``p``;
    rows are the selection unit and the domain ratio is estimated as the
    proportion of "field=1" rows among sampled domain rows.  The interval is
    the Wilson score interval for a binomial proportion, not a zero-width
    Wald interval: with zero sampled events the honest statement is a wide
    interval, and replacing it by zero would repeat exactly the mistake the
    plan avoids.
    """
    domain = metric['domain']
    field = metric['field']
    sample_rows = baseline_sample(rows, probability, seed)
    domain_rows = [row for row in sample_rows
                   if all(row.get(key) == value for key, value in domain.items())]
    if not domain_rows:
        return {'status': 'undefined_no_domain_rows', 'estimate': None,
                'half_width': None, 'sampled_rows': len(sample_rows),
                'sampled_domain_rows': 0,
                'spent': sum(int(row['cost']) for row in sample_rows)}
    denominator = len(domain_rows)
    numerator = sum(float(row[field]) for row in domain_rows)
    value = numerator / denominator
    lower, upper = _wilson(value, denominator)
    half_width = (upper - lower) / 2.0
    return {
        'status': 'ok',
        'estimate': value,
        'standard_error': math.sqrt(value * (1.0 - value) / denominator)
        if 0.0 < value < 1.0 else None,
        'half_width': half_width,
        'interval_lower': lower,
        'interval_upper': upper,
        'interval_method': 'wilson-score',
        'sampled_rows': len(sample_rows),
        'sampled_domain_rows': denominator,
        'population_domain_rows': sum(
            1 for row in rows
            if all(row.get(key) == value for key, value in domain.items())),
        'spent': sum(int(row['cost']) for row in sample_rows),
    }


def _wilson(proportion, trials, z=1.959963984540054):
    if trials <= 0:
        return 0.0, 1.0
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials
                           + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _wilson_list(proportion, trials, z=1.959963984540054):
    lower, upper = _wilson(proportion, trials, z)
    return [lower, upper]


def replay(plan, document, seeds, baseline_probability=None):
    verify_plan(plan)
    rows = document.get('replay_population')
    if rows is None:
        rows = document.get('population_rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('重放检查需要非空的 replay_population 或 population_rows')
    metrics = plan['metrics']
    truth = {metric['id']: true_metric_value(rows, metric) for metric in metrics}
    if baseline_probability is None:
        baseline_probability = document.get('baseline_probability')
    if (isinstance(baseline_probability, bool)
            or not isinstance(baseline_probability, (int, float))
            or not 0 < baseline_probability <= 1):
        raise ValueError('baseline_probability 必须在零到一之间')

    plan_records = {metric['id']: [] for metric in metrics}
    baseline_records = {metric['id']: [] for metric in metrics}
    spends = []
    within_budget = []
    baseline_spends = []
    baseline_rows = []
    for seed in seeds:
        execution = sample(document, plan, seed=seed)
        spends.append(int(execution['spent']))
        within_budget.append(bool(execution['within_budget']))
        result = estimate(execution, plan)
        for entry in result['metrics']:
            plan_records[entry['metric']].append(entry)
        base_spent = None
        base_rows = None
        for metric in metrics:
            record = baseline_estimate(rows, metric, baseline_probability, seed)
            baseline_records[metric['id']].append(record)
            if base_spent is None:
                base_spent = record['spent']
                base_rows = record['sampled_rows']
        baseline_spends.append(base_spent)
        baseline_rows.append(base_rows)

    plan_summary = {metric['id']: _summarise(plan_records[metric['id']], truth[metric['id']],
                                            'design')
                    for metric in metrics}
    baseline_summary = {metric['id']: _summarise(baseline_records[metric['id']],
                                                truth[metric['id']], 'srs')
                        for metric in metrics}
    return {
        'schema': 'telemetry-sample-replay/1',
        'plan_fingerprint': plan['plan_fingerprint'],
        'seeds': list(seeds),
        'truth_from_replay': truth,
        'budget': plan['budget'],
        'spend': {
            'min': min(spends) if spends else None,
            'max': max(spends) if spends else None,
            'mean': (sum(spends) / len(spends)) if spends else None,
            'within_budget_all_seeds': all(within_budget),
            'hard_violations': sum(1 for value in within_budget if not value),
        },
        'plan': plan_summary,
        'baseline_uniform': {
            'probability': baseline_probability,
            'selection_unit': 'row_id (independent Bernoulli)',
            'interval_method': 'wilson-score for sampled domain rows',
            'spend': {
                'min': min(baseline_spends) if baseline_spends else None,
                'max': max(baseline_spends) if baseline_spends else None,
                'mean': (sum(baseline_spends) / len(baseline_spends))
                        if baseline_spends else None,
                'mean_sampled_rows': (sum(baseline_rows) / len(baseline_rows))
                                     if baseline_rows else None,
            },
            'metrics': baseline_summary,
        },
        'comparison': _comparison(plan_summary, baseline_summary),
        'metrics': metrics,
    }


def _comparison(plan_summary, baseline_summary):
    rows = []
    for metric_id in plan_summary:
        plan = plan_summary[metric_id]
        base = baseline_summary.get(metric_id, {})
        rows.append({
            'metric': metric_id,
            'plan_defined_rate': (plan['defined_replays'] / plan['replays']
                                  if plan['replays'] else None),
            'baseline_defined_rate': (base['defined_replays'] / base['replays']
                                      if base and base['replays'] else None),
            'plan_median_half_width': plan.get('median_half_width'),
            'baseline_median_half_width': base.get('median_half_width'),
            'plan_coverage': plan.get('coverage'),
            'baseline_coverage': base.get('coverage'),
            'plan_rmse': plan.get('rmse'),
            'baseline_rmse': base.get('rmse'),
        })
    return rows


def _summarise(records, truth, design_kind):
    defined = [entry for entry in records if entry.get('status') == 'ok']
    summary = {
        'design_kind': design_kind,
        'replays': len(records),
        'defined_replays': len(defined),
        'undefined_replays': len(records) - len(defined),
        'undefined_reasons': sorted({entry.get('status') for entry in records
                                     if entry.get('status') != 'ok'}),
    }
    if not defined or truth is None:
        summary.update({'bias': None, 'rmse': None, 'median_half_width': None,
                        'coverage': None, 'coverage_ci95': None})
        return summary
    estimates = np.array([entry['estimate'] for entry in defined], dtype=float)
    half_widths = np.array([entry['half_width'] for entry in defined], dtype=float)
    lower = np.array([entry['interval_lower'] for entry in defined], dtype=float)
    upper = np.array([entry['interval_upper'] for entry in defined], dtype=float)
    if not (np.all(np.isfinite(half_widths)) and np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))):
        raise ValueError('估计区间出现非有限值')
    bias = float(estimates.mean() - truth)
    rmse = float(math.sqrt(np.mean((estimates - truth) ** 2)))
    covered = (truth >= lower) & (truth <= upper)
    summary.update({
        'mean_estimate': float(estimates.mean()),
        'bias': bias,
        'rmse': rmse,
        'mean_standard_error': (float(np.mean([entry['standard_error']
                                               for entry in defined]))
                                if all(entry['standard_error'] is not None
                                       for entry in defined) else None),
        'median_half_width': float(np.median(half_widths)),
        'coverage': float(covered.mean()),
        'coverage_ci95': _wilson_list(float(covered.mean()), len(defined)),
    })
    return summary

