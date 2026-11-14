"""Design-based estimation for an executed device-cluster sample.

The estimator matches the plan's design:

* every selected device contributes with weight ``1 / pi_d = M_h / n_h``;
* a metric is a ratio of Horvitz-Thompson totals
  ``R = sum_d w_d t_d / sum_d w_d a_d``, where ``a_d`` counts the device's rows
  in the metric domain and ``t_d`` sums the field over them;
* the variance is the first-order cluster variance
  ``sum_h (M_h/A)^2 (1 - n_h/M_h) S2_h / n_h`` with ``S2_h`` the per-device
  sample variance of ``z_d = t_d - R a_d``;
* if a stratum has fewer than two selected devices, or the sample variance is
  exactly zero, the variance frozen into the plan is used instead, and the
  component records which source was used.  A stratum with zero observed
  failures therefore keeps the positive Clopper-Pearson floor; zero observed
  failures is never treated as zero variance;
* the interval is ``R +/- t(0.975, df) SE`` with Satterthwaite ``df``.

The plan is used for the frozen priors and to check the delivered inclusion
probabilities; nothing from the design fixtures is re-read.
"""

import collections
import math

from .common import PlanError, satterthwaite_df, t_quantile
from .design import verify_plan


def _metric_plan(plan, metric_id):
    for metric in plan['metrics']:
        if metric['id'] == metric_id:
            priors = {}
            for component in metric.get('stratum_components', []):
                priors[component['stratum']] = {
                    'variance': float(component.get('residual_variance_prior', 0.0)),
                    'df': max(1, int(component.get('pilot_devices', 1)) - 1),
                    'floor_kind': component.get('variance_floor_kind'),
                    'domain_records': int(component.get('domain_records', 0)),
                }
            return metric, priors
    raise PlanError('计划缺少指标：' + metric_id)


def _device_totals(sample_rows, metric):
    domain = metric['domain']
    field = metric['field']
    totals = collections.defaultdict(lambda: {'a': 0.0, 't': 0.0, 'weight': None,
                                              'stratum': None, 'probability': None})
    for row in sample_rows:
        device_id = row.get('selection_unit_id')
        if device_id is None:
            raise PlanError('样本缺少 selection_unit_id')
        probability = row.get('inclusion_probability')
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise PlanError('样本包含概率无效')
        if not math.isfinite(probability) or probability <= 0:
            raise PlanError('样本包含概率必须大于零')
        entry = totals[device_id]
        if entry['weight'] is None:
            entry['weight'] = 1.0 / probability
            entry['probability'] = float(probability)
            entry['stratum'] = row.get('stratum')
        elif not math.isclose(entry['weight'], 1.0 / probability,
                              rel_tol=0, abs_tol=1e-12):
            raise PlanError(f'设备 {device_id} 的包含概率不一致')
        if all(row.get(key) == value for key, value in domain.items()):
            entry['a'] += 1.0
            entry['t'] += float(row[field])
    return totals


def _stratum_frames(execution):
    return {report['stratum']: report for report in execution.get('strata', [])}


def selection_probability(selected, frame):
    if frame <= 0:
        return None
    return selected / frame


def _check_probabilities(sample_rows, frames):
    by_stratum = collections.defaultdict(set)
    for row in sample_rows:
        by_stratum[row.get('stratum')].add(row.get('inclusion_probability'))
    for stratum, report in frames.items():
        expected = selection_probability(report['selected_devices'], report['frame_devices'])
        delivered = by_stratum.get(stratum, set())
        if report['selected_devices'] == 0:
            continue
        if expected is None or len(delivered) != 1:
            raise PlanError('样本包含概率与执行分层不一致：' + stratum)
        value = next(iter(delivered))
        if not math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-12):
            raise PlanError('样本包含概率与执行分层不一致：' + stratum)


def _design_variance(selected, frame, denominator, used_variance):
    if selected <= 0 or frame <= 0 or denominator <= 0:
        return 0.0
    return (frame / denominator) ** 2 * (1.0 - selected / frame) / selected \
        * used_variance


def _logit_interval(value, standard_error, df):
    """Bounded interval for a rate: delta method on the logit scale.

    The clustered variance can make a linear interval exceed [0, 1].  For a
    binary metric the logit transform keeps the interval inside the parameter
    space and behaves better when the rate is small, without pretending the
    sampling design is simple random sampling.
    """
    if not 0.0 < value < 1.0:
        raise ValueError('logit 变换需要严格在 (0,1) 内的估计')
    logit = math.log(value / (1.0 - value))
    logit_se = standard_error / (value * (1.0 - value))
    width = t_quantile(0.975, df) * logit_se
    lower = 1.0 / (1.0 + math.exp(-(logit - width)))
    upper = 1.0 / (1.0 + math.exp(-(logit + width)))
    return lower, upper, width


def estimate_metric(execution, plan, metric_definition):
    metric_id = metric_definition['id']
    metric, priors = _metric_plan(plan, metric_id)
    totals = _device_totals(execution['sample'], metric)
    frames = _stratum_frames(execution)

    numerator = sum(entry['weight'] * entry['t'] for entry in totals.values())
    denominator = sum(entry['weight'] * entry['a'] for entry in totals.values())
    if denominator <= 0:
        return {
            'metric': metric_id,
            'status': 'undefined_no_domain_rows',
            'estimate': None,
            'standard_error': None,
            'half_width': None,
            'interval_lower': None,
            'interval_upper': None,
            'degrees_of_freedom': None,
            'numerator_estimate': numerator,
            'denominator_estimate': denominator,
            'strata': [],
            'notes': ['样本中该指标域没有记录，未定义而不是零'],
        }
    value = numerator / denominator
    zero_event_note = []
    if numerator == 0:
        zero_event_note.append(
            '样本中该指标域零事件；区间由计划先验支撑，未把零事件当作零方差')

    z_by_stratum = collections.defaultdict(dict)
    for device_id, entry in totals.items():
        z_by_stratum[entry['stratum']][device_id] = entry['t'] - value * entry['a']

    components = []
    for stratum in sorted(frames):
        report = frames[stratum]
        selected = int(report['selected_devices'])
        frame = int(report['frame_devices'])
        z_values = list(z_by_stratum.get(stratum, {}).values())
        prior = priors.get(stratum, {'variance': 0.0, 'df': 1, 'floor_kind': None,
                                     'domain_records': 0})
        sample_df = max(0, len(z_values) - 1)
        sample_variance = None
        if sample_df >= 1:
            mean = sum(z_values) / len(z_values)
            sample_variance = sum((value - mean) ** 2 for value in z_values) / sample_df
        if sample_variance is not None and sample_variance > 0.0:
            used, source, df = sample_variance, 'sample', sample_df
        elif prior['domain_records'] <= 0:
            # No device in this stratum can ever belong to the metric domain.
            used, source, df = 0.0, 'stratum_outside_metric_domain', 1
        elif prior['variance'] > 0.0:
            used = prior['variance']
            source = ('plan_prior_floor' if sample_variance is not None and sample_variance == 0.0
                      else 'plan_prior_single_device')
            df = prior['df']
        else:
            used, source, df = None, 'undefined_insufficient_variance', None
        contribution = None if used is None else _design_variance(
            selected, frame, denominator, used)
        components.append({
            'stratum': stratum,
            'frame_devices': frame,
            'selected_devices': selected,
            'inclusion_probability': selection_probability(selected, frame),
            'device_weight': (frame / selected) if selected else None,
            'sample_df': sample_df,
            'sample_variance': sample_variance,
            'variance_used': used,
            'variance_source': source,
            'degrees_of_freedom_used': df,
            'variance_contribution': contribution,
        })

    undefined = [component['stratum'] for component in components
                 if component['variance_used'] is None]
    if undefined:
        return {
            'metric': metric_id,
            'status': 'undefined_variance',
            'estimate': value,
            'standard_error': None,
            'half_width': None,
            'interval_lower': None,
            'interval_upper': None,
            'degrees_of_freedom': None,
            'numerator_estimate': numerator,
            'denominator_estimate': denominator,
            'strata': components,
            'notes': ['以下分层无法给出方差（少于两台设备且计划无先验）：'
                      + ', '.join(undefined)],
        }

    variance = sum(component['variance_contribution'] for component in components)
    parts = [(component['variance_contribution'], component['degrees_of_freedom_used'])
             for component in components if component['variance_contribution'] > 0.0]
    df = satterthwaite_df(parts)
    standard_error = math.sqrt(max(variance, 0.0))
    multiplier = t_quantile(0.975, df)
    linear_half_width = multiplier * standard_error
    interval_method = 'cluster-t-linear'
    half_width = linear_half_width
    interval_lower = value - linear_half_width
    interval_upper = value + linear_half_width
    if metric.get('binary') and 0.0 < value < 1.0:
        try:
            interval_lower, interval_upper, logit_width = _logit_interval(
                value, standard_error, df)
            interval_method = 'cluster-t-logit'
            half_width = (interval_upper - interval_lower) / 2.0
        except (ValueError, OverflowError):
            pass
    return {
        'metric': metric_id,
        'status': 'ok',
        'estimate': value,
        'standard_error': standard_error,
        'half_width': half_width,
        'interval_lower': max(0.0, interval_lower),
        'interval_upper': min(1.0, interval_upper) if metric.get('binary')
        else interval_upper,
        'interval_method': interval_method,
        'degrees_of_freedom': df,
        'zero_events_in_sample': numerator == 0,
        'numerator_estimate': numerator,
        'denominator_estimate': denominator,
        'strata': components,
        'notes': zero_event_note,
    }


def estimate(execution, plan):
    verify_plan(plan)
    if not isinstance(execution, dict) \
            or execution.get('schema') != 'telemetry-sample-execution/1':
        raise PlanError('estimate 的输入必须是 sample 写出的执行结果')
    if execution.get('plan_fingerprint') != plan.get('plan_fingerprint'):
        raise PlanError('执行结果的计划指纹与 --plan 不一致')
    frames = _stratum_frames(execution)
    _check_probabilities(execution['sample'], frames)

    metric_entries = [estimate_metric(execution, plan, metric)
                      for metric in plan['metrics']]
    return {
        'schema': 'telemetry-sample-estimate/1',
        'plan_fingerprint': plan['plan_fingerprint'],
        'seed': execution.get('seed'),
        'spent': execution.get('spent'),
        'budget': execution.get('budget'),
        'within_budget': execution.get('within_budget'),
        'metrics': metric_entries,
        'summary': {
            'metrics_estimated': [entry['metric'] for entry in metric_entries
                                  if entry['status'] == 'ok'],
            'metrics_undefined': [entry['metric'] for entry in metric_entries
                                  if entry['status'] != 'ok'],
            'estimation_design': 'horvitz-thompson-ratio-of-means-device-cluster',
        },
    }
