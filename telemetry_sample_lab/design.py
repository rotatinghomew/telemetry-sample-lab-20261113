"""Build an executable multi-metric device-cluster sampling plan.

Design-time inputs only (``fixtures/design.json``): ``pilot``,
``population_sizes``, ``budget``, ``metrics``, ``seed``,
``baseline_probability``.  The future ``replay_population`` is never read;
``design_input_digest`` is computed from the design keys alone, so a saved
plan can be audited without the future results.

Design in one paragraph
-----------------------
Devices are the selection unit and are stratified by ``(tier, carries a
rare-app row)``.  Every row of a selected device is uploaded, so a row's
inclusion probability equals its device's and rows within a device are
correlated.  Next-period device counts are estimated from pilot device counts
scaled by the tier-level ratio of next-period record totals to pilot record
totals.  For each metric (a ratio of means over a domain) the planning variance
of the device-level residual ``z_d = t_d - R a_d`` is the upper chi-square bound
of the pilot variance; when the pilot shows no spread at all the bound is
replaced by a one-sided Clopper-Pearson binomial floor, so a stratum with zero
observed failures keeps positive variance.  Allocation is an exhaustive integer
search over device counts; every stratum keeps at least ``min_devices``
devices.  Priorities are lexicographic (worst ratio, then sum of squared
ratios, per priority level), then cost.  Budget costs use the worst-case
per-device cost seen in the pilot stratum, so a ``hard`` budget is never
exceeded whatever the seed draws.
"""

import math

import numpy as np

from .common import (PlanError, adjusted_variance, binomial_variance_floor,
                     design_input_digest, device_inclusion_probability, domain_mask,
                     plan_fingerprint, round_half_up, satterthwaite_df, t_quantile)

DESIGN_CONFIDENCE = 0.90
DEFAULT_MIN_DEVICES = 2
SEARCH_LIMIT = 1_500_000


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_metrics(metrics):
    if not isinstance(metrics, list) or not metrics:
        raise PlanError('metrics 必须是非空数组')
    seen = set()
    for metric in metrics:
        if not isinstance(metric, dict):
            raise PlanError('指标必须是对象')
        for key in ('id', 'field', 'domain', 'half_width', 'priority'):
            if key not in metric:
                raise PlanError('指标缺少字段：' + key)
        if metric['id'] in seen:
            raise PlanError('指标编号重复：' + str(metric['id']))
        seen.add(metric['id'])
        half_width = metric['half_width']
        if isinstance(half_width, bool) or not isinstance(half_width, (int, float)):
            raise PlanError('指标 half_width 必须是数值')
        if not math.isfinite(half_width) or not 0 < half_width < 1:
            raise PlanError('指标 half_width 必须在零到一之间')
        if isinstance(metric['priority'], bool) or not isinstance(metric['priority'], int):
            raise PlanError('指标 priority 必须是整数')
        if not isinstance(metric['domain'], dict):
            raise PlanError('指标 domain 必须是对象')
    return metrics


def validate_budget(budget):
    if not isinstance(budget, dict):
        raise PlanError('budget 必须是对象')
    amount = budget.get('amount')
    if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
        raise PlanError('budget.amount 必须是正整数')
    mode = budget.get('mode', 'hard')
    if mode not in ('hard', 'expected'):
        raise PlanError('budget.mode 必须是 hard 或 expected')
    return {'amount': amount, 'mode': mode}


def validate_population_sizes(sizes):
    if not isinstance(sizes, list) or not sizes:
        raise PlanError('population_sizes 必须是非空数组')
    result = {}
    for entry in sizes:
        if not isinstance(entry, dict) or 'tier' not in entry or 'app' not in entry:
            raise PlanError('population_sizes 条目必须含 tier 和 app')
        count = entry.get('estimated_records')
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PlanError('estimated_records 必须是非负整数')
        key = (entry['tier'], entry['app'])
        if key in result:
            raise PlanError(f'population_sizes 重复：{key}')
        result[key] = count
    return result


# ---------------------------------------------------------------------------
# Pilot summaries
# ---------------------------------------------------------------------------

def stratum_key(tier, has_rare):
    return f'{tier}|{"rare" if has_rare else "standard"}'


def build_device_frame(pilot_rows, metrics):
    devices = {}
    for row in pilot_rows:
        devices.setdefault(row['device_id'], []).append(row)
    rare_apps = sorted({value for metric in metrics
                        for key, value in metric['domain'].items() if key == 'app'})
    frame = []
    for device_id in sorted(devices):
        rows = devices[device_id]
        tiers = {row['tier'] for row in rows}
        if len(tiers) != 1:
            raise PlanError(f'设备 {device_id} 跨多个 tier，无法稳定分层')
        tier = tiers.pop()
        has_rare = any(row['app'] in rare_apps for row in rows)
        frame.append({
            'device_id': device_id,
            'rows': rows,
            'tier': tier,
            'has_rare_app': has_rare,
            'stratum': stratum_key(tier, has_rare),
            'cost': sum(int(row['cost']) for row in rows),
        })
    if not frame:
        raise PlanError('pilot 为空，无法设计')
    return frame, rare_apps


def fit_strata(frame, population_sizes, pilot_rows):
    pilot_tier_records = {}
    for row in pilot_rows:
        pilot_tier_records[row['tier']] = pilot_tier_records.get(row['tier'], 0) + 1
    next_tier_records = {}
    for (tier, _app), count in population_sizes.items():
        next_tier_records[tier] = next_tier_records.get(tier, 0) + count
    strata = {}
    for device in frame:
        entry = strata.setdefault(device['stratum'], {
            'id': device['stratum'],
            'tier': device['tier'],
            'has_rare_app': device['has_rare_app'],
            'devices': [],
        })
        entry['devices'].append(device)
    for key, stratum in strata.items():
        tier = stratum['tier']
        if tier not in next_tier_records:
            raise PlanError(f'population_sizes 缺少 tier：{tier}')
        if pilot_tier_records.get(tier, 0) <= 0:
            raise PlanError(f'pilot 缺少 tier：{tier}')
        scale = next_tier_records[tier] / pilot_tier_records[tier]
        pilot_devices = len(stratum['devices'])
        exact = pilot_devices * scale
        stratum['scale'] = scale
        stratum['pilot_devices'] = pilot_devices
        stratum['estimated_devices_exact'] = exact
        stratum['estimated_devices'] = max(1, round_half_up(exact))
        stratum['records_per_device'] = float(np.mean(
            [len(device['rows']) for device in stratum['devices']]))
        stratum['cost_bound'] = max(device['cost'] for device in stratum['devices'])
        stratum['cost_mean'] = float(np.mean(
            [device['cost'] for device in stratum['devices']]))
        stratum['estimated_records'] = round_half_up(
            stratum['estimated_devices'] * stratum['records_per_device'])
    return strata


def metric_components(metric, strata, pilot):
    domain = metric['domain']
    field = metric['field']
    domain_rows = [row for row in pilot if domain_mask(row, domain)]
    if not domain_rows:
        raise PlanError(f'指标 {metric["id"]} 的域在 pilot 中没有记录')
    pilot_rate = sum(float(row[field]) for row in domain_rows) / len(domain_rows)
    components = {}
    for key, stratum in strata.items():
        a_values, t_values = [], []
        for device in stratum['devices']:
            rows = [row for row in device['rows'] if domain_mask(row, domain)]
            a_values.append(len(rows))
            t_values.append(sum(float(row[field]) for row in rows))
        a_series = np.asarray(a_values, dtype=float)
        t_series = np.asarray(t_values, dtype=float)
        n_devices = int(a_series.size)
        a_mean = float(a_series.mean())
        domain_records = int(a_series.sum())
        events = int(t_series.sum())
        floor, floor_kind, bounds = binomial_variance_floor(
            a_mean, domain_records, events, DESIGN_CONFIDENCE)
        s2a = adjusted_variance(a_series, DESIGN_CONFIDENCE)[0]
        s2t, _, s2t_factor, _ = adjusted_variance(t_series, DESIGN_CONFIDENCE)
        s2at = 0.0
        if n_devices > 1:
            s2at = float(np.cov(a_series, t_series, ddof=1)[0, 1]) * s2t_factor
        if s2t > 0.0:
            used_t = s2t
        elif floor_kind == 'no_domain_records':
            used_t = 0.0
        else:
            # Planning prior for "no spread at all": take the more conservative
            # of the variance adjusted for "at least one event" and the
            # Clopper-Pearson floor.  Filling all-zero strata with exactly the
            # floor would make intervals collapse whenever the replay happens
            # to see zero events, hiding real risk.
            used_t = max(s2t, floor)
            floor_kind = 'zero_events_prior' if events == 0 else floor_kind
        z_used = max(used_t - 2.0 * pilot_rate * s2at
                     + pilot_rate * pilot_rate * s2a, 0.0)
        if floor_kind in ('zero_events_prior', 'all_events_floor', 'zero_events_floor'):
            z_used = max(z_used, floor)
        components[key] = {
            'devices': n_devices,
            'a_mean': a_mean,
            'domain_records': domain_records,
            'events': events,
            's2a_adjusted': s2a,
            's2t_adjusted': s2t,
            's2t_used': used_t,
            's2at_adjusted': s2at,
            'z_used': z_used,
            'binomial_floor': floor,
            'floor_kind': floor_kind,
            'rate_bounds': bounds,
            'variance_factor': s2t_factor,
            'pilot_rate_in_stratum': (events / domain_records) if domain_records else None,
        }
    total_denominator = sum(strata[key]['estimated_devices'] * components[key]['a_mean']
                            for key in strata)
    if total_denominator <= 0:
        raise PlanError(f'指标 {metric["id"]} 的定义域在估计总体中为空')
    return {
        'metric': metric['id'],
        'domain': domain,
        'field': field,
        'pilot_rate': pilot_rate,
        'pilot_events': int(sum(float(row[field]) for row in domain_rows)),
        'pilot_domain_records': len(domain_rows),
        'components': components,
        'denominator_estimate': total_denominator,
        'target_half_width': float(metric['half_width']),
        'priority': int(metric['priority']),
    }


# ---------------------------------------------------------------------------
# Precision of one allocation
# ---------------------------------------------------------------------------

def _variance_parts(model, allocation, strata):
    parts = []
    total = model['denominator_estimate']
    for key, stratum in strata.items():
        n = allocation[key]
        if n <= 0:
            continue
        frame = stratum['estimated_devices']
        if n > frame:
            raise PlanError(f'分层 {key} 的样本量 {n} 超过估计规模 {frame}')
        coefficient = (frame / total) ** 2 * model['components'][key]['z_used']
        if coefficient <= 0.0:
            continue
        value = coefficient * (1.0 / n - 1.0 / frame)
        if value > 0.0:
            parts.append((value, n - 1))
    return parts


def precision(model, allocation, strata):
    parts = _variance_parts(model, allocation, strata)
    variance = sum(value for value, _ in parts)
    df = satterthwaite_df(parts)
    standard_error = math.sqrt(max(variance, 0.0))
    return {
        'half_width': t_quantile(0.975, df) * standard_error,
        'standard_error': standard_error,
        'degrees_of_freedom': df,
        'variance': variance,
    }


def budget_cost(allocation, strata, mode='hard'):
    rule = 'cost_bound' if mode == 'hard' else 'cost_mean'
    return sum(int(allocation[key]) * strata[key][rule] for key in strata)


# ---------------------------------------------------------------------------
# Lattice search
# ---------------------------------------------------------------------------

def _bounds(strata, min_devices):
    return {key: (min(min_devices, strata[key]['estimated_devices']),
                  strata[key]['estimated_devices']) for key in sorted(strata)}


def _lattice_size(bounds):
    size = 1
    for minimum, maximum in bounds.values():
        if maximum < minimum:
            return 0
        size *= maximum - minimum + 1
    return size


def _lattice(bounds):
    keys = sorted(bounds)
    axes = [np.arange(bounds[key][0], bounds[key][1] + 1) for key in keys]
    grid = np.meshgrid(*axes, indexing='ij')
    matrix = np.stack([part.reshape(-1) for part in grid], axis=1)
    return keys, matrix


def _student_t_isf_numpy(df):
    """Vectorised ``t.ppf(0.975, df)`` via the beta quantile identity."""
    from scipy.special import betaincinv
    # Pr(T > t) = 0.025  <=>  I_{df/(df+t^2)}(df/2, 1/2) = 0.05
    value = betaincinv(df / 2.0, 0.5, 0.05)
    return np.sqrt(df * (1.0 - value) / value)


class _MetricLattice:
    """Pre-computed precision values of one metric over the whole lattice."""

    def __init__(self, model, strata, keys, matrix):
        self.keys = keys
        self.matrix = matrix
        frames = np.array([strata[key]['estimated_devices'] for key in keys], dtype=float)
        coefficients = np.array([
            (frames[i] / model['denominator_estimate']) ** 2
            * model['components'][key]['z_used']
            for i, key in enumerate(keys)
        ], dtype=float)
        n = matrix.astype(float)
        parts = coefficients * (1.0 / n - 1.0 / frames)  # variance component per stratum
        parts = np.where(parts > 0.0, parts, 0.0)
        variance = parts.sum(axis=1)
        # Satterthwaite: (sum v)^2 / sum(v^2/(n-1)) over strata with v>0.
        degrees = np.maximum(n - 1.0, 0.0)
        ratios = np.divide(parts * parts, degrees, out=np.zeros_like(parts),
                           where=(degrees > 0.0) & (parts > 0.0))
        denominator = ratios.sum(axis=1)
        df = np.ones_like(variance)
        positive = (denominator > 0.0) & (variance > 0.0)
        df[positive] = (variance[positive] ** 2) / denominator[positive]
        df = np.maximum(df, 1.0)
        self.half_width = _student_t_isf_numpy(df) * np.sqrt(np.maximum(variance, 0.0))
        self.target = model['target_half_width']
        self.priority = model['priority']


def _evaluate_lattice(models, strata, metrics, min_devices, budget):
    bounds = _bounds(strata, min_devices)
    if _lattice_size(bounds) == 0:
        raise PlanError('估计规模小于最少设备数，无法执行分层抽样')
    if budget_cost({key: bounds[key][0] for key in bounds}, strata,
                   budget['mode']) > budget['amount']:
        raise PlanError(
            '即使每个分层只抽最少设备也超出预算，无法执行：'
            f'最低需要 {budget_cost({key: bounds[key][0] for key in bounds}, strata, budget["mode"])}，'
            f'预算 {budget["amount"]}。计划不会为凑预算丢弃稀少层')
    keys, matrix = _lattice(bounds)
    rule = 'cost_bound' if budget['mode'] == 'hard' else 'cost_mean'
    cost = np.zeros(matrix.shape[0], dtype=float)
    for i, key in enumerate(keys):
        cost += matrix[:, i] * strata[key][rule]
    evaluations = {metric['id']: _MetricLattice(models[metric['id']], strata, keys, matrix)
                   for metric in metrics}
    return {
        'keys': keys,
        'matrix': matrix,
        'cost': cost,
        'evaluations': evaluations,
        'within_budget': cost <= budget['amount'],
        'budget_amount': budget['amount'],
        'candidate_space': int(matrix.shape[0]),
        'minimum_allocation': {key: bounds[key][0] for key in keys},
    }


def _objective_keys(evaluations, metrics, cost):
    """Lexicographic columns for numpy.lexsort: last column is primary.

    Returns ``(columns, labels)`` where ``columns[k]`` must be passed to
    ``np.lexsort`` as the k-th key with the last key primary.  So the list is
    in ascending significance: cost first, then reverse priority levels.
    """
    levels = sorted({int(metric['priority']) for metric in metrics})
    primary_first = []
    for level in levels:
        ratios = [evaluations[metric['id']].half_width / metric['half_width']
                  for metric in metrics if int(metric['priority']) == level]
        stacked = np.vstack(ratios)
        primary_first.append(('p{}-max-ratio'.format(level), stacked.max(axis=0)))
        primary_first.append(('p{}-sum-squared-ratio'.format(level),
                              np.sum(stacked * stacked, axis=0)))
    primary_first.append(('cost', cost.astype(float)))
    columns = [column for _, column in reversed(primary_first)]
    labels = [label for label, _ in primary_first]
    return columns, labels


def allocate(models, strata, budget, metrics, min_devices):
    bounds = _bounds(strata, min_devices)
    if _lattice_size(bounds) > SEARCH_LIMIT:
        return _greedy_allocation(models, strata, budget, metrics, min_devices)
    lattice = _evaluate_lattice(models, strata, metrics, min_devices, budget)
    feasible = lattice['within_budget']
    if not feasible.any():
        raise PlanError('预算内没有任何可执行抽样方案')
    columns, labels = _objective_keys(lattice['evaluations'], metrics, lattice['cost'])
    masked = [np.where(feasible, column, np.inf) for column in columns]
    order = np.lexsort(masked)
    index = int(order[0])
    allocation = dict(zip(lattice['keys'], lattice['matrix'][index].tolist()))
    search = {
        'strategy': 'exhaustive-integer-lexicographic',
        'candidate_space': lattice['candidate_space'],
        'evaluated_within_budget': int(feasible.sum()),
        'chosen_budgeted_cost': budget_cost(allocation, strata, 'hard'),
        'chosen_expected_cost': budget_cost(allocation, strata, 'expected'),
        'objective': labels,
        'budget_cost_rule': ('worst-case pilot device cost per stratum'
                             if budget['mode'] == 'hard'
                             else 'mean pilot device cost per stratum'),
    }
    return _allocation_entries(allocation, strata, budget, models, metrics), search, lattice


def _allocation_entries(allocation, strata, budget, models, metrics):
    entries = []
    for key in sorted(strata):
        stratum = strata[key]
        count = int(allocation[key])
        entries.append({
            'stratum': key,
            'devices': count,
            'estimated_frame_devices': stratum['estimated_devices'],
            'selection_unit': 'device_id',
            'inclusion_probability': device_inclusion_probability(
                count, stratum['estimated_devices']),
            'per_record_inclusion_probability': device_inclusion_probability(
                count, stratum['estimated_devices']),
            'budgeted_cost': count * stratum['cost_bound'],
            'expected_cost': count * stratum['cost_mean'],
            'expected_records': count * stratum['records_per_device'],
        })
    return entries


def _greedy_allocation(models, strata, budget, metrics, min_devices):
    bounds = _bounds(strata, min_devices)
    if budget_cost({key: bounds[key][0] for key in bounds}, strata,
                   budget['mode']) > budget['amount']:
        raise PlanError('即使每层只抽最少设备也超出预算，无法给出可执行方案')
    allocation = {key: bounds[key][0] for key in bounds}
    current = _objective_scalar(models, allocation, strata, metrics, 0)
    while True:
        choice = None
        for key in bounds:
            if allocation[key] >= bounds[key][1]:
                continue
            candidate = dict(allocation)
            candidate[key] += 1
            if budget_cost(candidate, strata, budget['mode']) > budget['amount']:
                continue
            value = _objective_scalar(models, candidate, strata, metrics, 0)
            if value < current and (choice is None or value < choice[0]):
                choice = (value, key)
        if choice is None:
            break
        allocation[choice[1]] += 1
        current = choice[0]
    search = {
        'strategy': 'deterministic-greedy-fallback',
        'candidate_space': _lattice_size(bounds),
        'evaluated_within_budget': None,
        'chosen_budgeted_cost': budget_cost(allocation, strata, 'hard'),
        'chosen_expected_cost': budget_cost(allocation, strata, 'expected'),
        'objective': ['worst ratio by ascending priority', 'sum of squared ratios per priority',
                      'cost'],
        'budget_cost_rule': ('worst-case pilot device cost per stratum'
                             if budget['mode'] == 'hard'
                             else 'mean pilot device cost per stratum'),
    }
    return _allocation_entries(allocation, strata, budget, models, metrics), search, None


def _objective_scalar(models, allocation, strata, metrics, cost):
    key = []
    for level in sorted({int(metric['priority']) for metric in metrics}):
        ratios = [precision(models[metric['id']], allocation, strata)['half_width']
                  / metric['half_width']
                  for metric in metrics if int(metric['priority']) == level]
        key.extend([max(ratios), sum(value * value for value in ratios)])
    key.append(cost)
    return tuple(key)


# ---------------------------------------------------------------------------
# Per-metric reference points
# ---------------------------------------------------------------------------

def _metric_references(lattice, model, metric):
    half_width = lattice['evaluations'][metric['id']].half_width
    cost = lattice['cost']
    within = lattice['within_budget']
    best_within_budget = float(half_width[within].min()) if within.any() else None
    meeting = half_width <= metric['half_width'] + 1e-12
    minimum_budget = int(cost[meeting].min()) if meeting.any() else None
    return {
        'best_half_width_within_budget': best_within_budget,
        'minimum_budget_for_this_metric': minimum_budget,
    }


def _risk_notes(strata, models, sensitivity, min_devices, budget):
    """What happens when the pilot is fragile, rare strata are empty, or sizes move."""
    smallest = min(strata, key=lambda key: strata[key]['pilot_devices'])
    rare = [key for key in strata if strata[key]['has_rare_app']]
    zero_strata = []
    if budget['mode'] == 'hard':
        budget_text = (
            'hard：按分层最大设备成本预留，任何种子下实际支出不超过预算；'
            '交付设备若比 pilot 更贵，执行会在抽样前收缩各层设备数并重算包含概率，'
            '每层至少保留 1 台，连 1 台都付不起时拒绝执行')
    else:
        budget_text = (
            'expected：按分层平均设备成本规划，实际支出随种子波动，只保证期望不超预算；'
            '若单次实际支出偶然超过 amount，执行结果会如实标出，不会被压回预算线')
    for metric_id, model in models.items():
        for key, component in sorted(model['components'].items()):
            if component['events'] == 0 and component['domain_records'] > 0:
                zero_strata.append({'metric': metric_id, 'stratum': key,
                                    'floor': component['binomial_floor'],
                                    'kind': component['floor_kind']})
    return {
        'pilot_instability': (
            f'pilot 的最小分层是 {smallest}（{strata[smallest]["pilot_devices"]} 台设备）；'
            '规划方差使用 90% 上侧卡方因子，自由度越小因子越大，因此计划区间偏保守。'
            '执行时若样本方差为正则用样本方差，否则回落到计划先验，不会出现凭空的零宽区间。'),
        'rare_strata_zero_failure': {
            'rule': ('零事件分层按单侧 Clopper-Pearson 上界生成正的下限方差；'
                     '零失败不会被解释成零方差'),
            'affected': zero_strata,
        },
        'rare_strata': (
            f'含稀少应用的分层是 {", ".join(rare) if rare else "无"}；每个分层都保留至少 '
            f'{min_devices} 台设备，'
            '预算不足时拒绝执行而不是删层。'),
        'frame_size_bias': (
            '层规模只是估计：计划用 N_h 分配，执行与估计用交付数据里真实的设备数 M_h，'
            '包含概率按 n_h/M_h 重算，因此层规模偏差影响精度而不直接造成权重错误。'
            '若某层在交付中整体缺失，执行会记录覆盖警告，估计会如实标出该层没有贡献。'),
        'sensitivity': sensitivity,
        'budget_mode': budget_text,
    }


def _budget_mode_effect(spend):
    if spend['mode'] == 'hard':
        return ('hard 模式按最坏设备成本预留，计划精度可能比 expected 模式略低，'
                '但任何种子下支出都不会越过 amount')
    return ('expected 模式按平均设备成本规划，实际支出会围绕计划值波动；'
            '个别种子可能超过 amount，执行结果与回放都会如实标出，不会伪装成未超预算')


def _required_budget(lattice, metrics):
    columns = []
    for metric in metrics:
        evaluation = lattice['evaluations'][metric['id']]
        columns.append(evaluation.half_width <= metric['half_width'] + 1e-12)
    meets = np.all(np.vstack(columns), axis=0)
    if not meets.any():
        return None
    return int(lattice['cost'][meets].min())


def _is_binary_metric(metric, pilot):
    values = {float(row[metric['field']]) for row in pilot
              if domain_mask(row, metric['domain'])}
    return bool(values) and values <= {0.0, 1.0}


def _pilot_diagnostics(strata, models):
    """Facts that make the pilot fragile, recorded instead of hidden."""
    warnings = []
    detail = []
    for key in sorted(strata):
        stratum = strata[key]
        detail.append({
            'stratum': key,
            'pilot_devices': stratum['pilot_devices'],
            'scale_from_tier_records': stratum['scale'],
            'estimated_devices_exact': stratum['estimated_devices_exact'],
            'estimated_devices_rounded': stratum['estimated_devices'],
        })
        if stratum['pilot_devices'] < 5:
            warnings.append(
                f'{key} 的 pilot 只有 {stratum["pilot_devices"]} 台设备，'
                '方差上界很宽，计划区间会明显保守')
        elif stratum['pilot_devices'] < 15:
            warnings.append(
                f'{key} 的 pilot 只有 {stratum["pilot_devices"]} 台设备，'
                '方差估计的正则化因子较大，计划精度偏保守')
        if stratum['estimated_devices_exact'] < 2:
            warnings.append(f'{key} 的下一期设备估计不足 2 台，只能接近普查')
        if abs(stratum['scale'] - 1.0) > 0.25:
            warnings.append(
                f'{key} 所在 tier 的规模估计相对 pilot 缩放 {stratum["scale"]:.3f} 倍，'
                '层规模偏差会直接改变抽样比')
    for metric_id, model in models.items():
        for key, component in sorted(model['components'].items()):
            if component['events'] == 0 and component['domain_records'] > 0:
                warnings.append(
                    f'{metric_id}/{key} 在 pilot 中零事件；计划使用 '
                    f'{component["floor_kind"]} 先验，未把零失败当作零方差')
            elif component['events'] == component['domain_records'] \
                    and component['domain_records'] > 0:
                warnings.append(
                    f'{metric_id}/{key} 在 pilot 中全部事件；方差同样不能按零处理')
    return {'warnings': warnings, 'strata': detail}


def _population_bias_sensitivity(models, strata, allocation, metric_entries):
    """Planned half-width if every stratum frame is off by a factor.

    The allocation is held fixed; only the frame size changes.  This shows
    what a biased ``population_sizes`` costs in precision, and it also shows
    when the sample becomes a census (n_h >= M_h) so the fpc term disappears.
    """
    factors = [0.5, 0.8, 1.0, 1.25, 2.0]
    output = []
    for metric_entry, metric in metric_entries:
        model = models[metric['id']]
        rows = []
        for factor in factors:
            scaled = {}
            for key, stratum in strata.items():
                copy = dict(stratum)
                copy['estimated_devices'] = max(1, round_half_up(
                    stratum['estimated_devices_exact'] * factor))
                scaled[key] = copy
            try:
                planned = precision(model, allocation, scaled)
            except PlanError:
                planned = {'half_width': None, 'standard_error': None,
                           'degrees_of_freedom': None}
            census = [key for key in allocation
                      if allocation[key] >= scaled[key]['estimated_devices']]
            rows.append({
                'frame_scale': factor,
                'planned_half_width': planned['half_width'],
                'censused_strata': census,
            })
        output.append({'metric': metric['id'], 'rows': rows})
    return {
        'note': ('保持分配不变，仅按比例改变各层设备规模；用来展示层规模估计偏差的'
                 '影响。n_h >= M_h 的层成为普查，fpc 项消失'),
        'factors': factors,
        'metrics': output,
    }



    columns = []
    for metric in metrics:
        evaluation = lattice['evaluations'][metric['id']]
        columns.append(evaluation.half_width <= metric['half_width'] + 1e-12)
    meets = np.all(np.vstack(columns), axis=0)
    if not meets.any():
        return None
    return int(lattice['cost'][meets].min())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def design(document, options=None):
    options = dict(options or {})
    metrics = validate_metrics(document.get('metrics'))
    budget = validate_budget(document.get('budget'))
    pilot = document.get('pilot')
    if not isinstance(pilot, list) or not pilot:
        raise PlanError('pilot 必须是非空数组')
    population_sizes = validate_population_sizes(document.get('population_sizes'))
    seed = document.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PlanError('seed 必须是整数')
    min_devices = options.get('min_devices', DEFAULT_MIN_DEVICES)
    if isinstance(min_devices, bool) or not isinstance(min_devices, int) or min_devices < 1:
        raise PlanError('min_devices 必须是正整数')

    frame, rare_apps = build_device_frame(pilot, metrics)
    strata = fit_strata(frame, population_sizes, pilot)
    models = {metric['id']: metric_components(metric, strata, pilot) for metric in metrics}
    allocation, search, lattice = allocate(models, strata, budget, metrics, min_devices)
    allocation_map = {entry['stratum']: entry['devices'] for entry in allocation}
    expected_spend = sum(entry['expected_cost'] for entry in allocation)
    budgeted_spend = sum(entry['budgeted_cost'] for entry in allocation)
    if budget['mode'] == 'hard' and budgeted_spend > budget['amount']:
        raise PlanError('内部错误：hard 预算下分配超过预算上限')
    spend = {
        'mode': budget['mode'],
        'amount': budget['amount'],
        'planned_expected_spend': expected_spend,
        'planned_worst_case_spend': budgeted_spend,
        'hard_ceiling': budgeted_spend if budget['mode'] == 'hard' else None,
        'expected_ceiling': budget['amount'] if budget['mode'] == 'expected' else None,
        'note': ('hard：按分层最大设备成本预留，任何种子下实际支出都不超过 amount；'
                 'expected：按分层平均设备成本规划，实际支出随种子波动，只保证期望不超预算'),
    }
    targets = {metric['id']: float(metric['half_width']) for metric in metrics}

    precision_entries = []
    if lattice is None:
        references = {metric['id']: {'best_half_width_within_budget': None,
                                     'minimum_budget_for_this_metric': None}
                      for metric in metrics}
        required = None
    else:
        references = {metric['id']: _metric_references(lattice, models[metric['id']], metric)
                      for metric in metrics}
        required = _required_budget(lattice, metrics)
    for metric in metrics:
        metric_id = metric['id']
        planned = precision(models[metric_id], allocation_map, strata)
        met = planned['half_width'] <= targets[metric_id] + 1e-12
        precision_entries.append({
            'metric': metric_id,
            'priority': int(metric['priority']),
            'target_half_width': targets[metric_id],
            'planned_half_width': planned['half_width'],
            'planned_standard_error': planned['standard_error'],
            'planned_degrees_of_freedom': planned['degrees_of_freedom'],
            'planned_variance': planned['variance'],
            'target_met': met,
            'shortfall_absolute': max(0.0, planned['half_width'] - targets[metric_id]),
            'ratio_to_target': planned['half_width'] / targets[metric_id],
            'best_half_width_this_metric_alone_within_budget':
                references[metric_id]['best_half_width_within_budget'],
            'minimum_budget_this_metric_alone':
                references[metric_id]['minimum_budget_for_this_metric'],
        })
    unmet = [entry['metric'] for entry in precision_entries if not entry['target_met']]

    stratum_entries = []
    for key in sorted(strata):
        stratum = strata[key]
        stratum_entries.append({
            'id': key,
            'tier': stratum['tier'],
            'has_rare_app': stratum['has_rare_app'],
            'pilot_devices': stratum['pilot_devices'],
            'estimated_devices': stratum['estimated_devices'],
            'estimated_devices_exact': stratum['estimated_devices_exact'],
            'scale_from_tier_records': stratum['scale'],
            'records_per_device_pilot': stratum['records_per_device'],
            'cost_bound_per_device': stratum['cost_bound'],
            'cost_mean_per_device': stratum['cost_mean'],
            'estimated_records': stratum['estimated_records'],
        })

    metric_entries = []
    for metric in metrics:
        model = models[metric['id']]
        metric_entries.append({
            'id': metric['id'],
            'field': metric['field'],
            'domain': dict(metric['domain']),            'half_width': targets[metric['id']],
            'priority': int(metric['priority']),
            'pilot_rate': model['pilot_rate'],
            'pilot_events': model['pilot_events'],
            'pilot_domain_records': model['pilot_domain_records'],
            'binary': _is_binary_metric(metric, pilot),
            'stratum_components': [
                {
                    'stratum': key,
                    'pilot_devices': component['devices'],
                    'domain_records': component['domain_records'],
                    'events': component['events'],
                    'pilot_rate': component['pilot_rate_in_stratum'],
                    'variance_floor_kind': component['floor_kind'],
                    'binomial_floor': component['binomial_floor'],
                    'rate_bounds': component['rate_bounds'],
                    'variance_adjusted': component['s2t_adjusted'],
                    'variance_used_for_planning': component['s2t_used'],
                    'residual_variance_prior': component['z_used'],
                }
                for key, component in sorted(model['components'].items())
            ],
        })

    plan = {
        'schema': 'telemetry-sample-plan/1',
        'design': {
            'method': 'stratified-one-stage-device-cluster',
            'selection_unit': 'device_id',
            'record_rule': 'selected devices upload every row',
            'strata_definition': 'tier x (device carries a rare-app row)',
            'strata': stratum_entries,
            'rare_apps': rare_apps,
            'population_source': 'pilot device counts scaled by tier record totals',
            'min_devices_per_stratum': min_devices,
            'variance_confidence': DESIGN_CONFIDENCE,
            'zero_event_rule': ('one-sided Clopper-Pearson floor; zero observed failures '
                                'never means zero variance'),
            'estimator': {
                'family': 'horvitz-thompson-ratio-of-means',
                'numerator': 'sum_h (M_h/n_h) sum_selected t_d',
                'denominator': 'sum_h (M_h/n_h) sum_selected a_d',
                'variance': 'sum_h (M_h/A)^2 (1 - n_h/M_h) S2_h / n_h',
                'interval': 't interval, Satterthwaite degrees of freedom',
                'selection_unit': 'device; every row of a selected device is included',
                'frame_convention': ('execution and estimation use the delivered frame count '
                                     'M_h; the plan N_h is only a planning estimate'),
            },
        },
        'budget': budget,
        'spend': spend,
        'seed': seed,
        'metrics': metric_entries,
        'allocation': allocation,
        'precision': precision_entries,
        'feasibility': {
            'status': 'met' if not unmet else 'partially_met',
            'targets_met': [entry['metric'] for entry in precision_entries if entry['target_met']],
            'targets_unmet': unmet,
            'required_budget_all_targets': required,
            'budget_available': budget['amount'],
            'budget_gap': (required - budget['amount']) if required is not None else None,
            'detail': [
                {'metric': entry['metric'],
                 'priority': entry['priority'],
                 'target_half_width': entry['target_half_width'],
                 'planned_half_width': entry['planned_half_width'],
                 'target_met': entry['target_met'],
                 'shortfall_absolute': entry['shortfall_absolute']}
                for entry in precision_entries
            ],
            'reason': None if not unmet else (
                '预算内无法同时满足全部精度目标；输出按优先级的最小最大比值折中，'
                '未达目标、缺口和所需预算均如实保留，没有超出预算或丢弃稀少层'),
            'budget_mode_effect': _budget_mode_effect(spend),
        },
        'search': search,
    }
    plan['pilot_diagnostics'] = _pilot_diagnostics(strata, models)
    plan['population_bias_sensitivity'] = _population_bias_sensitivity(
        models, strata, allocation_map, list(zip(metric_entries, metrics)))
    plan['risk_notes'] = _risk_notes(strata, models, plan['population_bias_sensitivity'],
                                     min_devices, budget)
    plan['design_input_digest'] = design_input_digest(document)
    plan['plan_fingerprint'] = plan_fingerprint(plan)
    return plan


def verify_plan(plan):
    if not isinstance(plan, dict) or plan.get('schema') != 'telemetry-sample-plan/1':
        raise PlanError('不是 telemetry-sample-plan/1 计划')
    fingerprint = plan.get('plan_fingerprint')
    if not isinstance(fingerprint, str):
        raise PlanError('计划缺少 plan_fingerprint')
    if plan_fingerprint(plan) != fingerprint:
        raise PlanError('计划指纹不匹配，计划在保存后被改动')
    for key in ('design', 'budget', 'allocation', 'metrics', 'seed'):
        if key not in plan:
            raise PlanError('计划缺少字段：' + key)
    return plan


def plan_allocation_map(plan):
    return {entry['stratum']: int(entry['devices']) for entry in plan['allocation']}

