"""Execute a saved plan against a population frame.

Execution never reads design fixtures: it uses only the plan (allocation,
inclusion probabilities, priors, fingerprint) and the delivered population
rows.  The device frame is rebuilt from the delivered rows, so the actual
inclusion probability of each device is ``n_h / M_h`` with ``M_h`` the number
of devices really present in stratum ``h``.  Selection is deterministic in
``(seed, stratum, device_id)`` so a reordered frame yields the same sample.
"""

import collections

from .common import PlanError, device_inclusion_probability, sha256_text
from .design import plan_allocation_map, verify_plan


def device_groups(rows):
    groups = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row['device_id'], []).append(row)
    return groups



def build_frame(rows, rare_apps):
    """Rebuild device strata from delivered rows (device-level membership)."""
    frame = collections.OrderedDict()
    for device_id, device_rows in device_groups(rows).items():
        tiers = {row['tier'] for row in device_rows}
        if len(tiers) != 1:
            raise PlanError(f'设备 {device_id} 跨多个 tier，无法执行分层抽样')
        tier = tiers.pop()
        has_rare = any(row['app'] in rare_apps for row in device_rows)
        stratum = f'{tier}|{"rare" if has_rare else "standard"}'
        frame[device_id] = {
            'device_id': device_id,
            'stratum': stratum,
            'rows': device_rows,
            'cost': sum(int(row['cost']) for row in device_rows),
        }
    return frame


def draw(seed, stratum, device_id):
    return sha256_text(f'{seed}:{stratum}:{device_id}')


def hard_budget_allocation(planned, strata_devices, budget_amount):
    """Shrink device counts until even the worst draw fits the hard budget.

    The worst draw of ``n`` devices from a stratum is the ``n`` most expensive
    devices, so checking that subset is sufficient for every seed.  Counts are
    reduced before drawing, from the stratum holding the most expensive
    currently-selected device, but never below one device while its stratum is
    non-empty: dropping a stratum entirely would silently remove it from the
    estimate, which is not an acceptable way to "fit" the budget.  If even one
    device per non-empty stratum exceeds the budget, execution refuses instead
    of over-spending or dropping a stratum.

    The reduced ``n_h`` then defines the actual inclusion probability
    ``n_h / M_h`` recorded on every selected row, so the estimator remains
    design-consistent.
    """
    counts = {}
    for stratum, devices in strata_devices.items():
        counts[stratum] = min(planned.get(stratum, 0), len(devices))
    minimum = {stratum: (1 if devices else 0)
               for stratum, devices in strata_devices.items()}
    ranked_costs = {stratum: sorted((device['cost'] for device in devices), reverse=True)
                    for stratum, devices in strata_devices.items()}

    def worst_total():
        return sum(sum(ranked_costs[stratum][:counts[stratum]])
                   for stratum in counts)

    if worst_total() <= budget_amount:
        return counts, worst_total()
    floor_total = sum(ranked_costs[stratum][:minimum[stratum]][0]
                      if minimum[stratum] else 0 for stratum in counts)
    if floor_total > budget_amount:
        raise PlanError(
            '硬预算下无法覆盖每个分层至少一台设备：交付数据的设备成本超过了计划的'
            f'预留上限（最低需要 {floor_total}，预算 {budget_amount}）。'
            '拒绝执行而不是超预算或丢弃稀少层')
    while worst_total() > budget_amount:
        candidates = [stratum for stratum in counts if counts[stratum] > minimum[stratum]]
        if not candidates:
            raise PlanError('硬预算约束无法满足，拒绝执行')
        chosen = max(candidates, key=lambda stratum: (
            ranked_costs[stratum][counts[stratum] - 1], stratum))
        counts[chosen] -= 1
    return counts, worst_total()


def sample(document, plan, seed=None):
    verify_plan(plan)
    rows = document.get('population_rows')
    if rows is None:
        rows = document.get('replay_population')
    if not isinstance(rows, list) or not rows:
        raise PlanError('执行输入必须提供非空的 population_rows 或 replay_population')
    rare_apps = plan['design']['rare_apps']
    allocation = plan_allocation_map(plan)
    run_seed = plan['seed'] if seed is None else seed
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise PlanError('执行种子必须是整数')

    frame = build_frame(rows, rare_apps)
    by_stratum = collections.defaultdict(list)
    for device in frame.values():
        by_stratum[device['stratum']].append(device)
    unknown = sorted(set(by_stratum) - set(allocation))
    if unknown:
        raise PlanError('执行数据出现计划未覆盖的分层：' + ', '.join(unknown))
    if plan['budget']['mode'] == 'hard':
        effective, worst_case = hard_budget_allocation(
            allocation, by_stratum, plan['budget']['amount'])
        shrink = {key: allocation[key] - effective.get(key, 0)
                  for key in allocation if effective.get(key, 0) < allocation[key]}
    else:
        effective = {key: min(allocation[key], len(by_stratum.get(key, [])))
                     for key in allocation}
        worst_case = None
        shrink = {}

    selected_rows = []
    stratum_reports = []
    total_cost = 0
    for stratum in sorted(set(list(by_stratum) + list(allocation))):
        devices = sorted(by_stratum.get(stratum, []),
                         key=lambda device: device['device_id'])
        frame_devices = len(devices)
        planned_devices = int(allocation.get(stratum, 0))
        wanted = min(int(effective.get(stratum, 0)), frame_devices)
        ranked = sorted(devices, key=lambda device: (draw(run_seed, stratum,
                                                          device['device_id']),
                                                     device['device_id']))
        chosen = ranked[:wanted]
        probability = device_inclusion_probability(wanted, frame_devices)
        for device in chosen:
            for row in device['rows']:
                payload = {key: value for key, value in row.items()
                           if key not in ('stratum', 'selection_unit', 'selection_unit_id',
                                          'inclusion_probability',
                                          'device_inclusion_probability')}
                payload.update({
                    'stratum': stratum,
                    'selection_unit': 'device_id',
                    'selection_unit_id': device['device_id'],
                    'inclusion_probability': probability,
                    'device_inclusion_probability': probability,
                })
                selected_rows.append(payload)
            total_cost += device['cost']
        stratum_reports.append({
            'stratum': stratum,
            'frame_devices': frame_devices,
            'planned_devices': planned_devices,
            'selected_devices': wanted,
            'inclusion_probability': probability,
            'frame_cost_bound': max((device['cost'] for device in devices), default=0),
            'spent_in_stratum': sum(device['cost'] for device in chosen),
        })
    selected_rows.sort(key=lambda row: row['row_id'])
    zero_selected = [report['stratum'] for report in stratum_reports
                     if report['frame_devices'] > 0 and report['selected_devices'] == 0]
    missing_strata = sorted(set(allocation) - set(by_stratum))
    warnings = []
    if missing_strata:
        warnings.append('交付数据缺少计划分层：' + ', '.join(missing_strata))
    if zero_selected:
        warnings.append('以下分层交付了设备但一台未选：' + ', '.join(zero_selected))
    for report in stratum_reports:
        if report['selected_devices'] == 1:
            warnings.append(f'{report["stratum"]} 只抽到 1 台设备，该层方差将依赖计划先验')
    return {
        'schema': 'telemetry-sample-execution/1',
        'plan_fingerprint': plan['plan_fingerprint'],
        'seed': run_seed,
        'budget': plan['budget'],
        'spent': total_cost,
        'within_budget': total_cost <= plan['budget']['amount'],
        'effective_allocation': {stratum: int(effective.get(stratum, 0))
                                 for stratum in sorted(set(list(allocation)
                                                           + list(effective)))},
        'hard_budget_shrink': shrink,
        'hard_budget_worst_case': worst_case,
        'empty_strata': missing_strata,
        'warnings': warnings,
        'strata': stratum_reports,
        'sample': selected_rows,
    }
