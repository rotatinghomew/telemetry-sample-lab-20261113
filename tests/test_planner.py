import copy
import json
import tempfile
import unittest
from pathlib import Path

from telemetry_sample_lab.design import (PlanError, design, plan_allocation_map,
                                         validate_budget, validate_metrics,
                                         verify_plan)
from telemetry_sample_lab.estimate import estimate
from telemetry_sample_lab.execute import build_frame, hard_budget_allocation, sample


def load(name):
    return json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / name).read_text(encoding='utf-8'))


class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load('design.json')
        cls.plan = design(cls.document)

    def test_plan_has_required_sections(self):
        for key in ('schema', 'design', 'budget', 'spend', 'allocation', 'precision',
                    'feasibility', 'search', 'metrics', 'plan_fingerprint',
                    'design_input_digest'):
            self.assertIn(key, self.plan)

    def test_every_stratum_keeps_devices(self):
        for entry in self.plan['allocation']:
            self.assertGreaterEqual(entry['devices'], 1, entry)
            self.assertLessEqual(entry['devices'], entry['estimated_frame_devices'])

    def test_inclusion_probability_matches_allocation(self):
        for entry in self.plan['allocation']:
            self.assertAlmostEqual(entry['inclusion_probability'],
                                   entry['devices'] / entry['estimated_frame_devices'])

    def test_selection_unit_is_device_and_records_inherit_it(self):
        self.assertEqual(self.plan['design']['selection_unit'], 'device_id')
        self.assertIn('every row', self.plan['design']['record_rule'])

    def test_design_ignores_replay_population(self):
        altered = copy.deepcopy(self.document)
        for row in altered['replay_population']:
            row['failure'] = 1 - row['failure']
            row['long_task'] = 1 - row['long_task']
        without = {key: value for key, value in self.document.items()
                   if key != 'replay_population'}
        self.assertEqual(design(altered)['plan_fingerprint'],
                         self.plan['plan_fingerprint'])
        self.assertEqual(design(without)['plan_fingerprint'],
                         self.plan['plan_fingerprint'])

    def test_design_digest_excludes_replay(self):
        changed = copy.deepcopy(self.document)
        changed['replay_population'] = changed['replay_population'][:1]
        self.assertEqual(design(changed)['design_input_digest'],
                         self.plan['design_input_digest'])

    def test_feasibility_reports_unmet_targets(self):
        feasibility = self.plan['feasibility']
        precision = {entry['metric']: entry for entry in self.plan['precision']}
        self.assertEqual(sorted(feasibility['targets_unmet']),
                         sorted(metric for metric, entry in precision.items()
                                if not entry['target_met']))
        for metric in feasibility['targets_unmet']:
            self.assertGreater(precision[metric]['planned_half_width'],
                               precision[metric]['target_half_width'])

    def test_feasibility_reports_required_budget(self):
        feasibility = self.plan['feasibility']
        self.assertIsNotNone(feasibility['required_budget_all_targets'])
        self.assertGreater(feasibility['required_budget_all_targets'],
                           self.plan['budget']['amount'])
        self.assertEqual(feasibility['budget_gap'],
                         feasibility['required_budget_all_targets']
                         - feasibility['budget']['amount']
                         if 'budget' in feasibility else
                         feasibility['required_budget_all_targets']
                         - feasibility['budget_available'])

    def test_zero_event_strata_have_positive_prior(self):
        for metric in self.plan['metrics']:
            for component in metric['stratum_components']:
                if component['domain_records'] > 0 and component['events'] == 0:
                    self.assertGreater(component['residual_variance_prior'], 0.0)
                    self.assertIn('floor', component['variance_floor_kind']
                                  + component['variance_floor_kind'])

    def test_hard_budget_reserves_worst_case(self):
        self.assertEqual(self.plan['spend']['mode'], 'hard')
        self.assertLessEqual(self.plan['spend']['planned_worst_case_spend'],
                             self.plan['budget']['amount'])
        for entry in self.plan['allocation']:
            self.assertGreaterEqual(entry['budgeted_cost'], entry['expected_cost'])

    def test_population_bias_sensitivity_present(self):
        sensitivity = self.plan['population_bias_sensitivity']
        self.assertIn(1.0, sensitivity['factors'])
        for metric in sensitivity['metrics']:
            self.assertEqual(len(metric['rows']), len(sensitivity['factors']))

    def test_pilot_diagnostics_present(self):
        self.assertIn('warnings', self.plan['pilot_diagnostics'])
        self.assertEqual(len(self.plan['pilot_diagnostics']['strata']),
                         len(self.plan['allocation']))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.document = load('design.json')

    def test_rejects_bad_budget(self):
        with self.assertRaises(PlanError):
            validate_budget({'amount': 0, 'mode': 'hard'})
        with self.assertRaises(PlanError):
            validate_budget({'amount': 10, 'mode': 'sometimes'})

    def test_rejects_bad_metric_target(self):
        metrics = copy.deepcopy(self.document['metrics'])
        metrics[0]['half_width'] = 0
        with self.assertRaises(PlanError):
            validate_metrics(metrics)

    def test_rejects_tampered_plan(self):
        plan = design(self.document)
        plan['allocation'][0]['devices'] += 1
        with self.assertRaises(PlanError):
            verify_plan(plan)

    def test_rejects_missing_frame_tier(self):
        document = copy.deepcopy(self.document)
        document['population_sizes'] = [entry for entry in document['population_sizes']
                                        if entry['tier'] != 'entry']
        with self.assertRaises(PlanError):
            design(document)

    def test_rejects_duplicate_population_entry(self):
        document = copy.deepcopy(self.document)
        document['population_sizes'].append(dict(document['population_sizes'][0]))
        with self.assertRaises(PlanError):
            design(document)

    def test_expected_budget_mode_recorded(self):
        document = copy.deepcopy(self.document)
        document['budget'] = {'amount': 100, 'mode': 'expected'}
        plan = design(document)
        self.assertEqual(plan['spend']['mode'], 'expected')
        self.assertLessEqual(plan['spend']['planned_expected_spend'], 100)
        self.assertIsNone(plan['spend']['hard_ceiling'])


class SampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load('design.json')
        cls.plan = design(cls.document)

    def test_deterministic_for_same_seed(self):
        left = sample(self.document, self.plan, seed=7)
        right = sample(self.document, self.plan, seed=7)
        self.assertEqual([row['row_id'] for row in left['sample']],
                         [row['row_id'] for row in right['sample']])

    def test_reordered_frame_same_sample(self):
        reversed_document = copy.deepcopy(self.document)
        reversed_document['replay_population'] = \
            self.document['replay_population'][::-1]
        left = sample(self.document, self.plan, seed=11)
        right = sample(reversed_document, self.plan, seed=11)
        self.assertEqual({row['row_id'] for row in left['sample']},
                         {row['row_id'] for row in right['sample']})

    def test_hard_budget_never_exceeded_across_seeds(self):
        for seed in range(1, 40):
            execution = sample(self.document, self.plan, seed=seed)
            self.assertLessEqual(execution['spent'], self.plan['budget']['amount'])
            self.assertTrue(execution['within_budget'])

    def test_every_row_carries_device_probability(self):
        execution = sample(self.document, self.plan, seed=3)
        by_device = {}
        for row in execution['sample']:
            self.assertEqual(row['selection_unit'], 'device_id')
            by_device.setdefault(row['selection_unit_id'], set()).add(
                row['inclusion_probability'])
        for probabilities in by_device.values():
            self.assertEqual(len(probabilities), 1)
            probability = probabilities.pop()
            self.assertTrue(0 < probability <= 1)

    def test_hard_budget_shrinks_without_dropping_strata(self):
        document = copy.deepcopy(self.document)
        for row in document['replay_population']:
            row['cost'] = 8 if row['app'] == 'rare' else 1
        execution = sample(document, self.plan, seed=1)
        self.assertLessEqual(execution['spent'], 100)
        self.assertEqual(len(execution['sample']) > 0, True)
        for report in execution['strata']:
            self.assertGreaterEqual(report['selected_devices'], 1, report)
            self.assertEqual(report['selected_devices'], len(
                {row['selection_unit_id'] for row in execution['sample']
                 if row['stratum'] == report['stratum']}))

    def test_hard_budget_refuses_when_minimum_unaffordable(self):
        document = copy.deepcopy(self.document)
        for row in document['replay_population']:
            row['cost'] = 100
        with self.assertRaises(PlanError):
            sample(document, self.plan, seed=1)

    def test_unknown_stratum_rejected(self):
        document = copy.deepcopy(self.document)
        document['replay_population'][0]['tier'] = 'new-tier'
        with self.assertRaises(PlanError):
            sample(document, self.plan, seed=1)

    def test_frame_rebuild_groups_by_device(self):
        rows = self.document['replay_population']
        frame = build_frame(rows, self.plan['design']['rare_apps'])
        self.assertEqual(len(frame), len({row['device_id'] for row in rows}))

    def test_worst_case_allocation_is_seed_independent(self):
        from collections import defaultdict
        grouped = defaultdict(list)
        for device in build_frame(self.document['replay_population'],
                                  self.plan['design']['rare_apps']).values():
            grouped[device['stratum']].append(device)
        counts, worst = hard_budget_allocation(plan_allocation_map(self.plan), grouped, 100)
        self.assertLessEqual(worst, 100)
        for stratum in grouped:
            self.assertGreaterEqual(counts[stratum], 1)


class EstimateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load('design.json')
        cls.plan = design(cls.document)
        cls.execution = sample(cls.document, cls.plan)
        cls.result = estimate(cls.execution, cls.plan)

    def test_matches_plan_fingerprint(self):
        self.assertEqual(self.result['plan_fingerprint'], self.plan['plan_fingerprint'])

    def test_every_metric_has_status(self):
        for entry in self.result['metrics']:
            self.assertIn(entry['status'], ('ok', 'undefined_variance',
                                            'undefined_no_domain_rows'))
            if entry['status'] == 'ok':
                self.assertLessEqual(entry['interval_lower'], entry['estimate'])
                self.assertGreaterEqual(entry['interval_upper'], entry['estimate'])

    def test_stratum_component_records_probability(self):
        for entry in self.result['metrics']:
            for component in entry['strata']:
                if component['selected_devices'] > 0:
                    self.assertAlmostEqual(
                        component['inclusion_probability'],
                        component['selected_devices'] / component['frame_devices'])
                    self.assertAlmostEqual(component['device_weight'],
                                           1.0 / component['inclusion_probability'])

    def test_zero_sample_variance_uses_prior(self):
        found_prior = False
        for seed in range(1, 60):
            execution = sample(self.document, self.plan, seed=seed)
            result = estimate(execution, self.plan)
            for metric in result['metrics']:
                for component in metric['strata']:
                    if component['sample_variance'] == 0.0 \
                            and component['selected_devices'] >= 2 \
                            and component['variance_source'] not in (
                                'stratum_outside_metric_domain',
                                'undefined_insufficient_variance'):
                        self.assertNotEqual(component['variance_source'], 'sample')
                        self.assertGreater(component['variance_used'], 0.0)
                        found_prior = True
        self.assertTrue(found_prior)

    def test_rejects_tampered_probability(self):
        execution = copy.deepcopy(self.execution)
        execution['sample'][0]['inclusion_probability'] = 0.5
        with self.assertRaises(PlanError):
            estimate(execution, self.plan)

    def test_rejects_wrong_plan(self):
        other_document = copy.deepcopy(self.document)
        other_document['budget'] = {'amount': 101, 'mode': 'hard'}
        other_plan = design(other_document)
        with self.assertRaises(PlanError):
            estimate(self.execution, other_plan)

    def test_census_recovers_exact_rate(self):
        document = copy.deepcopy(self.document)
        # Census of the delivered frame (replay rows) with an ample budget.
        census_rows = document['replay_population']
        plan = copy.deepcopy(self.plan)
        plan['budget'] = {'amount': 10 ** 6, 'mode': 'hard'}
        plan['spend']['amount'] = 10 ** 6
        for entry in plan['allocation']:
            entry['devices'] = entry['estimated_frame_devices']
        from telemetry_sample_lab.common import plan_fingerprint
        plan['plan_fingerprint'] = plan_fingerprint(plan)
        execution = sample(document, plan)
        result = estimate(execution, plan)
        truth = {metric['id']: sum(float(row[metric['field']]) for row in census_rows
                                   if all(row.get(k) == v for k, v in metric['domain'].items()))
                 / sum(1 for row in census_rows
                       if all(row.get(k) == v for k, v in metric['domain'].items()))
                 for metric in plan['metrics']}
        for entry in result['metrics']:
            self.assertEqual(entry['status'], 'ok')
            self.assertAlmostEqual(entry['estimate'], truth[entry['metric']],
                                   msg=entry['metric'])


if __name__ == '__main__':
    unittest.main()


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load('design.json')
        cls.plan = design(cls.document)
        from telemetry_sample_lab.replay import replay
        cls.result = replay(cls.plan, cls.document, range(1, 61))

    def test_hard_budget_holds_across_replays(self):
        spend = self.result['spend']
        self.assertTrue(spend['within_budget_all_seeds'])
        self.assertEqual(spend['hard_violations'], 0)
        self.assertLessEqual(spend['max'], self.plan['budget']['amount'])

    def test_plan_is_defined_for_every_replay(self):
        for metric, summary in self.result['plan'].items():
            self.assertEqual(summary['undefined_replays'], 0, metric)
            self.assertIsNotNone(summary['coverage'], metric)

    def test_baseline_reports_its_own_undefineds(self):
        for metric, summary in self.result['baseline_uniform']['metrics'].items():
            self.assertEqual(summary['defined_replays'] + summary['undefined_replays'],
                             summary['replays'])
            if summary['defined_replays']:
                self.assertIn('coverage', summary)

    def test_comparison_lists_every_metric(self):
        metrics = {metric['id'] for metric in self.plan['metrics']}
        self.assertEqual({row['metric'] for row in self.result['comparison']}, metrics)

    def test_bias_is_small_for_non_rare_metrics(self):
        for metric in ('failure', 'long_tasks'):
            summary = self.result['plan'][metric]
            self.assertLess(abs(summary['bias']), 0.02, summary)
            self.assertLess(summary['rmse'], 0.05, summary)
            self.assertGreater(summary['coverage'], 0.90, summary)

    def test_truth_uses_replay_only(self):
        for metric in self.plan['metrics']:
            rows = [row for row in self.document['replay_population']
                    if all(row.get(k) == v for k, v in metric['domain'].items())]
            expected = sum(row[metric['field']] for row in rows) / len(rows)
            self.assertAlmostEqual(self.result['truth_from_replay'][metric['id']], expected)


class PlannerCliTests(unittest.TestCase):
    def test_full_pipeline(self):
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            plan = directory / 'plan.json'
            sample_path = directory / 'sample.json'
            estimate_path = directory / 'estimate.json'
            replay_path = directory / 'replay.json'
            environment = {'PYTHONPATH': str(root), 'PATH': '/usr/bin:/bin'}
            commands = [
                [sys.executable, '-m', 'telemetry_sample_lab.planner', 'design',
                 '--input', str(root / 'fixtures/design.json'), '--output', str(plan)],
                [sys.executable, '-m', 'telemetry_sample_lab.planner', 'sample',
                 '--input', str(root / 'fixtures/design.json'), '--plan', str(plan),
                 '--output', str(sample_path)],
                [sys.executable, '-m', 'telemetry_sample_lab.planner', 'estimate',
                 '--input', str(sample_path), '--plan', str(plan),
                 '--output', str(estimate_path)],
                [sys.executable, '-m', 'telemetry_sample_lab.planner', 'replay',
                 '--input', str(root / 'fixtures/design.json'), '--plan', str(plan),
                 '--output', str(replay_path), '--seeds', '20'],
            ]
            for command in commands:
                completed = subprocess.run(command, cwd=str(root), env=environment,
                                           capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            for path in (plan, sample_path, estimate_path, replay_path):
                self.assertTrue(path.exists(), path)
            self.assertEqual(len(json.loads(plan.read_text())['allocation']),
                             len(json.loads(plan.read_text())['design']['strata']))
            self.assertEqual(len(json.loads(estimate_path.read_text())['metrics']),
                             len(json.loads(plan.read_text())['metrics']))
