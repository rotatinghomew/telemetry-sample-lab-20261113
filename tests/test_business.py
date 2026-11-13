import copy
import json
import tempfile
import unittest
from pathlib import Path


def load(name):
    return json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / name).read_text(encoding='utf-8'))


from telemetry_sample_lab.sampling import uniform_draw, bernoulli_sample, ht_total


class BusinessTests(unittest.TestCase):
    def setUp(self):
        self.rows = load('design.json')['pilot']

    def test_reorder_stability(self):
        left = bernoulli_sample(self.rows, .2, 41)
        right = bernoulli_sample(self.rows[::-1], .2, 41)
        self.assertEqual({row['row_id'] for row in left}, {row['row_id'] for row in right})

    def test_census(self):
        sample = bernoulli_sample(self.rows, 1, 0)
        self.assertEqual(ht_total(sample, 'failure'), sum(row['failure'] for row in self.rows))

    def test_bad_probability(self):
        for probability in [0, -1, 1.1]:
            with self.assertRaises(ValueError):
                bernoulli_sample(self.rows, probability, 0)

    def test_duplicate_id(self):
        with self.assertRaises(ValueError):
            bernoulli_sample([self.rows[0], self.rows[0]], .5, 1)

    def test_half_inclusion(self):
        self.assertEqual(ht_total([{'failure': 1, 'inclusion_probability': .5}], 'failure'), 2)

    def test_seed_changes_draw(self):
        self.assertNotEqual(uniform_draw(1, 'same'), uniform_draw(2, 'same'))
