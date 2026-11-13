import hashlib
import math


def uniform_draw(seed, identity):
    payload = f'{seed}:{identity}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') / (2 ** 64)


def bernoulli_sample(rows, probability, seed):
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError('包含概率必须在零到一之间且大于零')
    identities = [row['row_id'] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('记录编号重复')
    return [{**row, 'inclusion_probability': probability} for row in rows
            if uniform_draw(seed, row['row_id']) < probability]


def ht_total(sample, field):
    total = 0.0
    for row in sample:
        probability = row['inclusion_probability']
        if not math.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError('样本包含概率无效')
        total += row[field] / probability
    return total


def inspect(document):
    sample = bernoulli_sample(document['pilot'], document['baseline_probability'], document['seed'])
    return {'sample': sample, 'cost': sum(row['cost'] for row in sample),
            'failure_total_estimate': ht_total(sample, 'failure'), 'design': 'independent-record-bernoulli'}
