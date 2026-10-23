"""Pure public-state fingerprints, matching the development scene contract v2."""
import hashlib
import json
import math

from .dev_v2 import SHA, require

V1 = 'attentionbench.public-scene-contract.v1'
V2 = 'attentionbench.public-scene-contract.v2'


def fingerprint(state, rounded=False):
    def convert(value):
        if isinstance(value, dict):
            return {key: convert(child) for key, child in sorted(value.items())}
        if isinstance(value, list):
            return [convert(child) for child in value]
        if isinstance(value, (float, int)) and type(value) is not bool:
            if not math.isfinite(value):
                raise ValueError('nonfinite public condition')
            return round(value, 6) if rounded else value
        raise ValueError('numeric public state required')
    data = json.dumps(convert(state), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def verify_v2_receipt(contract, config, state, receipt):
    """Recompute both hashes and the verdict; a mismatch is never a match."""
    require(contract.get('schema') == V2 and contract.get('tolerance') == 1e-6,
            'frozen scene contract/tolerance required')
    seed = config.get('seed')
    require(type(seed) is int, 'seed-keyed scene contract requires the configuration seed as an integer')
    entry = contract.get('tasks', {}).get(config['task_id'], {}).get('seeds', {}).get(str(seed))
    require(isinstance(entry, dict), 'scene contract has no reference for task/seed')
    expected = entry.get('normalized_sha256')
    require(isinstance(expected, str) and SHA.fullmatch(expected), 'scene contract normalized digest required')
    require(isinstance(receipt, dict) and receipt.get('contract') == config['scene_contract']
            and receipt.get('task_id') == config['task_id'] and type(receipt.get('seed')) is int
            and receipt['seed'] == seed and receipt.get('tolerance') == 1e-6,
            'scene receipt/contract/seed binding mismatch')
    normalized = fingerprint(state, rounded=True)
    require(receipt.get('normalized_sha256') == normalized and receipt.get('raw_sha256') == fingerprint(state),
            'scene receipt hashes differ from raw state')
    matched = normalized == expected
    require(type(receipt.get('matched')) is bool and receipt['matched'] == matched,
            'scene verdict differs from raw state')
    return matched
