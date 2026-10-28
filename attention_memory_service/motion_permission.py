"""Independent verifier for import-only simulated Dev motion permission."""
import ast
import hashlib
import json
import sqlite3

ACTION = 'dev_tier_sim_motion'
PURPOSE = 'memory_validation'
# The supported RoboCasa365 SDK interface; do not execute imported SDK/source code.
MOTION = {'left_arm': {'move_delta'}, 'right_arm': {'move_delta'},
          'left_gripper': {'open', 'close'}, 'right_gripper': {'open', 'close'},
          'bimanual': {'move_joints'}, 'base': {'move_delta'}}


def operations(source, require):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        require(False, 'invalid sealed motion source')
    imported = {alias.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                and n.module == 'robot_sdk' for alias in n.names}
    referenced = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    return sorted(f'{group}.{method}' for group in imported
                  for method in MOTION.get(group, ()) if method in referenced)


def verify(permission, descriptor, config, bundle, source, require):
    require(descriptor.get('import_purpose') == PURPOSE and 'approval_id' not in descriptor,
            'simulated Dev motion requires validation-only purpose and no approval ID')
    require(config.get('budget_tier') == 'dev' and config.get('runner_boundary') == 'dev_loop'
            and config.get('budget_boundary', 'dev_loop') == 'dev_loop'
            and bundle.get('runner_boundary') == 'dev_loop' and bundle.get('channel') == 'dev_v2'
            and bundle.get('formal_eligible') is False and bundle.get('entry_lock') is None,
            'simulated Dev motion requires frozen dev tier and dev boundary')
    require(bundle.get('suite') == 'robocasa365'
            and all(v.get('execution_target', 'robocasa365_sim') == 'robocasa365_sim' for v in (config, bundle)),
            'simulated Dev motion requires robocasa365_sim')
    inventory = operations(source, require)
    require(bool(inventory), 'simulated Dev motion requires a motion inventory')
    expected = dict(run_id='run:' + descriptor['cell'], cell=descriptor['cell'], code_version=descriptor['version'],
                    code_sha256=hashlib.sha256(source).hexdigest(),
                    config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                    action=ACTION, actor=ACTION, action_version=1, execution_target='robocasa365_sim',
                    risk='bounded simulated robot motion under existing SDK safety limits',
                    scope=dict(operations=inventory, robot=config['robot'], task_id=config['task_id'], seed=config['seed'],
                               sdk_calls=config['sdk_call_limit'], control_steps=config['horizon'], wall_seconds=config['attempt_wall_seconds']))
    require(permission == expected, 'simulated Dev motion scope differs from frozen source/config binding')


def verify_agent(permission, descriptor, config, bundle, source, job, require):
    """Bounded Dev agent motion carries no human/planner approval identity."""
    require(config.get('budget_tier') == 'dev' and config.get('runner_boundary') == 'dev_loop'
            and config.get('budget_boundary', 'dev_loop') == 'dev_loop'
            and config.get('dev_mode') == 'agent' and bundle.get('runner_boundary') == 'dev_loop'
            and bundle.get('channel') == 'dev_agent' and bundle.get('formal_eligible') is False
            and bundle.get('entry_lock') is None, 'agent motion requires frozen agent dev tier and boundary')
    require(bundle.get('suite') == 'robocasa365'
            and all(v.get('execution_target', 'robocasa365_sim') == 'robocasa365_sim' for v in (config, bundle)),
            'agent motion requires robocasa365_sim')
    require(not any(k in obj for obj in (descriptor, permission) for k in ('actor', 'approval_id')),
            'agent motion cannot carry actor or approval_id')
    require(descriptor.get('agent_execution') == dict(request_id=job['receipt'].get('agent_request_id')),
            'agent motion request differs from execute job')
    inventory = operations(source, require)
    require(bool(inventory), 'agent motion requires a motion inventory')
    expected = dict(run_id='run:' + descriptor['cell'], cell=descriptor['cell'], code_version=descriptor['version'],
                    code_sha256=hashlib.sha256(source).hexdigest(),
                    config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                    action='agent_dev_tier_motion', action_version=1, execution_target='robocasa365_sim',
                    risk='bounded simulated robot motion under existing SDK safety limits',
                    scope=dict(operations=inventory, robot=config['robot'], task_id=config['task_id'], seed=config['seed'],
                               sdk_calls=config['sdk_call_limit'], control_steps=config['horizon'], wall_seconds=config['attempt_wall_seconds']))
    require(permission == expected, 'agent motion scope differs from frozen source/config binding')


def require_result_permission(descriptor, require):
    require(descriptor.get('import_purpose') != PURPOSE
            and descriptor.get('permission_scope', {}).get('action') != ACTION,
            'Memory validation-only evidence cannot record results or trusted use')


def public_summary(store, memory_id, provenance):
    """Only an action/actor label leaves the importer evidence projection."""
    with sqlite3.connect(store.path) as db:
        row = db.execute('SELECT payload FROM dev_v2_imports WHERE id=?', (provenance.get('source_import_id'),)).fetchone()
        source = json.loads(row[0]) if row else {}
        ids = {source.get('failed_execution_import_id')}
        targets = set(db.execute('SELECT cell,version FROM dev_v2_validation_grants WHERE memory_id=?', (memory_id,)))
        origin = provenance.get('origin')
        if origin:targets.add((origin['cell'], origin['origin_version']))
        for raw, in db.execute('SELECT payload FROM memory_cross_seed_probes WHERE memory_id=?', (memory_id,)):
            ids.add(json.loads(raw)['request']['execution_import_id'])
        rows = [db.execute('SELECT payload FROM dev_v2_imports WHERE id=? AND kind=\'execution\'', (key,)).fetchone() for key in ids if key]
        rows += [db.execute('SELECT payload FROM dev_v2_imports WHERE attempt_id=? AND kind=\'execution\'',
                            (f'attempt:{cell}:{version-1}',)).fetchone() for cell, version in targets]
        for row in rows:
            if row and json.loads(row[0]).get('permission', {}).get('action') == ACTION:
                return dict(action=ACTION, actor=ACTION)
    return None
