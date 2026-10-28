"""Receipt ceilings selected only from the sealed execution configuration."""
import math


def bounds(config):
    dev = (config.get('budget_tier') == 'dev'
           and config.get('runner_boundary', 'dev_loop') == 'dev_loop'
           and config.get('budget_boundary', 'dev_loop') == 'dev_loop')
    calls, steps, seconds = (320, 9600, 1920) if dev else (80, 2400, 480)
    return (min(calls, config['sdk_call_limit']), min(steps, config['horizon']),
            min(seconds, config['attempt_wall_seconds']) + 60)


def valid(config, receipt):
    calls, steps, seconds = bounds(config)
    dispatched = receipt['sdk_call_budget']['dispatched']
    controls = receipt['control_steps']
    elapsed = receipt['elapsed_seconds']
    return (type(dispatched) is int and 0 <= dispatched <= calls
            and type(controls) is int and 0 <= controls <= steps
            and type(elapsed) in (int, float) and math.isfinite(elapsed)
            and 0 <= elapsed <= seconds)
