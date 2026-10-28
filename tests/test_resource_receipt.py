import pytest
from attention_memory_service.resource_receipt import valid,bounds

@pytest.mark.parametrize('tier,calls,seconds,steps,accepted',[
    ('dev',86,550,3000,True),('dev',320,1980,9600,True),('dev',321,550,1,False),
    ('dev',86,1980.01,1,False),('dev',1,1,9601,False),('legacy',81,1,1,False),
    (None,81,1,1,False),('formal',81,1,1,False)])
def test_receipt_bounds(tier,calls,seconds,steps,accepted):
    config=dict(sdk_call_limit=400,horizon=10000,attempt_wall_seconds=3000)
    if tier:config['budget_tier']=tier
    assert valid(config,dict(sdk_call_budget=dict(dispatched=calls),control_steps=steps,elapsed_seconds=seconds)) is accepted

def test_config_limits_and_formal_boundary():
    config=dict(budget_tier='dev',sdk_call_limit=30,horizon=200,attempt_wall_seconds=300)
    assert bounds(config)==(30,200,360)
    assert bounds(dict(config,sdk_call_limit=320,horizon=9600,attempt_wall_seconds=1920,runner_boundary='formal'))==(80,2400,540)
