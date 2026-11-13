"""Screened task skills, derived from REPORT-stage2.md (2026-10-30).

This versioned table is authoritative; harness code imports it. Family labels
are for held-out/ablation analysis and never determine skill applicability.
"""
VERSION = 'screened-skills-v1-20261113'
TASKS = {
    'open_cabinet': ('A', ('pull_handle:cabinet_door',)),
    'close_blender_lid': ('A', ('close_lid:blender',)),
    'open_drawer': ('A', ('pull_handle:drawer',)),
    'turn_on_microwave': ('B', ('press_button:microwave',)),
    'turn_off_stove': ('B', ('twist_knob:stove',)),
    'open_stand_mixer_head': ('B', ('press_button:stand_mixer',)),
    'coffee_setup_mug': ('C', ('pick_place:coffee_machine',)),
    'prepare_coffee': ('C', ('carry_out_of:cabinet', 'pick_place:cabinet_interior',
                              'pick_place:coffee_machine', 'press_button:coffee_machine')),
    'kettle_boiling': ('C', ('pick_place:stove', 'twist_knob:stove')),
    'heat_kebab_sandwich': ('C', ('pick_place:oven_interior', 'close_door:oven', 'twist_knob:oven')),
    'pick_place_counter_to_cabinet': ('D', ('pick_place:cabinet_interior',)),
    'categorize_condiments': ('D', ('pick_place:cabinet_interior',)),
    'weigh_ingredients': ('D', ('carry_out_of:cabinet', 'pick_place:cabinet_interior',
                                  'pick_place:scale', 'close_door:cabinet')),
    'arrange_tea': ('D', ('pick_place:tray', 'carry_out_of:cabinet',
                         'pick_place:cabinet_interior', 'close_door:cabinet')),
}


def tags_for(task_id):
    return TASKS.get(task_id, (None, ()))[1]
