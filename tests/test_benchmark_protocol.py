"""Protocol safeguards: physical budgets, leakage, uncertainty and retrieval."""
import json
from pathlib import Path

import numpy as np
import pytest

from cross_robot_real2sim.benchmark.core import Ledger,feedback_envelope,require_feedback,validate,wilson
from cross_robot_real2sim.benchmark.retrieval import subsequence_dtw


def test_budget_is_charged_at_first_physics_even_if_episode_is_interrupted(tmp_path):
    log=Ledger(tmp_path/'ledger.jsonl')
    assert log.count('adaptation')==0
    log.begin('adaptation','our_full',0,'adaptation_0000','weights',budget_cap=1)
    # No completion event: this episode still consumed an interaction.
    assert log.count('adaptation')==1
    with pytest.raises(RuntimeError,match='budget'):log.begin('adaptation','our_full',0,'adaptation_0001','weights',budget_cap=1)
    assert log.count('adaptation')==1


def test_roles_and_training_seeds_have_separate_exact_counts(tmp_path):
    log=Ledger(tmp_path/'ledger.jsonl')
    for role in ['adaptation','development','evaluation']:
        episode=log.begin(role,'method',3,'state','weights');log.end(episode,physical_steps=220)
    assert [log.count(r) for r in ['adaptation','development','evaluation']]==[1,1,1]
    assert log.count('adaptation','method',4)==0


@pytest.mark.parametrize('role',['evaluation','development','smoke'])
def test_evaluation_observations_cannot_enter_feedback(role):
    with pytest.raises(ValueError):feedback_envelope(dict(role=role),[dict(action=[1]*7)])
    with pytest.raises(ValueError):require_feedback(dict(role=role))


def test_binomial_interval_does_not_report_zero_uncertainty_for_zero_success():
    lo,hi=wilson(0,50);assert lo==0;assert hi==pytest.approx(.071347599,abs=1e-8)
    lo,hi=wilson(50,50);assert lo==pytest.approx(.928652401,abs=1e-8);assert hi==1
    assert wilson(0,0)==(None,None)


def test_subsequence_matching_finds_a_segment_not_only_whole_trajectory():
    query=np.array([[2],[3],[4]],float);candidate=np.array([[-9],[-8],[2],[3],[4],[20]],float)
    result=subsequence_dtw(query,candidate)
    assert result['cost']==0;assert (result['start'],result['end'])==(2,5)


def test_fert_slope_constraint_advances_only_one_or_two_frames():
    result=subsequence_dtw(np.array([[0],[2],[4]],float),np.arange(6)[:,None],slope=True)
    assert result['cost']==0
    assert set(np.diff(result['path']))<={1,2}
    assert subsequence_dtw(np.ones((5,1)),np.ones((2,1)),slope=True) is None


def test_requested_profiles_are_valid_and_preserve_the_full_budget_grid():
    root=Path(__file__).resolve().parents[1]
    for name in ['smoke','pilot','full']:validate(json.loads((root/f'configs/benchmark/{name}.json').read_text()))
    full=json.loads((root/'configs/benchmark/full.json').read_text())
    assert full['budgets']==[0,1,2,4,8,16];assert full['evaluation_states']==50


def test_insufficient_unique_adaptation_states_are_rejected():
    root=Path(__file__).resolve().parents[1];config=json.loads((root/'configs/benchmark/full.json').read_text());config['adaptation_states']=2
    with pytest.raises(ValueError,match='adaptation states'):validate(config)


def test_statistics_record_does_not_mutate_when_later_training_cost_is_added(tmp_path):
    from cross_robot_real2sim.benchmark.experiment import checkpoint_record
    costs=[dict(phase='initial',wall_seconds=1)]
    row=checkpoint_record(tmp_path,tmp_path,'method',0,0,0,None,None,100,[],costs)
    costs.append(dict(phase='future',wall_seconds=2))
    assert len(row['cost_events'])==1


def test_static_generation_has_no_target_physics_entry_point():
    import ast
    from cross_robot_real2sim.benchmark import data
    tree=ast.parse(Path(data.__file__).read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['step','mj_step'] for n in ast.walk(tree))
