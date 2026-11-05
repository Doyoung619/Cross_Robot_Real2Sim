"""Fresh stage passes preserve evidence without reusing stale learned results."""
import importlib
import json
from pathlib import Path


def test_blocked_stage4_archives_old_checkpoint_and_evaluation(tmp_path, monkeypatch):
    module=importlib.import_module('cross_robot_real2sim.stage4_policy.run')
    output=tmp_path/'04_policy'
    output.mkdir()
    (output/'checkpoint.pt').write_bytes(b'explicit stale test checkpoint')
    (output/'eval_metrics.json').write_text('{"fixture":"old evaluation"}')
    (output/'eval_rollouts').mkdir()
    (output/'eval_rollouts'/'old.npz').write_bytes(b'explicit stale rollout fixture')
    (output/'status.json').write_text('{"status":"running"}')
    (tmp_path/'videos').mkdir()
    (tmp_path/'videos'/'target_bc.mp4').write_bytes(b'explicit stale video fixture')
    monkeypatch.setattr(module,'build_dataset',lambda *a,**kw:{'status':'blocked_insufficient_teacher_data','samples':0})
    result=module.run({'seed':42,'mode':'bootstrap'},tmp_path)
    assert result['training_status']=='blocked_insufficient_teacher_data'
    assert not (output/'checkpoint.pt').exists()
    assert not (output/'eval_metrics.json').exists()
    assert not (output/'eval_rollouts').exists()
    archives=list((output/'history').iterdir())
    assert len(archives)==1
    assert (archives[0]/'checkpoint.pt').read_bytes()==b'explicit stale test checkpoint'
    assert (archives[0]/'eval_rollouts'/'old.npz').exists()
    assert json.loads((output/'status.json').read_text())['status']=='completed_degraded'
    assert not (tmp_path/'videos'/'target_bc.mp4').exists()
    assert (archives[0]/'target_bc.mp4').exists()
    assert json.loads((output/'metrics.json').read_text())['training_status']=='blocked_insufficient_teacher_data'


def test_stage5_archives_previous_feedback_before_new_diagnosis(tmp_path, monkeypatch):
    module=importlib.import_module('cross_robot_real2sim.stage5_diagnosis.run')
    output=tmp_path/'05_diagnosis'
    stale=output/'feedback_run'/'03_retarget'/'rollouts'
    stale.mkdir(parents=True)
    (stale/'feedback_old.npz').write_bytes(b'explicit stale feedback fixture')
    (output/'additional_teacher.json').write_text('[{"fixture":"old teacher"}]')
    (output/'before_after.json').write_text('{"fixture":"old correction"}')
    def correction(config,run_dir,rows):
        assert not (output/'feedback_run').exists()
        assert not (output/'additional_teacher.json').exists()
        assert not (output/'before_after.json').exists()
        return {'status':'intervention_not_executed','reason':'explicit no-rollout test fixture'}
    monkeypatch.setattr(module,'corrective_iteration',correction)
    comparison=importlib.import_module('cross_robot_real2sim.viz.comparison_video')
    monkeypatch.setattr(comparison,'make_comparison',lambda *a,**kw:None)
    result=module.run({'seed':42,'mode':'bootstrap'},tmp_path)
    assert result['episodes_analyzed']==0
    archives=list((output/'history').iterdir())
    assert len(archives)==1
    assert (archives[0]/'feedback_run'/'03_retarget'/'rollouts'/'feedback_old.npz').exists()
    assert not (output/'feedback_run').exists()
    assert json.loads((output/'before_after.json').read_text())['status']=='intervention_not_executed'


def test_upstream_failure_cannot_reuse_previous_task_or_policy(tmp_path, monkeypatch):
    pipeline=importlib.import_module('cross_robot_real2sim.orchestrator.run_pipeline')
    selection=importlib.import_module('cross_robot_real2sim.stage1_selection.run')
    monkeypatch.setattr(pipeline,'input_signature',lambda *a,**kw:{'human_videos':[]})
    def failing_selection(*args,**kwargs):
        raise ValueError('explicit source-selection failure fixture')
    monkeypatch.setattr(selection,'run',failing_selection)
    for folder,name in [('01_selection','selection.json'),('02_task','task_spec.json'),
                        ('03_retarget','stats.json'),('04_policy','eval_metrics.json')]:
        path=tmp_path/folder/name
        path.parent.mkdir(parents=True)
        path.write_text('{"fixture":"stale prior pass"}')
    status=pipeline.run_pipeline({'mode':'bootstrap','seed':42},tmp_path)
    assert status['status']=='failed'
    assert all(status['stages'][str(step)]['status']=='skipped_missing_input' for step in (2,3,4))
    assert not (tmp_path/'02_task/task_spec.json').exists()
    assert not (tmp_path/'03_retarget/stats.json').exists()
    assert not (tmp_path/'04_policy/eval_metrics.json').exists()
    assert list((tmp_path/'04_policy/history').glob('*/eval_metrics.json'))
