"""Check that frozen workflow settings reach the requested atomate2 stage."""
import sys
from types import ModuleType
from unittest.mock import patch

from Process_Vasp.workflows.atomate_runner import _job


def test_stage_run_options_reach_maker():
    jobs = ModuleType('atomate2.vasp.jobs.core')
    sets = ModuleType('atomate2.vasp.sets.core')
    captured = []

    class Maker:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def make(self, structure, **kwargs):
            return structure

    class Generator:
        def __init__(self, **kwargs):
            pass

    jobs.RelaxMaker = jobs.StaticMaker = jobs.NonSCFMaker = Maker
    sets.RelaxSetGenerator = sets.StaticSetGenerator = sets.NonSCFSetGenerator = Generator
    options = {'max_errors': 2, 'custodian_kwargs': {'max_errors_per_job': 2}}
    with patch.dict(sys.modules, {'atomate2.vasp.jobs.core': jobs, 'atomate2.vasp.sets.core': sets}):
        _job('static', 'structure', None, {'run_vasp_kwargs': {'static': options}})
        _job('relax', 'structure', None, {'run_vasp_kwargs': {'static': options}})
    assert captured[0]['run_vasp_kwargs'] == options
    assert 'run_vasp_kwargs' not in captured[1]
