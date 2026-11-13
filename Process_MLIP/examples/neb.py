"""Edit the paths and run one NEB screening with an existing model."""
from Process_MLIP import model_config, run_mlip_neb

STRUCTURE = 'deintercalated.vasp'
PARENT = 'parent_site_catalog.vasp'
OUTPUT = 'neb_results'
MODEL = model_config('mace', model_path='/path/to/mace-omat-0-medium.model',
                     head=None, device='cuda', default_dtype='float32')

if __name__ == '__main__':
    print(run_mlip_neb(STRUCTURE, PARENT, OUTPUT, model=MODEL, mobile_ion='Na',
                       random_num=3, max_distance=3.5, seed=42)['summary_csv'])
