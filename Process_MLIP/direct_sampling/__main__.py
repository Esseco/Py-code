import argparse, json
from .features import build_encoder
from .pca import structures_to_pca_csv
from .workflow import extract, run
p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
e=sub.add_parser('extract'); e.add_argument('trajectories',nargs='+'); e.add_argument('--model',required=True); e.add_argument('--encoder',choices=['m3gnet','mace'],default='m3gnet'); e.add_argument('--cache',required=True); e.add_argument('--device',default='cpu'); e.add_argument('--head')
s=sub.add_parser('sample'); s.add_argument('--cache',required=True); s.add_argument('--output',required=True); s.add_argument('--n-samples',type=int,required=True); s.add_argument('--threshold',type=float,default=0.5); s.add_argument('--seed',type=int,default=42)
c=sub.add_parser('pca'); c.add_argument('structures',nargs='+'); c.add_argument('--model',required=True); c.add_argument('--encoder',choices=['m3gnet','mace'],default='m3gnet'); c.add_argument('--output',required=True); c.add_argument('--device',default='cpu'); c.add_argument('--head')
a=p.parse_args()
if a.command=='extract':
    if a.encoder=='m3gnet' and (a.device!='cpu' or a.head is not None): p.error('--device and --head apply only to MACE; TensorFlow controls M3GNet device visibility')
    extract(a.trajectories, build_encoder(a.model,a.encoder,device=a.device,head=a.head), a.cache)
elif a.command=='sample':
    run(a.cache,a.output,a.n_samples,threshold=a.threshold,seed=a.seed)
else:
    print(json.dumps(structures_to_pca_csv(a.structures,a.output,model=a.model,encoder=a.encoder,device=a.device,head=a.head),indent=2))
