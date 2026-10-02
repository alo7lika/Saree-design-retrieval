import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import load_model,embed_frame,embed_one


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--gallery-csv',required=True)
    ap.add_argument('--query',required=True); ap.add_argument('--data-root',default='.'); ap.add_argument('--top-k',type=int,default=5)
    a=ap.parse_args(); frame=pd.read_csv(a.gallery_csv)
    if not {'path','design_id'}.issubset(frame.columns): raise ValueError('Gallery CSV needs path,design_id columns')
    model,device,_=load_model(a.checkpoint)
    gz=embed_frame(model,frame,a.data_root,device); q=embed_one(model,a.query,device)
    sims=gz@q; order=np.argsort(-sims)[:max(1,a.top_k)]
    best={}
    for i in order:
        y=str(frame.iloc[i].design_id)
        best[y]=max(best.get(y,-1),float(sims[i]))
    for rank,(design,score) in enumerate(sorted(best.items(),key=lambda p:-p[1])[:a.top_k],1):
        print(f'{rank}\t{design}\tcosine={score:.5f}')

if __name__=='__main__': main()
