import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import read_manifest
from common import load_model,embed_frame
from metrics import retrieval_metrics,pair_metrics,cross_split_pair_metrics


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--data-root',default='.')
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--cpu-threads',type=int,default=4)
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=read_manifest(a.manifest,a.data_root); model,dev,state=load_model(a.checkpoint)
    if dev.type=='cpu': torch.set_num_threads(a.cpu_threads)
    gal=df[df.split=='gallery'].reset_index(drop=True); qry=df[df.split=='query'].reset_index(drop=True)
    if gal.empty or qry.empty: raise ValueError('Manifest must include gallery and query rows')
    known=lambda s: not pd.isna(s) and str(s).strip().lower() not in {'unknown','unknown_unannotated','na','n/a','none',''}
    gcolor=set(gal.loc[gal.colorway_id.map(known),'colorway_id'].astype(str))
    qcolor=set(qry.loc[qry.colorway_id.map(known),'colorway_id'].astype(str))
    audit={'gallery_images':len(gal),'query_images':len(qry),'gallery_designs':int(gal.design_id.nunique()),
           'query_designs':int(qry.design_id.nunique()),'design_overlap':int(len(set(gal.design_id)&set(qry.design_id))),
           'gallery_colorways':len(gcolor),'query_colorways':len(qcolor),
           'shared_colorway_ids':len(gcolor&qcolor),'colorway_annotations_available':bool(gcolor or qcolor),
           'identity_scope':'category_proxy' if all(str(x).startswith('CATEGORY_PROXY_') for x in df.design_id.unique()) else ('visual_design_labels' if all(str(x).startswith('D') for x in df.design_id.unique()) else 'provided_design_ids'),
           'query_designs_missing_from_gallery':sorted(set(qry.design_id)-set(gal.design_id))}
    (out/'split_audit.json').write_text(json.dumps(audit,indent=2))
    if audit['query_designs_missing_from_gallery']: raise ValueError('Every query design must occur in gallery')
    gz=embed_frame(model,gal,a.data_root,dev,workers=a.workers); qz=embed_frame(model,qry,a.data_root,dev,workers=a.workers)
    ident=retrieval_metrics(qz,np.array(qry.design_id),gz,np.array(gal.design_id))
    # Pair verification threshold calibration: prefer val data; otherwise explicitly label query development estimate.
    val=df[df.split=='val'].reset_index(drop=True); threshold_source='validation pairs'
    if len(val)>=2:
        vz=embed_frame(model,val,a.data_root,dev,workers=a.workers)
        vm=pair_metrics(vz,np.array(val.design_id)); threshold=vm['threshold']
    else:
        # This threshold selection uses query labels and must never be called blind test verification.
        combined=pd.concat([gal,qry],ignore_index=True); cz=embed_frame(model,combined,a.data_root,dev,workers=a.workers)
        vm=pair_metrics(cz,np.array(combined.design_id)); threshold=vm['threshold']; threshold_source='development gallery+query pairs (optimistic; validation split absent)'
    test=cross_split_pair_metrics(gz,np.array(gal.design_id),qz,np.array(qry.design_id),threshold=threshold)
    result={'identification':ident,'verification_gallery_query':test,'verification_calibration':vm,
            'verification_threshold_source':threshold_source}
    if audit['colorway_annotations_available']:
        cross=cross_split_pair_metrics(gz,np.array(gal.design_id),qz,np.array(qry.design_id),threshold=threshold,
            gallery_colorways=np.array(gal.colorway_id),query_colorways=np.array(qry.colorway_id),cross_palette_only=True)
        result['verification_cross_palette']=cross
    (out/'threshold.json').write_text(json.dumps({'threshold':threshold,'source':threshold_source},indent=2))
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    # Latency benchmark excludes image decode and first warmup.
    model.eval(); dummy=torch.randn(1,3,224,224,device=dev)
    for _ in range(20): model(dummy)
    if dev.type=='cuda': torch.cuda.synchronize()
    timings=[]
    for _ in range(100):
        t=time.perf_counter(); model(dummy)
        if dev.type=='cuda': torch.cuda.synchronize()
        timings.append((time.perf_counter()-t)*1000)
    timings=np.asarray(timings)
    nparams=sum(p.numel() for p in model.parameters())
    (out/'efficiency.json').write_text(json.dumps({'parameters':nparams,'embedding_dimensions':state.get('embedding_dim',256),
      'latency_ms_batch1_model_only_p50':float(np.median(timings),),'latency_ms_batch1_model_only_mean':float(timings.mean()),
      'device':str(dev),'input':'1x3x224x224','threads':a.cpu_threads if dev.type=='cpu' else None},indent=2))
    print(json.dumps({'split_audit':audit,'metrics':result},indent=2))

if __name__=='__main__': main()
