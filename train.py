import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import read_manifest, SareeDataset, IdentityBatchSampler
from model import DesignEmbedder, supervised_contrastive_loss


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True); ap.add_argument('--data-root',default='.')
    ap.add_argument('--output-dir',required=True); ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--batches-per-epoch',type=int,default=100)
    ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--cpu-threads',type=int,default=4); ap.add_argument('--finetune-last-blocks',type=int,default=2)
    ap.add_argument('--pretrained',type=lambda x:x.lower()=='true',default=True); ap.add_argument('--seed',type=int,default=13)
    a=ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type=='cpu': torch.set_num_threads(a.cpu_threads)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=read_manifest(a.manifest,a.data_root); train=df[df.split=='train'].copy()
    if train.empty: raise ValueError('Manifest has no train rows')
    ids=sorted(train.design_id.unique()); idmap={s:i for i,s in enumerate(ids)}; train['label']=train.design_id.map(idmap)
    # Use P×K batches; choose P and K so their product does not exceed requested size.
    p=max(2,min(8,train.label.nunique())); k=max(2,a.batch_size//p)
    sampler=IdentityBatchSampler(train.label.tolist(),p=p,k=k,batches=a.batches_per_epoch)
    ds=SareeDataset(train,a.data_root,train=True)
    dl=DataLoader(ds,batch_sampler=sampler,num_workers=a.workers,pin_memory=dev.type=='cuda')
    model=DesignEmbedder(pretrained=a.pretrained).to(dev)
    for param in model.features.parameters(): param.requires_grad=False
    if a.finetune_last_blocks>0:
        for block in list(model.features.children())[-a.finetune_last_blocks:]:
            for param in block.parameters(): param.requires_grad=True
    print(f'device={dev} parameters={sum(x.numel() for x in model.parameters()):,} train_images={len(train)} designs={len(ids)}')
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr,weight_decay=1e-4)
    val=df[df.split=='val'].copy()
    has_val=not val.empty
    if has_val:
        # Validation identities can be disjoint from the train identities.
        val['label']=val.design_id.astype('category').cat.codes
        vloader=DataLoader(SareeDataset(val,a.data_root,train=False),batch_size=a.batch_size,
                           shuffle=False,num_workers=a.workers,pin_memory=dev.type=='cuda')
    best=-1.0 if has_val else float('inf')
    for epoch in range(a.epochs):
        model.train(); losses=[]
        for x,y,_ in dl:
            x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True); loss=supervised_contrastive_loss(model(x),y)
            loss.backward(); opt.step(); losses.append(loss.item())
        mean=float(np.mean(losses)); score=mean
        val_r1=None
        if has_val:
            model.eval(); zv=[]; yv=[]
            with torch.inference_mode():
                for x,y,_ in vloader:
                    zv.append(model(x.to(dev)).cpu()); yv.append(y)
            z=torch.cat(zv); y=torch.cat(yv)
            sim=z@z.T; sim.fill_diagonal_(float('-inf'))
            val_r1=float((y[sim.argmax(dim=1)]==y).float().mean().item())
            score=val_r1
        print(f'epoch={epoch+1}/{a.epochs} train_supcon={mean:.4f} val_recall@1={val_r1}')
        improved=(score>best) if has_val else (score<best)
        if improved:
            best=score
            torch.save({'state_dict':model.state_dict(),'embedding_dim':256,'pretrained':a.pretrained,
                        'design_ids':ids,'input_size':224},out/'best.pt')
    (out/'train_summary.json').write_text(json.dumps({'device':str(dev),'parameters':sum(x.numel() for x in model.parameters()),
      'train_images':len(train),'train_designs':len(ids),'best_selection_score':best,
      'finetune_last_blocks':a.finetune_last_blocks,'cpu_threads':a.cpu_threads if dev.type=='cpu' else None,
      'selection':'validation nearest-neighbor Recall@1' if has_val else 'training supervised contrastive loss'},indent=2))

if __name__=='__main__': main()
