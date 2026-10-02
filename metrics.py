import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def retrieval_metrics(query_z, query_ids, gallery_z, gallery_ids, ks=(1, 5)):
    sim = query_z @ gallery_z.T
    out = {f"recall@{k}": [] for k in ks}; aps = []
    for row, target in zip(sim, query_ids):
        order = np.argsort(-row); relevant = (gallery_ids[order] == target)
        if not relevant.any(): continue
        for k in ks: out[f"recall@{k}"].append(float(relevant[:k].any()))
        ranks = np.flatnonzero(relevant) + 1
        aps.append(float(np.mean(np.arange(1, len(ranks)+1) / ranks)))
    if not aps: raise ValueError("No query design has a gallery match")
    return {k: float(np.mean(v)) for k, v in out.items()} | {"mAP": float(np.mean(aps)), "queries": len(aps)}


def pair_metrics(emb, ids, threshold=None):
    rng = np.random.default_rng(17); n = len(ids)
    pairs, labels = [], []
    by_id = {}
    for i, y in enumerate(ids): by_id.setdefault(y, []).append(i)
    positive = [(a,b) for v in by_id.values() for a in v for b in v if a < b]
    negative = [(a,b) for a in range(n) for b in range(a+1,n) if ids[a] != ids[b]]
    if not positive or not negative: raise ValueError("Verification needs positive and negative pairs")
    count = min(len(positive), len(negative), 10000)
    pairs = rng.choice(len(positive), count, replace=False).tolist()
    pp = [positive[i] for i in pairs]
    nn = [negative[i] for i in rng.choice(len(negative), count, replace=False)]
    pairs = pp + nn; labels = np.r_[np.ones(count), np.zeros(count)]
    scores = np.array([emb[a] @ emb[b] for a,b in pairs])
    auc = float(roc_auc_score(labels, scores)); fpr,tpr,th = roc_curve(labels, scores)
    fnr = 1-tpr; ix = int(np.argmin(np.abs(fpr-fnr))); eer=float((fpr[ix]+fnr[ix])/2)
    selected = float(th[ix]) if threshold is None else float(threshold)
    acc = float(accuracy_score(labels, scores >= selected))
    return {"roc_auc": auc, "eer": eer, "threshold": selected, "accuracy": acc,
            "pairs_per_class": count, "fpr_at_threshold": float(fpr[ix]) if threshold is None else None}


def cross_split_pair_metrics(gallery_z, gallery_ids, query_z, query_ids, threshold=None,
                             gallery_colorways=None, query_colorways=None, cross_palette_only=False):
    """Balanced verification from all gallery×query pairs; optionally keep only cross-palette positives."""
    same=np.asarray(gallery_ids)[:,None] == np.asarray(query_ids)[None,:]
    if cross_palette_only:
        if gallery_colorways is None or query_colorways is None:
            raise ValueError('Cross-palette evaluation requires annotated colorway IDs')
        gc=np.asarray(gallery_colorways).astype(str)[:,None]
        qc=np.asarray(query_colorways).astype(str)[None,:]
        known=(gc!='unknown_unannotated') & (qc!='unknown_unannotated')
        same &= known & (gc!=qc)
    pos=np.argwhere(same); neg=np.argwhere(~(np.asarray(gallery_ids)[:,None] == np.asarray(query_ids)[None,:]))
    if not len(pos) or not len(neg): raise ValueError('Verification needs positive and negative gallery-query pairs')
    n=min(len(pos),len(neg),10000); rng=np.random.default_rng(17)
    pos=pos[rng.choice(len(pos),n,replace=False)]; neg=neg[rng.choice(len(neg),n,replace=False)]
    pairs=np.concatenate([pos,neg]); labels=np.r_[np.ones(n),np.zeros(n)]
    scores=np.array([gallery_z[i] @ query_z[j] for i,j in pairs])
    auc=float(roc_auc_score(labels,scores)); fpr,tpr,th=roc_curve(labels,scores)
    ix=int(np.argmin(np.abs(fpr-(1-tpr)))); eer=float((fpr[ix]+1-tpr[ix])/2)
    selected=float(th[ix]) if threshold is None else float(threshold)
    return {'roc_auc':auc,'eer':eer,'threshold':selected,'accuracy':float(accuracy_score(labels,scores>=selected)),
            'pairs_per_class':n,'fpr_at_threshold':float(fpr[ix]) if threshold is None else None}
