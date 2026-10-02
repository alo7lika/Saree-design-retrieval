import argparse,json
from pathlib import Path
from common import load_model,embed_one


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--image-a',required=True)
    ap.add_argument('--image-b',required=True); ap.add_argument('--threshold',required=True); a=ap.parse_args()
    model,device,state=load_model(a.checkpoint); x=embed_one(model,a.image_a,device); y=embed_one(model,a.image_b,device)
    score=float(x@y); data=json.loads(Path(a.threshold).read_text()); threshold=float(data['threshold'])
    ids=state.get('design_ids',[])
    scope=('category_proxy' if ids and all(str(x).startswith('CATEGORY_PROXY_') for x in ids)
           else 'visual_design_labels' if ids and all(str(x).startswith('D') for x in ids)
           else 'provided_design_ids')
    print(json.dumps({'same_label':score>=threshold,'label_scope':scope,'cosine_similarity':score,'threshold':threshold,
                      'threshold_source':data.get('source','unknown')},indent=2))

if __name__=='__main__': main()
