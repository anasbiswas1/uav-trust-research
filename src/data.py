"""Data loading, schema detection, and attack-family-held-out splitting for
UAV network-IDS datasets.
"""
import glob, os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

LABEL_NAMES = {"label","class","attack","category","type","target","attack_type","traffic_type"}
NORMAL_NAMES = {"normal","benign","0","none","clean"}
KNOWN_CLASSES = {"normal","blackhole","flooding","sybil","wormhole","benign","attack","dos","ddos","replay","fuzzy"}

def load_csvs(data_dir):
    csvs=glob.glob(os.path.join(data_dir,"**/*.csv"),recursive=True)
    if not csvs: raise FileNotFoundError("No CSV under "+data_dir)
    frames=[pd.read_csv(c,low_memory=False) for c in csvs]
    widest=max(frames,key=lambda d:d.shape[1]).columns
    keep=[f for f in frames if set(widest).issubset(f.columns)]
    df=pd.concat(keep,ignore_index=True) if len(keep)>1 else frames[0]
    df.columns=[str(c).strip() for c in df.columns]; return df

def detect_schema(df,label_col=None,normal_value=None):
    if label_col is None:
        cands=[c for c in df.columns if c.lower() in LABEL_NAMES]
        if not cands:
            for c in df.columns:
                vals=set(str(v).strip().lower() for v in df[c].dropna().unique()[:50])
                if len(vals & KNOWN_CLASSES)>=2: cands.append(c)
        if not cands: raise ValueError("pass label_col")
        label_col=cands[0]
    if normal_value is None:
        for v in df[label_col].unique():
            if str(v).strip().lower() in NORMAL_NAMES: normal_value=v
        if normal_value is None: raise ValueError("pass normal_value")
    families=[v for v in df[label_col].unique() if v!=normal_value]
    return label_col,normal_value,families

def _split_idx(idx,fracs,seed):
    idx=np.array(idx); rng=np.random.default_rng(seed); rng.shuffle(idx)
    out,start,keys={},0,list(fracs)
    for i,k in enumerate(keys):
        stop=len(idx) if i==len(keys)-1 else start+int(round(fracs[k]*len(idx)))
        out[k]=idx[start:stop]; start=stop
    return out

def prepare_splits(df,label_col,normal_value,held_out_family,drop_patterns,
                   normal_fracs,family_fracs,seed,max_categories=50,numeric_coerce_frac=0.95):
    """Attack-family-held-out splits with deterministic feature cleaning.

    A column is kept as numeric if at least numeric_coerce_frac of its non-null values
    parse as numbers (real features misread as text, e.g. frame.len). Any column that
    does NOT meet that bar is treated as categorical, and is one-hot encoded only if it
    has at most max_categories distinct values; otherwise it is dropped as a
    high-cardinality identifier (e.g. wlan.tag, ip.ttl, arp.hw.type). This depends only
    on parse rate and categorical cardinality, never on numeric distinct-count, so the
    feature space is identical on full data and on any subsample.
    """
    families=[v for v in df[label_col].unique() if v!=normal_value]
    match=[f for f in families if str(f).strip().lower()==str(held_out_family).strip().lower()]
    held_out=match[0] if match else sorted(families,key=lambda f:(df[label_col]==f).sum())[0]
    seen_families=[f for f in families if f!=held_out]

    drop_cols=[c for c in df.columns if c!=label_col and any(p in c.lower() for p in drop_patterns)]
    feat=df.drop(columns=drop_cols+[label_col]).copy()

    # coerce mostly-numeric columns; the rest stay categorical
    coerced=[]
    for c in feat.columns:
        if not pd.api.types.is_numeric_dtype(feat[c]):
            conv=pd.to_numeric(feat[c],errors="coerce"); nn=feat[c].notna().sum()
            if nn>0 and conv.notna().sum()>=numeric_coerce_frac*nn: feat[c]=conv; coerced.append(c)

    const=[c for c in feat.columns if feat[c].nunique(dropna=False)<=1]; feat=feat.drop(columns=const)
    cat=[c for c in feat.columns if not pd.api.types.is_numeric_dtype(feat[c])]
    high_card=[c for c in cat if feat[c].nunique(dropna=False)>max_categories]   # junk that failed coercion
    if high_card: feat=feat.drop(columns=high_card); cat=[c for c in cat if c not in high_card]
    feat=pd.get_dummies(feat,columns=cat,dummy_na=False).replace([np.inf,-np.inf],np.nan).fillna(0.0)

    y=(df[label_col].values!=normal_value).astype(int); X=feat.values.astype(float); fam=df[label_col].values
    tr,ca,seen,shift=[],[],[],[]
    ns=_split_idx(np.where(fam==normal_value)[0],normal_fracs,seed)
    tr+=list(ns["train"]); ca+=list(ns["cal"]); seen+=list(ns["test_seen"]); shift+=list(ns["test_shift"])
    for j,f in enumerate(seen_families):
        fs=_split_idx(np.where(fam==f)[0],family_fracs,seed+j+1)
        tr+=list(fs["train"]); ca+=list(fs["cal"]); seen+=list(fs["test_seen"])
    shift+=list(np.where(fam==held_out)[0])
    tr,ca,seen,shift=(np.array(sorted(a)) for a in (tr,ca,seen,shift))
    scaler=StandardScaler().fit(X[tr]); tf=lambda ix:scaler.transform(X[ix])
    return {"held_out":held_out,"seen_families":seen_families,
        "dropped":{"id_leakage":drop_cols,"constant":const,"high_cardinality":high_card,
                   "encoded":cat,"coerced_numeric":coerced},
        "feature_names":list(feat.columns),
        "X_train":tf(tr),"y_train":y[tr],"fam_train":fam[tr],
        "X_cal":tf(ca),"y_cal":y[ca],
        "X_seen":tf(seen),"y_seen":y[seen],"fam_seen":fam[seen],
        "X_shift":tf(shift),"y_shift":y[shift],"fam_shift":fam[shift],
        "n":{"train":len(tr),"cal":len(ca),"seen":len(seen),"shift":len(shift)}}
