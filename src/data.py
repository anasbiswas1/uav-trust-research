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
                   normal_fracs,family_fracs,seed,max_categories=50,
                   numeric_coerce_frac=0.95,max_distinct=50):
    """Attack-family-held-out splits with deterministic feature cleaning.

    Cleaning order: (1) drop leakage/id columns by name; (2) drop ANY column whose
    distinct-value count exceeds max_distinct, numeric or not, so high-cardinality
    identifiers (e.g. wlan.tag, frame.number) cannot survive by parsing as numbers;
    (3) coerce mostly-numeric columns to numeric to preserve real features misread as
    text; (4) drop constants; (5) one-hot the remaining low-cardinality categoricals.
    """
    families=[v for v in df[label_col].unique() if v!=normal_value]
    match=[f for f in families if str(f).strip().lower()==str(held_out_family).strip().lower()]
    held_out=match[0] if match else sorted(families,key=lambda f:(df[label_col]==f).sum())[0]
    seen_families=[f for f in families if f!=held_out]

    drop_cols=[c for c in df.columns if c!=label_col and any(p in c.lower() for p in drop_patterns)]
    feat=df.drop(columns=drop_cols+[label_col]).copy()

    # (2) hard identifier cap BEFORE coercion: any column with too many distinct values is an id
    high_distinct=[c for c in feat.columns if feat[c].nunique(dropna=False)>max_distinct
                   and not pd.api.types.is_float_dtype(feat[c])]
    if high_distinct: feat=feat.drop(columns=high_distinct)

    # (3) recover numeric columns misread as text
    coerced=[]
    for c in feat.columns:
        if not pd.api.types.is_numeric_dtype(feat[c]):
            conv=pd.to_numeric(feat[c],errors="coerce"); nn=feat[c].notna().sum()
            if nn>0 and conv.notna().sum()>=numeric_coerce_frac*nn: feat[c]=conv; coerced.append(c)

    const=[c for c in feat.columns if feat[c].nunique(dropna=False)<=1]; feat=feat.drop(columns=const)
    cat=[c for c in feat.columns if not pd.api.types.is_numeric_dtype(feat[c])]
    high_card=[c for c in cat if feat[c].nunique(dropna=False)>max_categories]
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
        "dropped":{"id_leakage":drop_cols,"high_distinct":high_distinct,"constant":const,
                   "high_cardinality":high_card,"encoded":cat,"coerced_numeric":coerced},
        "feature_names":list(feat.columns),
        "X_train":tf(tr),"y_train":y[tr],"fam_train":fam[tr],
        "X_cal":tf(ca),"y_cal":y[ca],
        "X_seen":tf(seen),"y_seen":y[seen],"fam_seen":fam[seen],
        "X_shift":tf(shift),"y_shift":y[shift],"fam_shift":fam[shift],
        "n":{"train":len(tr),"cal":len(ca),"seen":len(seen),"shift":len(shift)}}
