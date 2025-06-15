# analyzer.py

import os
import logging
import math
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Any
from functools import lru_cache
from modules import FORMULA_REGISTRY, compute_global_features
from brain import EXT_GM20_Skip_Pattern_Confidence_Vec, MathUtils, BoardAnalyzerUtils

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def simulate_with_formulas(
    grid_bytes: bytes,
    rows: int,
    cols: int,
    n_iter: int,
    weights: Dict[str, float]=None
) -> Dict[Tuple[int,int], Dict[int,float]]:
    math_utils=MathUtils()
    analyzer=BoardAnalyzerUtils()
    rng=np.random.default_rng()
    grid=np.frombuffer(grid_bytes,dtype=np.int64).reshape(rows,cols)
    batch=10000 if rows*cols<50 else 5000 if rows*cols<200 else 1000
    blanks=np.argwhere(grid==-1)
    known=np.argwhere(grid!=-1)
    known_vals=grid[grid!=-1]
    counter={tuple(b):Counter() for b in map(tuple,blanks)}
    w=weights or {"excel":0.6,"shuffle":0.4}
    names=list(w)
    lin=rows*known[:,0]+known[:,1]
    skip0=EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    mean,std=compute_global_features(grid.astype(np.float32))
    full=n_iter//batch; rem=n_iter%batch

    for idx in range(full+(1 if rem else 0)):
        size=batch if idx<full else rem
        if size==0: continue
        boards=np.zeros((size,rows*cols),dtype=np.int64)
        choice=rng.choice(names,size=size,p=[w[n] for n in names])
        for i,f in enumerate(choice):
            boards[i]=FORMULA_REGISTRY[f](rows,cols,rng).ravel()
        valid=np.all(boards[:,lin]==known_vals,axis=1)
        if not valid.any(): continue
        vbs=boards[valid].reshape(-1,rows,cols)
        mask=np.array([analyzer.check_sequences(b) for b in vbs])
        vbs=vbs[mask]
        if vbs.size:
            bscores=np.array([EXT_GM20_Skip_Pattern_Confidence_Vec(b) for b in vbs])
            cmask=np.array([np.corrcoef(skip0.ravel(),b.ravel())[0,1]>0.8 for b in bscores])
            vbs=vbs[cmask]; bscores=bscores[cmask]
        for bi,b in enumerate(vbs):
            for (r,c) in blanks:
                val=int(b[r,c])
                win=b[max(0,r-1):r+2,max(0,c-1):c+2]
                kn=win[win!=-1]
                res=1.0
                if kn.size: res=1/(1+abs(val-kn.mean()))
                gw=math.exp(-((val-mean)**2)/(2*std**2))
                counter[(r,c)][val]+=bscores[bi,r,c]*res*gw
        if all(cnt and max(cnt.values())/sum(cnt.values())>0.95 for cnt in counter.values()):
            break

    probs={}
    for pos,cnt in counter.items():
        tot=sum(cnt.values()) or 1.0
        probs[pos]={num: MathUtils().normalize_value(v,0,tot) for num,v in cnt.items()}
    return probs

def weight_prob_by_modules(grid: np.ndarray, prob: Dict[Tuple[int,int],Dict[int,float]])->Dict:
    mu=MathUtils(); an=BoardAnalyzerUtils()
    blanks=np.argwhere(grid==-1)
    skip=EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    mean,std=compute_global_features(grid.astype(np.float32))

    for r,c in blanks:
        win=grid[max(0,r-1):r+2,max(0,c-1):c+2]
        kv=win[win!=-1]
        if kv.size:
            for num in list(prob[(r,c)].keys()):
                prob[(r,c)][num]*=1/(1+abs(num-kv.mean()))
    for r,c in blanks:
        for num in list(prob[(r,c)].keys()):
            prob[(r,c)][num]*=skip[r,c]
    for r,c in blanks:
        rowsq=an.get_arithmetic_or_geometric_sequences(grid[r])
        colsq=an.get_arithmetic_or_geometric_sequences(grid[:,c])
        for num in list(prob[(r,c)].keys()):
            if any(num in s for s in rowsq+colsq):
                prob[(r,c)][num]*=1.5
    for r,c in blanks:
        for num in list(prob[(r,c)].keys()):
            prob[(r,c)][num]*=math.exp(-((num-mean)**2)/(2*std**2))

    for pos in prob:
        tot=sum(prob[pos].values()) or 1.0
        prob[pos]={k:mu.normalize_value(v,0,tot) for k,v in prob[pos].items()}
    return prob

def predict_scratch_card(grid: List[List[int]], n_iter: int, formula_only: bool=False)->Dict[str,Any]:
    grid_np=np.array(grid,dtype=np.int64)
    pm=simulate_with_formulas(grid_np.tobytes(),grid_np.shape[0],grid_np.shape[1],n_iter)
    if not formula_only and os.getenv("USE_FORMULA_ONLY")!="1":
        pm=weight_prob_by_modules(grid_np,pm)
    res=[]
    for (r,c),d in pm.items():
        top=sorted(d.items(),key=lambda x:x[1],reverse=True)[:3]
        nums,confs=zip(*top) if top else ((),())
        res.append({"row":int(r),"col":int(c),"candidates":list(nums),"confidences":[round(v,4) for v in confs]})
    res_sorted=sorted(res,key=lambda x:x["confidences"][0] if x["confidences"] else 0,reverse=True)
    return {"predictions":res_sorted,"full_probabilities":{f"{r},{c}":d for (r,c),d in pm.items()}}