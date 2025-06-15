# brain.py

import numpy as np
import math
import logging
from collections import Counter
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class MathUtils:
    def sigmoid(self, x: float, k: float = 1.0) -> float:
        try:
            v = max(-700, min(700, -k*x))
            return 1/(1+math.exp(v))
        except OverflowError:
            return 0.0 if -k*x>0 else 1.0

    def normalize_value(self, value: float, min_val: float, max_val: float, clamp: bool=True) -> float:
        if math.isclose(max_val,min_val):
            return 0.5 if math.isclose(value,min_val) else (0.0 if value<min_val else 1.0)
        n = (value-min_val)/(max_val-min_val)
        return max(0.0,min(1.0,n)) if clamp else n

    def manhattan_distance(self, p1: Tuple[int,int], p2: Tuple[int,int]) -> int:
        return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])

class BoardAnalyzerUtils:
    def get_neighborhood_values(self, grid: np.ndarray, r: int, c: int, radius: int=1,
                                eight_connectivity: bool=True,
                                val_func=lambda x: float(x) if x!=-1 else None,
                                include_center: bool=False) -> List[float]:
        neighbors=[]
        rows,cols=grid.shape
        for dr in range(-radius,radius+1):
            for dc in range(-radius,radius+1):
                if not include_center and dr==0 and dc==0: continue
                if not eight_connectivity and abs(dr)+abs(dc)!=1: continue
                nr,nc=r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols:
                    v=val_func(grid[nr,nc])
                    if v is not None: neighbors.append(v)
        return neighbors

    def get_arithmetic_or_geometric_sequences(self, line: np.ndarray, min_len: int=3, allow_gaps: int=1) -> List[List[int]]:
        seqs=[]
        n=len(line)
        for i in range(n):
            if line[i]==-1: continue
            for j in range(i+1,n):
                if line[j]==-1:
                    gap=0
                    for k in range(j,n):
                        if line[k]==-1: gap+=1
                        else:
                            if gap<=allow_gaps:
                                diff=line[k]-line[i]
                                vals=[line[i],line[k]]
                                cg=gap
                                for l in range(k+1,n):
                                    if line[l]==-1:
                                        cg+=1
                                        if cg>allow_gaps: break
                                        continue
                                    exp=vals[-1]+diff
                                    if math.isclose(line[l],exp):
                                        vals.append(line[l]); cg=0
                                    else: break
                                if len(vals)>=min_len: seqs.append(vals)
                            break
                else:
                    diff=line[j]-line[i]
                    if diff==0 and line[i]!=0: continue
                    vals=[line[i],line[j]]
                    for k in range(j+1,n):
                        if line[k]==-1: continue
                        exp=vals[-1]+diff
                        if math.isclose(line[k],exp): vals.append(line[k])
                        else: break
                    if len(vals)>=min_len: seqs.append(vals)
        return seqs

    def get_card_max_value_from_gridDimensions(self, shape: Tuple[int,int]) -> int:
        r,c=shape; return r*c if r>0 and c>0 else 0

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        r,c=grid.shape
        maxv=self.get_card_max_value_from_gridDimensions((r,c))
        allv=set(range(1,maxv+1))
        used={int(x) for x in grid.flatten() if x!=-1 and x>0}
        return allv-used

    def check_sequences(self, board: np.ndarray, min_len: int=3, allow_gaps: int=1) -> bool:
        rows,cols=board.shape
        for r in range(rows):
            if self.get_arithmetic_or_geometric_sequences(board[r],min_len,allow_gaps): return True
        for c in range(cols):
            if self.get_arithmetic_or_geometric_sequences(board[:,c],min_len,allow_gaps): return True
        return False

def EXT_GM20_Skip_Pattern_Confidence_Vec(grid: np.ndarray, request_id: Optional[str]="N/A") -> np.ndarray:
    rows,cols=grid.shape
    scores=np.zeros((rows,cols),dtype=float)
    revealed=[{"value":int(grid[r,c]),"r":r,"c":c} for r in range(rows) for c in range(cols) if grid[r,c]!=-1 and grid[r,c]>0]
    if not revealed: return scores
    base={(v):((v-1)//cols,(v-1)%cols) for v in range(1,rows*cols+1)}
    skipv={}
    for info in revealed:
        v,r,c=info["value"],info["r"],info["c"]
        er,ec=base[v]; skipv[v]=(r-er,c-ec)
    if not skipv: return scores
    from collections import Counter
    cnt=Counter(skipv.values())
    mocc=max(1,int(len(skipv)*0.05))
    pats=[]
    for vec,ct in cnt.most_common():
        if ct>=mocc:
            vals=sorted([v for v,sv in skipv.items() if sv==vec])
            streng= MathUtils().normalize_value(float(ct),float(mocc),float(len(skipv)))
            pats.append({"skip":vec,"values":vals,"strength":streng})
        else: break
    if not pats: return scores
    legal=BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r,c]!=-1: continue
            best=0.0
            for num in legal:
                if num not in base: continue
                br,bc=base[num]
                for pat in pats:
                    dr,dc=pat["skip"]
                    if (br+dr,bc+dc)==(r,c):
                        fac=0.5; vals=pat["values"]
                        if vals:
                            seq=sorted(vals+[num])
                            difs=np.diff(seq)
                            if len(set(difs))==1 and difs[0]!=0: fac+=0.4
                            elif len(seq)>=3 and min(vals)<num<max(vals): fac+=0.1
                        score=pat["strength"]*fac
                        best=max(best,score)
            from brain import MathUtils
            scores[r,c]=MathUtils().normalize_value(best,0,1.0)
    return scores