import logging
from typing import List, Optional, Dict, Tuple, Any
from abc import ABC, abstractmethod
import time
import os
import json

import numpy as np  # For vectorized operations
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 0. Configuration & File Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_CONTRIBUTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "module_contribution_weights.json")
LOCAL_HISTORICAL_MEMORY_PATH = os.path.join(BASE_DIR, "local_historical_memory.json")

# --- Global Configuration for Weights ---
DEFAULT_MODULE_CONTRIBUTION_WEIGHTS: Dict[str, float] = {
    "M1_BaseScore": 1.0,
    "M2_VectorAP": 1.0,
    "M3_VectorGP": 0.7,
    "M4_SymmetryAxial": 0.6,
    "M5_SegmentDiff": 0.5,
    "M6_Historical": 1.2,
    "M10_BridgeChain": 0.7,
    "M11_GridLinearGrowth": 0.7,
    "M14_SymmetryFill": 0.6,
    "M17_CenterCompletion": 0.7,
    "M21_EndSegmentMean": 0.6,
    "M6_MemorySimilarity": 1.0,
}
CURRENT_MODULE_CONTRIBUTION_WEIGHTS: Dict[str, float] = {}
DEFAULT_LOGIC_CODE_WEIGHT_IF_MISSING = 0.1
DEFAULT_MODULE_CONTRIBUTION_WEIGHT_IF_MISSING = 0.5

def load_module_contribution_weights() -> None:
    global CURRENT_MODULE_CONTRIBUTION_WEIGHTS
    if os.path.exists(MODULE_CONTRIBUTION_WEIGHTS_PATH):
        try:
            with open(MODULE_CONTRIBUTION_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            CURRENT_MODULE_CONTRIBUTION_WEIGHTS = {**DEFAULT_MODULE_CONTRIBUTION_WEIGHTS, **loaded}
            logger.info("Loaded module weights from file.")
        except Exception:
            logger.exception("Failed loading module weights, using defaults.")
            CURRENT_MODULE_CONTRIBUTION_WEIGHTS = DEFAULT_MODULE_CONTRIBUTION_WEIGHTS.copy()
    else:
        logger.info("Module weights file not found; creating with defaults.")
        CURRENT_MODULE_CONTRIBUTION_WEIGHTS = DEFAULT_MODULE_CONTRIBUTION_WEIGHTS.copy()
        save_module_contribution_weights()

def save_module_contribution_weights() -> None:
    try:
        with open(MODULE_CONTRIBUTION_WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(CURRENT_MODULE_CONTRIBUTION_WEIGHTS, f, indent=2, sort_keys=True)
        logger.info("Saved module weights to file.")
    except Exception:
        logger.exception("Failed saving module weights.")

load_module_contribution_weights()

# -----------------------------------------------------------------------------
# 1. Pydantic Models
# -----------------------------------------------------------------------------
class BoardInput(BaseModel):
    new_card: List[Optional[int]] = Field(..., description="1D list row-major; None=masked")
    proposed_values: List[int]
    cols: int = Field(..., gt=0)
    position_codes: Optional[List[str]] = None
    logic_code_weights: Optional[Dict[str, float]] = Field(default_factory=dict)
    active_modules: Optional[List[str]] = None
    module_weights: Optional[Dict[str, float]] = None
    top_n_count: int = Field(default=3, gt=0)
    historical_api_endpoint: Optional[str] = None

    @validator('position_codes')
    def check_pos_codes(cls, v, values):
        if v is not None and len(v) != len(values.get('new_card', [])):
            raise ValueError("position_codes length must match new_card")
        return v

    @validator('cols')
    def check_cols_divides(cls, v, values):
        grid = values.get('new_card', [])
        if grid and len(grid) % v != 0:
            raise ValueError("new_card length must be divisible by cols")
        return v

class PositionScore(BaseModel):
    position_code: str
    score: float

class ValuePrediction(BaseModel):
    proposed_value: int
    top_n_positions: List[PositionScore]

class InferenceResponse(BaseModel):
    predictions: List[ValuePrediction]
    processing_time_ms: Optional[float]
    warnings: Optional[List[str]]

class ModuleInfo(BaseModel):
    module_id: str
    name: str
    description: str

# -----------------------------------------------------------------------------
# 2. Internal Board Structures
# -----------------------------------------------------------------------------
class InternalBoardCell:
    def __init__(self, row:int, col:int, logic_code:str, value:Optional[int], is_fixed:bool, base_score:float):
        self.row, self.col = row, col
        self.logic_code = logic_code
        self.value = value
        self.is_fixed = is_fixed
        self.base_score = base_score

class InternalBoardState:
    def __init__(self, inp: BoardInput):
        self.source_input = inp
        self.cols = inp.cols
        self.grid_1d = inp.new_card
        self.rows = len(self.grid_1d) // self.cols if self.cols else 0
        self.logic_codes_1d = inp.position_codes or self._generate_default_codes()
        self.board: List[List[InternalBoardCell]] = []
        self._build_board()

    def _generate_default_codes(self) -> List[str]:
        def col_letter(n):
            s=""
            while n>=0:
                s=chr(ord('A')+n%26)+s
                n=n//26-1
            return s
        L=[]
        for r in range(self.rows):
            for c in range(self.cols):
                L.append(f"{col_letter(c)}{r+1}")
        return L

    def _build_board(self):
        idx=0
        for r in range(self.rows):
            row=[]
            for c in range(self.cols):
                val = self.grid_1d[idx]
                code = self.logic_codes_1d[idx]
                base = self.source_input.logic_code_weights.get(code, DEFAULT_LOGIC_CODE_WEIGHT_IF_MISSING)
                cell = InternalBoardCell(r, c, code, val, val is not None, base)
                row.append(cell)
                idx+=1
            self.board.append(row)

    def as_numpy(self) -> np.ndarray:
        """Return a 2D float array with np.nan for masked."""
        arr = np.full((self.rows, self.cols), np.nan, dtype=float)
        for r in range(self.rows):
            for c in range(self.cols):
                v = self.board[r][c].value
                arr[r, c] = float(v) if v is not None else np.nan
        return arr

    def get_cell(self, r:int, c:int) -> InternalBoardCell:
        return self.board[r][c]

    def get_board_id(self) -> str:
        tup = tuple(-999 if v is None else v for v in self.grid_1d)
        h = hash((self.rows,self.cols,tup))
        empties = self.grid_1d.count(None)
        return f"{self.rows}x{self.cols}_empty{empties}_hash{h}"

# -----------------------------------------------------------------------------
# 3. Logic Module Framework
# -----------------------------------------------------------------------------
class LogicModule(ABC):
    def __init__(self, module_id:str, name:str, description:str):
        self.module_id, self.name, self.description = module_id, name, description

    @abstractmethod
    def analyze(self, board_state:InternalBoardState, cell:Tuple[int,int], pv:int) -> float: ...

    def get_info(self) -> ModuleInfo:
        return ModuleInfo(module_id=self.module_id, name=self.name, description=self.description)

class M1_BaseScoreModule(LogicModule):
    def __init__(self):
        super().__init__("M1_BaseScore", "Base Score", "Uses logic_code_weights")
    def analyze(self, bs, cell, pv):
        c = bs.get_cell(*cell)
        return c.base_score

class M2_VectorAPModule(LogicModule):
    # unchanged...
    ...

class M3_VectorGPModule(LogicModule):
    # unchanged...
    ...

class M4_SymmetryAxialModule(LogicModule):
    def __init__(self):
        super().__init__("M4_SymmetryAxial", "Axial Symmetry", "Checks symmetry across axes")
    def analyze(self, bs:InternalBoardState, cell:Tuple[int,int], pv:int) -> float:
        arr = bs.as_numpy()
        r, c = cell
        # propose value
        arr[r, c] = pv
        scores = []
        # horizontal symmetry
        mask = ~np.isnan(arr)
        horiz = (arr == np.fliplr(arr))
        scores.append(np.nanmean(horiz[mask & np.fliplr(mask)]))
        # vertical
        vert = (arr == np.flipud(arr))
        scores.append(np.nanmean(vert[mask & np.flipud(mask)]))
        # main diag
        diag_flip = arr.T
        mask_diag = mask.T
        diag = (arr == diag_flip)
        scores.append(np.nanmean(diag[mask & mask_diag]))
        # final
        return float(np.nanmax(scores))

class M5_SegmentDiffModule(LogicModule):
    def __init__(self):
        super().__init__("M5_SegmentDiff", "Segment Diff", "Variance of differences in segments")
    def analyze(self, bs:InternalBoardState, cell:Tuple[int,int], pv:int) -> float:
        arr = bs.as_numpy()
        r, c = cell
        arr[r, c] = pv
        best = 0.0
        # directions: horizontal, vertical
        for dr, dc in [(0,1),(1,0)]:
            # collect along line
            line = []
            # go backward
            i = -1
            while True:
                rr, cc = r+dr*i, c+dc*i
                if 0<=rr<bs.rows and 0<=cc<bs.cols and not np.isnan(arr[rr,cc]):
                    line.insert(0, arr[rr,cc]); i-=1
                else: break
            # include cell
            line.append(arr[r,c])
            # go forward
            i=1
            while True:
                rr, cc = r+dr*i, c+dc*i
                if 0<=rr<bs.rows and 0<=cc<bs.cols and not np.isnan(arr[rr,cc]):
                    line.append(arr[rr,cc]); i+=1
                else: break
            if len(line)>=3:
                diffs = np.diff(line)
                var = np.nanvar(diffs)
                score = 1.0/(1.0+var)
                best = max(best, score)
        return float(best)

class M6_HistoricalModule(LogicModule):
    # unchanged...
    ...

# ... other placeholder modules ...

# -----------------------------------------------------------------------------
# 4. Module Registry
# -----------------------------------------------------------------------------
class ModuleRegistry:
    def __init__(self):
        self._mods: Dict[str, LogicModule] = {}
        self._register_defaults()

    def _register_defaults(self):
        for mod in [
            M1_BaseScoreModule(),
            M2_VectorAPModule(),
            M3_VectorGPModule(),
            M4_SymmetryAxialModule(),
            M5_SegmentDiffModule(),
            M6_HistoricalModule(),
            # ... etc ...
        ]:
            self._mods[mod.module_id] = mod

    def get_module(self, mid:str) -> Optional[LogicModule]:
        return self._mods.get(mid)

    def get_all_modules(self) -> List[LogicModule]:
        return list(self._mods.values())

    def get_module_infos(self) -> List[ModuleInfo]:
        return [m.get_info() for m in self._mods.values()]

module_registry = ModuleRegistry()

# -----------------------------------------------------------------------------
# 5. Inference Engine
# -----------------------------------------------------------------------------
class InferenceEngine:
    def __init__(self, registry:ModuleRegistry=Depends(lambda: module_registry)):
        self.registry = registry

    def run_inference(self, bs:InternalBoardState) -> Tuple[List[ValuePrediction],List[str]]:
        preds, warns = [], []
        active = bs.source_input.active_modules or []
        mods = (self.registry.get_all_modules() if not active 
                else [self.registry.get_module(m) for m in active if self.registry.get_module(m)])
        weights = {**CURRENT_MODULE_CONTRIBUTION_WEIGHTS,
                   **(bs.source_input.module_weights or {})}

        for pv in bs.source_input.proposed_values:
            cands = []
            # skip if already present
            if any(cell.value==pv for row in bs.board for cell in row if cell.is_fixed):
                warns.append(f"{pv} already on board")
                preds.append(ValuePrediction(proposed_value=pv, top_n_positions=[]))
                continue

            for r in range(bs.rows):
                for c in range(bs.cols):
                    cell = bs.get_cell(r,c)
                    if cell.is_fixed: continue
                    num, den = 0.0, 0.0
                    # M1
                    m1 = self.registry.get_module("M1_BaseScore")
                    raw1 = m1.analyze(bs,(r,c),pv); w1 = weights.get(m1.module_id, DEFAULT_MODULE_CONTRIBUTION_WEIGHT_IF_MISSING)
                    num+=raw1*w1; den+=w1
                    # others
                    for m in mods:
                        if m.module_id=="M1_BaseScore": continue
                        raw = m.analyze(bs,(r,c),pv)
                        w = weights.get(m.module_id, DEFAULT_MODULE_CONTRIBUTION_WEIGHT_IF_MISSING)
                        num+=raw*w; den+=w
                    score = num/den if den>0 else 0.0
                    cands.append((cell.logic_code, round(score,4)))
            cands.sort(key=lambda x:x[1], reverse=True)
            top = [PositionScore(position_code=lc, score=sc) for lc,sc in cands[:bs.source_input.top_n_count]]
            preds.append(ValuePrediction(proposed_value=pv, top_n_positions=top))
        return preds, warns

# -----------------------------------------------------------------------------
# 6. FastAPI Setup
# -----------------------------------------------------------------------------
app = FastAPI(title="Adaptive Fill API", version="1.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze", response_model=InferenceResponse)
async def analyze(board_input:BoardInput=Body(...),
                  engine:InferenceEngine=Depends()):
    start = time.perf_counter()
    try:
        bs = InternalBoardState(board_input)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    preds, warns = engine.run_inference(bs)
    elapsed = round((time.perf_counter()-start)*1000,2)
    return InferenceResponse(predictions=preds,
                             processing_time_ms=elapsed,
                             warnings=warns or None)

@app.get("/config/logic_modules", response_model=List[ModuleInfo])
async def list_modules():
    return module_registry.get_module_infos()

@app.get("/config/module_weights", response_model=Dict[str,float])
async def get_weights():
    return CURRENT_MODULE_CONTRIBUTION_WEIGHTS

@app.on_event("shutdown")
async def shutdown():
    save_module_contribution_weights()