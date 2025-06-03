# new_module3.py
"""
整合所有推理模組的模組庫。匯入 brain1、brain2、brain3 中定義的所有 `_score` 函數。
"""
import numpy as np
from brain1 import *
from brain2 import *
from brain3 import *

# 所有模組已透過匯入整合至此命名空間，供分析器動態調用。