import numpy as np
import pandas as pd

s = pd.Series([1, 2, 3, 4])
s.index = [10, 20, 30, 40]

print(s.idxmax())