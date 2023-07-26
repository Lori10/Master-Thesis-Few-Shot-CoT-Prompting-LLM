import numpy as np

li = np.array([np.nan, 3, 4, np.nan, np.nan])
li[np.isnan(li)] = 10
print(li)