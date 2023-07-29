import numpy as np
import random

random.seed(42)
data = [4, 3, 10, 12, 20, 4]
for i in range(2):
    selected_data = random.sample(data, 2)
    print(selected_data)
    print('*************************')