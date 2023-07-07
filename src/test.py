import numpy as np
import json 

li = [2, 4, 5, 4]
print(np.mean(np.array([[10]])))
li_dic = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4},
          {'mean_acc' : 0.5}]

with open('test.txt', 'w') as f:
    f.write(json.dumps(li_dic, indent=4))