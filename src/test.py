import json 

x = json.load(open('all_uncertainties/gsm8k/unsorted_all_uncertainty_records'))['result']
print(x[:2])