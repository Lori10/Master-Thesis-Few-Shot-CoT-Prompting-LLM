import numpy as np

def f1_score(distances, uncertainties, args):
    # distances is the precision, uncertainties is the recall, beta is the weight of recall
    distances = np.array(distances)
    uncertainties = np.array(uncertainties)
    #print(f'Distances before Normalization:\n{distances}')
    #print(f'Uncertainties before Normalization:\n{uncertainties}')
    print('---------------------------------------------')
    if args.normalize_distance_uncertainty:
        distances = sum_norm(distances)
        uncertainties = sum_norm(uncertainties)
        #print(f'Distances after Normalization:\n{distances}')
        #print(f'Uncertainties after Normalization:\n{uncertainties}')
        # print(distances[3])
        # print(uncertainties[3])
        # print('------------')

    f1_scores = ((args.beta**2 + 1) * distances * uncertainties) / (args.beta**2 * distances + uncertainties)
    f1_scores[np.isnan(f1_scores)] = 0
    return f1_scores, distances, uncertainties

def sum_norm(scores):
    return scores/ sum(scores)

def square_prob(scores):
    return (scores ** 2)/ sum(scores ** 2)
    
def softmax(scores):
    return np.exp(scores) / np.sum(np.exp(scores), axis=0)

def normalization(scores):
    return (scores - min(scores)) / (max(scores) - min(scores))