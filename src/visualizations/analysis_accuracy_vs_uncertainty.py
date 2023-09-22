import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np 

def plot_uncertainty_vs_accuracy(uncertainty_scores, accuracy_scores, dataset_name):
    sns.scatterplot(x=accuracy_scores, y=uncertainty_scores)
    plt.xlabel('Uncertainty/Dissagreement')
    plt.ylabel('Accuracy')
    plt.savefig(f'plots/uncertainty_vs_accuracy_{dataset_name}.png')
    plt.show()

if __name__ == "__main__":
    
    uncertainty_dissagreement = [1, 3, 5]
    gsm8k_accuracy_scores = [80.7, 81.6, 81.2]
    aqua_accuracy_scores = [56.0, 58.3, 62.3]

    # aqua
    plot_uncertainty_vs_accuracy(aqua_accuracy_scores, uncertainty_dissagreement, 'aqua')

    # gsm8k
    plot_uncertainty_vs_accuracy(gsm8k_accuracy_scores, uncertainty_dissagreement, 'gsm8k')

    # average
    averages = (np.array(gsm8k_accuracy_scores) + np.array(aqua_accuracy_scores)) / 2
    plot_uncertainty_vs_accuracy(averages, uncertainty_dissagreement, 'average')

    