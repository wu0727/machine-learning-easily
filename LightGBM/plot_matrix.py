import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def plot_confusion_matrix(actual_val, pred_val, title):
    confusion_matrix = pd.crosstab(actual_val, pred_val,
                                   rownames = ['Actual'],
                                   colnames = ['Predicted'])
    
    plot = sns.heatmap(confusion_matrix, annot = True, fmt = ',.0f')
    plot.set_title(title)
    plt.show()
