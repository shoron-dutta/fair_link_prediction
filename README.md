# fair_link_prediction

To run link prediction experiment on a graph dataset:
run fair_link_pred.py
Arguments:
dataset: dataset name; default: "cora"; options = "cora", "citeseer", "pubmed"
num_epochs: number of epochs; default = 101
reg_lambda: hyperparameter for controling the mutual information regularization; default=0.7
alpha: parameter for Renyi's alpha order mutual information; default=0.5

This will run 6 instances of the code (6 different seeds) and report the mean and standard deviation for the metrics.

To run ablation study for different values of lambda:
run ablation_plot.py
Arguments:
dataset: dataset name; default: "cora"; options = "cora", "citeseer", "pubmed"
num_epochs: number of epochs; default = 101
alpha: parameter for Renyi's alpha order mutual information; default=0.5

Here, lambda will take values from this set: {0.2, 0.4, 0.6, 0.8, 1.0}
The code will produce a plot and store it in './figures/' directory.


entropy.py: contains code for computes Renyi's alpha order entropy and mutual information
utils.py: basic functionality code (mostly from FairDrop)
