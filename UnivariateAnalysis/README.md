# Cell typing and univariate analysis of spatial features

Documentation for the univariate analysis.


## Cell typing
CT.ipynb, GMM_classification.ipynb and GMM_metabolic_classification.ipynb 

## Feature generation 
Feature generation can be run after cell typing is complete, using the anndata (.h5ad). 

G-Cross AUC generated with Kaplan-Meier correction using gcross.py
Diversity and entropy calculations using Metabolic_diversity.ipynb

## Statistical analysis 
Statistical analysis was performed in R. Data should be in .csv format that contains both feature scores and patient clinical variables. Each sample should contain 1 score/feature. 

Mixed effects statistical modelling using: Mixedeffects_statistics.Rdm


