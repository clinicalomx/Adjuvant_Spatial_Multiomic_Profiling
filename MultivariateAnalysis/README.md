# Multivariate analysis of spatial features

Documentation for the multivariate analysis in Figure 6.

Proportions, and JSD features computed using the feature engineering approach
documented previously https://github.com/clinicalomx/metabolic-microenvironment-predictors-of-nsclc-immunotherapy-response/tree/main/feature_generation

MultivariableCrossValidation_RecurrenceStatus.ipynb computes spatial features and feature selection for Figure 6e,f,g.
MultivariableCrossValidation_RecurrenceStatus.ipynb computes spatial features and feature selection in cross-validation for Figure 6c.



## Run-Ready Environment

`conda create -n analysis scikit-survival scikit-learn matplotlib pandas anndata`


`pip install git+https://github.com/gregbellan/Stabl.git@v1.0.1-lw`


`git clone https://github.com/clinicalomx/spatial_analysis.git`

`cd spatial_analysis`

`pip install .`