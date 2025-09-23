# Multivariate analysis of spatial features

Documentation for the multivariate analysis.

Proportions, and JSD features computed using the feature engineering approach
documented previously https://github.com/clinicalomx/metabolic-microenvironment-predictors-of-nsclc-immunotherapy-response/tree/main/feature_generation

G-Cross AUC values used from the univariate analysis.

1-ExtractFeaturesFromDataStructure.ipynb
  Take the feature values from the JSD calculations that are stored in an anndata
  array and the G-Cross features and construct a feature dataframe.
2-RunThroughSTABLBinary.ipynb
  Take the feature values and run through Stabl to select informative features that
  model binary relapse.
3-RunThroughSTABLRFS.ipynb
  Take the feature values and run through Stabl to select informative features that
  model the recurrence-free-survival time.
4-MakePlots.ipynb
  Make plots of the selected features.