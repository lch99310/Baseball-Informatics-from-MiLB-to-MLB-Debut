# MLB Debut Prediction: Machine Learning Analysis

## Overview

This repository contains a comprehensive machine learning analysis predicting MLB debut success for drafted players using XGBoost models. The research implements multiple feature selection strategies and compares model performance to identify optimal predictive approaches for baseball talent evaluation.

## Research Methodology

### Objective
Develop predictive models to assess the likelihood of drafted baseball players making their MLB debut, providing actionable insights for talent evaluation and draft strategy optimization.

### Data Science Approach
The analysis employs a systematic machine learning pipeline incorporating data preprocessing, feature engineering, hyperparameter optimization, and model validation. Four distinct XGBoost models are implemented with different feature selection strategies to identify the most effective predictive framework.

### Model Architecture
- **Base Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Validation Strategy**: 5-fold cross-validation with stratified sampling
- **Hyperparameter Optimization**: Latin Hypercube Sampling
- **Performance Metrics**: AUC-ROC, Accuracy, Precision, Recall, F1-Score

## Model Implementations

### Model 1: Full Feature Set
Utilizes all available variables after dummy encoding categorical features. This baseline model establishes performance benchmarks using comprehensive player information including demographic, performance, and draft-related variables.

### Model 2: LASSO-Selected Features (All Data)
Implements feature selection using LASSO regularization to identify the most predictive variables from the complete dataset. This approach reduces dimensionality while maintaining predictive power through automated variable selection.

### Model 3: Expert-Selected Features
Incorporates domain expertise to select variables based on baseball analytics principles. Features include batting statistics (OPS, ISO, BB/SO ratio), physical characteristics (BMI, age), and draft position metrics that align with established talent evaluation frameworks.

### Model 4: LASSO-Selected Features (Latin Hypercube)
Applies LASSO regularization specifically tuned for the Latin Hypercube sampling approach, optimizing feature selection for the hyperparameter search methodology employed in model training.

## Technical Implementation

### Dependencies
```r
library(tidyverse)     # Data manipulation and visualization
library(tidymodels)    # Machine learning framework
library(xgboost)       # Gradient boosting implementation
library(vip)           # Variable importance visualization
library(doParallel)    # Parallel processing
```

### Data Preprocessing Pipeline
The preprocessing workflow includes categorical variable encoding, missing value handling, and feature scaling. Dummy variables are created for team affiliations, batting preferences, throwing preferences, and geographic indicators to ensure optimal model performance.

### Hyperparameter Optimization
Model tuning employs Latin Hypercube Sampling across six key parameters:
- Tree depth (model complexity)
- Minimum node size (overfitting control)
- Loss reduction threshold (pruning criterion)
- Sample size proportion (stochastic training)
- Feature sampling ratio (random forest component)
- Learning rate (gradient descent step size)

### Performance Evaluation Framework
Model assessment utilizes multiple metrics to ensure robust evaluation. AUC-ROC serves as the primary optimization criterion, supplemented by confusion matrix analysis and variable importance assessment. Cross-validation ensures generalizability and prevents overfitting.

## Key Features

### Automated Model Tuning
The implementation includes comprehensive functions for automated hyperparameter optimization with parallel processing capabilities. This ensures efficient computational resource utilization while maintaining reproducible results through seed management.

### Comprehensive Evaluation Suite
Each model generates detailed performance reports including confusion matrices, ROC curves, variable importance plots, and probability distribution analyses. These outputs facilitate thorough model interpretation and validation.

### Modular Architecture
The codebase employs a modular design with dedicated functions for data preparation, model tuning, and evaluation. This structure enhances code maintainability and enables easy extension for additional modeling approaches.

## Results and Insights

### Model Performance Comparison
The analysis provides quantitative comparison of all four modeling approaches through AUC-ROC evaluation. Performance differences highlight the effectiveness of various feature selection strategies in baseball talent prediction contexts.

### Variable Importance Analysis
Variable importance plots identify the most predictive factors for MLB debut success, providing actionable insights for scouts and analysts. These findings contribute to evidence-based talent evaluation methodologies.

### Predictive Visualizations
The implementation generates probability distribution plots and ROC curves that illustrate model performance characteristics and decision thresholds. These visualizations support both technical validation and stakeholder communication.

## Reproduction and Validation

### Setup Requirements
Ensure R environment includes all required packages listed in the dependencies section. The analysis utilizes tidymodels framework conventions for consistent and reproducible machine learning workflows.

### Execution Instructions
1. Load the complete script to initialize all functions and model specifications
2. Execute data preparation functions to create analysis-ready datasets
3. Run individual model sections to generate trained models and performance metrics
4. Compare results using the integrated model comparison framework

### Verification Protocol
All models implement consistent cross-validation procedures with fixed random seeds to ensure reproducible results. Performance metrics can be independently verified through the standardized evaluation functions provided in the codebase.

## Technical Specifications

- **Programming Language**: R
- **Machine Learning Framework**: tidymodels
- **Primary Algorithm**: XGBoost
- **Validation Method**: Stratified K-fold Cross-Validation
- **Optimization Approach**: Grid Search with Latin Hypercube Sampling
- **Performance Metric**: Area Under ROC Curve (AUC-ROC)

## Research Contributions

This analysis demonstrates the application of advanced machine learning techniques to baseball analytics, providing a systematic framework for talent evaluation. The comparative approach across multiple feature selection strategies offers insights into optimal modeling practices for sports analytics applications.

The implementation showcases professional-grade data science practices including proper validation procedures, comprehensive evaluation metrics, and reproducible research protocols. These methodologies ensure reliable and actionable insights for baseball talent assessment.

## Future Enhancements

Potential extensions include ensemble modeling approaches, temporal analysis of player development trajectories, and integration of advanced baseball metrics. The modular architecture facilitates incorporation of additional algorithms and evaluation frameworks.

---

**License**: MIT
**Contact**: lch99310@gmail.com 
**Last Updated**: July 27, 2025
