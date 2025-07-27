# MLB Debut Prediction: Machine Learning Analysis

> **Published Research**: This work has been published as a chapter in the Springer book series: [*Baseball Informatics—From MiLB to MLB Debut*](https://link.springer.com/chapter/10.1007/978-981-19-9658-0_5)

## Overview

A comprehensive machine learning analysis that predicts MLB debut success for drafted players using XGBoost models. This research implements multiple feature selection strategies and compares model performance to identify optimal predictive approaches for baseball talent evaluation.

The study demonstrates the application of advanced machine learning techniques to sports analytics, providing a systematic framework for talent assessment that has been validated through peer-reviewed publication.

## Key Features

- **Four distinct XGBoost models** with different feature selection strategies
- **Comprehensive exploratory data analysis** with 15+ visualization approaches
- **Automated hyperparameter optimization** using Latin Hypercube Sampling
- **Rigorous evaluation framework** with 5-fold cross-validation
- **Professional-grade code architecture** with modular design and reproducible workflows

## Repository Structure

```
├── From-MiLB-to-MLB-Debut-a-Machine-Learning-Application.Rmd  # Complete analysis notebook
├── xgboost-mlb-prediction.r                                   # Optimized XGBoost implementation
├── README.md                                                   # This file
└── data/
    └── mlb_draft_01to10.csv                                   # Dataset (not included)
```

## Methodology

### Data Science Approach

The analysis employs a systematic machine learning pipeline incorporating data preprocessing, feature engineering, hyperparameter optimization, and model validation. The research focuses specifically on XGBoost implementation due to its superior performance in structured data prediction tasks.

### Model Implementations

**Model 1: Full Feature Set**
Utilizes all available variables after dummy encoding categorical features, establishing performance benchmarks using comprehensive player information including demographic, performance, and draft-related variables.

**Model 2: LASSO-Selected Features (All Data)**
Implements feature selection using LASSO regularization to identify the most predictive variables from the complete dataset, reducing dimensionality while maintaining predictive power through automated variable selection.

**Model 3: Expert-Selected Features**
Incorporates domain expertise to select variables based on baseball analytics principles, including batting statistics (OPS, ISO, BB/SO ratio), physical characteristics (BMI, age), and draft position metrics that align with established talent evaluation frameworks.

**Model 4: LASSO-Selected Features (Latin Hypercube)**
Applies LASSO regularization specifically tuned for the Latin Hypercube sampling approach, optimizing feature selection for the hyperparameter search methodology employed in model training.

## Technical Specifications

- **Programming Language**: R
- **Machine Learning Framework**: tidymodels
- **Primary Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Validation Method**: Stratified 5-fold Cross-Validation
- **Optimization Approach**: Latin Hypercube Sampling
- **Performance Metric**: Area Under ROC Curve (AUC-ROC)

## Requirements

### Dependencies

```r
library(tidyverse)     # Data manipulation and visualization
library(tidymodels)    # Machine learning framework
library(xgboost)       # Gradient boosting implementation
library(vip)           # Variable importance visualization
library(doParallel)    # Parallel processing
library(fastDummies)   # Categorical variable encoding
library(glmnet)        # LASSO regularization
library(ggthemes)      # Additional ggplot themes
```

### System Requirements

- R version 4.0.0 or higher
- Minimum 8GB RAM recommended for model training
- Multi-core processor recommended for parallel processing

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/yourusername/mlb-debut-prediction.git
cd mlb-debut-prediction
```

### Install Required Packages

```r
# Install required packages if not already installed
packages <- c("tidyverse", "tidymodels", "xgboost", "vip", "doParallel", 
              "fastDummies", "glmnet", "ggthemes", "BBmisc")

install.packages(packages[!packages %in% installed.packages()[,"Package"]])
```

## Usage

### Quick Start with Optimized XGBoost Models

For users who want to focus specifically on the XGBoost implementation, use the optimized script:

```r
# Load the optimized XGBoost functions
source("xgboost-mlb-prediction.r")

# Load your MLB draft data
df_mlb <- read_csv("data/mlb_draft_01to10.csv")

# The script will automatically:
# 1. Prepare data with proper encoding
# 2. Create train-test splits with stratification
# 3. Train all four XGBoost models
# 4. Generate performance comparisons
# 5. Create visualizations and variable importance plots
```

### Complete Analysis Workflow

For the full exploratory data analysis and comprehensive modeling approach:

```r
# Open and run the complete R Markdown file
# This includes:
# - Extensive exploratory data analysis
# - Multiple visualization approaches
# - Feature selection techniques
# - Model comparisons across algorithms
```

### Data Requirements

The analysis expects a CSV file with the following structure:

- **Player demographics**: name, age_at_draft, height, weight, birth_place
- **Draft information**: draft_year, draft_round, draft_overall, team
- **Performance statistics**: avg, obp, slg, hr, bb, so, sb, cs, etc.
- **Target variable**: mlb_debut (binary: yes/no)

Users can adapt the code for similar baseball datasets by ensuring consistent column naming and data types.

## Model Performance

The analysis provides quantitative comparison of all modeling approaches through AUC-ROC evaluation. Performance differences highlight the effectiveness of various feature selection strategies in baseball talent prediction contexts.

Key findings from the research demonstrate that expert-selected features combined with XGBoost optimization provide robust predictive performance while maintaining model interpretability for practical talent evaluation applications.

## Visualizations and Analysis

The complete analysis includes comprehensive visualizations covering:

- Distribution analysis across teams, positions, physical characteristics, and performance metrics
- Temporal trends in player statistics and debut rates
- Correlation analysis between key performance indicators
- Model performance comparisons with ROC curves
- Variable importance rankings across different model configurations

## Publication and Citation

This research has been published in the Springer book series, validating the methodological rigor and practical applicability of the machine learning techniques implemented in this repository.

**Citation:**
```bibtex
@inbook{lee_mlb_debut_2023,
  title={Baseball Informatics—From MiLB to MLB Debut},
  author={Lee, Chung-Hao},
  publisher={Springer},
  year={2023},
  url={https://link.springer.com/chapter/10.1007/978-981-19-9658-0_5},
  doi={10.1007/978-981-19-9658-0_5}
}
```

## Contributing

This repository represents completed academic research. For questions about methodology or potential extensions, please refer to the published chapter or contact the author.

## License

 CC0-1.0 license

## Contact

**Author**: Chung-Hao Lee  
**Email**: lch99310@gmail.com  
**Published Work**: [Springer Chapter - Baseball Informatics](https://link.springer.com/chapter/10.1007/978-981-19-9658-0_5)

---

*This research demonstrates professional-grade data science practices including proper validation procedures, comprehensive evaluation metrics, and reproducible research protocols, ensuring reliable and actionable insights for baseball talent assessment.*
