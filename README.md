# **CMPT353 Final Project**

## **Table of Contents**
1. [Project Description](#project-description)
2. [Key Objectives](#key-objectives)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Outcomes](#outcomes)

---

## **Project Description**  

This project focuses on analyzing property tax data to uncover insights, predict tax levies, and understand the relationships between various property features like land value, zoning, and neighborhood codes. It uses data preprocessing, statistical analysis, and machine learning techniques to process and model the data effectively.

## **Key Objectives**
1. **Data Analysis**:
   - Understand trends and relationships in property data using visualizations and statistical methods like ANOVA and Chi-Square tests.
   - Identify how factors like land value and zoning impact tax levies.

2. **Machine Learning Models**:
   - Develop predictive models (Linear Regression, Random Forest, and K-Nearest Neighbors) to estimate property tax values.
   - Evaluate model accuracy and interpret results using metrics like RMSE and R².

3. **Visualizations**:
   - Create plots to visualize the actual vs. predicted values, feature importance, and other patterns in the data.

4. **Feature Engineering**:
   - Enhance the dataset by transforming skewed data, scaling features, and encoding categorical variables to improve model performance.

---


## Project Structure
```
Root
│
├── data (Raw data, not included due to its size)
│
├── data_clean (The data-set used for all the statistical tests and analysis)
│
├── data_partitioned (Raw data, partitioned and compressed)
│
├── etl
│    ├── 1_extract.py
│    ├── 2_cleaning.py
│    └── show.py
│
├── models
│    ├── Forest.py
│    ├── KNN.py
│    ├── tax_model.py
│    └── tax_model2.py       
│
├── plots
│    └── plot.py
│
├── stats
│    ├── Anova.py
│    ├── Chi.py
│    ├── tax_chi.py
│    └── tax_cor.py
│
├── README.md
│
└── requirements.txt

```

---

## **Dependencies**

This project uses the following libraries and tools:
- Python3 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Spark

Ensure these are installed before running the project.
   ```bash
    pip install -r requirements.txt
   ```
---
## **Installation**

To set up this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.sfu.ca/nsadaful/CMPT353-Final-Project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd CMPT353-Final-Project
    ```
---

## **Usage**

1. Navigate to the project directory.

2. 1_extract.py - This script performs the initial extraction step of the ETL process. It loads the raw dataset, processes it with Spark, and saves the partitioned and compressed output for subsequent steps.

    ```bash
    python3 etl/1_extract.py
    ```

3. 2_cleaning.py - This script performs the cleaning step of the ETL process.
    ```bash
   python3 etl/2_cleaning.py data_partitioned data_clean 
    ```

4. show.py - This script reads a dataset from the specified input directory and provides an overview by displaying the data and printing the total row count.
   ```bash
   python3 etl/show.py <input_directory>
   ```

5. tax_cor.py - This script analyzes feature importance in predicting property tax (`TAX_LEVY`) using a Random Forest Regressor. It processes both numerical and categorical data, encodes categorical features, and outputs feature importance as a CSV and visualization.
   ```bash
   python3 stats/tax_cor.py data_clean <output_directory>
    ```

6. tax_chi.py - This script performs Chi-Square tests to analyze the relationship between categorical variables in property tax data, such as land value categories, tax categories, and zoning classifications.
   ```bash
   python3 stats/tax_chi.py data_clean
    ```

7. Chi.py - This script performs Chi-Square tests to analyze the relationships between categorical variables such as zoning districts, year built, and price categories of houses. It also categorizes houses based on their total value.
   ```bash
   python3 stats/Chi.py data_clean
    ```

8. Anova.py - This script performs an ANOVA (Analysis of Variance) test to analyze the differences in mean land values (`PREVIOUS_LAND_VALUE`) across hardcoded neighborhood groups. It also performs a Tukey HSD (Honestly Significant Difference) test if the ANOVA test indicates significant differences.
   ```bash
   python3 stats/Anova.py <input_directory>
   ```
   
9. plot.py - This script generates visualizations to analyze the relationships between land values, tax levies, and zoning districts using data provided. It creates scatter and bar plots to provide insights and saves the results in the specified output directory.
   ```bash
   python3 plots/plot.py data_clean plots
   ```

10. tax_model.py - This script builds a simplified linear regression model to predict property tax (`TAX_LEVY`) based on numerical and categorical features. 
   ```bash
   python3 models/tax_model.py data_clean <output_directory>
   ```

11. tax_model2.py - This script builds a polynomial regression model to predict property tax (`TAX_LEVY`) using both numerical and categorical features.
   ```bash
   python3 models/tax_model2.py data_clean <output_directory>
   ```

12. KNN.py - This script trains a K-Nearest Neighbors (KNN) classifier to predict property labels (encoded `CURRENT_LAND_VALUE`) based on numerical and categorical features. It includes feature encoding, scaling, and model evaluation, and outputs predictions and visualizations.
   ```bash
   python3 models/KNN.py data_clean <output_directory>
   ```

13. Forest.py - This script trains a Random Forest Regressor to predict `CURRENT_LAND_VALUE` based on numerical and categorical features. It includes data preprocessing, feature encoding, and scaling. The model's performance is evaluated, and key visualizations, such as actual vs. predicted values, residual distribution, and feature importances, are generated.
   ```bash
   python3 models/Forest.py data_clean <output_directory>
   ```

## **Outcomes**  

- A comprehensive understanding of the factors influencing property tax.
- Predictive models that can estimate property tax values with reasonable accuracy.
- Meaningful visualizations that provide insights into the data and model performance.

---