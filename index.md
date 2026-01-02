---
title: "Machine Learning Prediction of Exercise Quality: An Analysis Using XGBoost and Wearable Sensors"
subtitle: "Practical Machine Learning Course Project | Johns Hopkins University"
author: "Hayelom D. Tesfay"
date: "January 01, 2026"
output:
  html_document:
    toc: true
    toc_float:
      collapsed: false       # TOC shows all sections initially
      smooth_scroll: true    # Animation when clicking TOC links
    toc_depth: 3             # Number of header levels in TOC
    number_sections: false   # Automatically number all headers
    theme: lumen             # Choose a Bootstrap theme (e.g., cerulean, flatly, cosmo)
    highlight: tango         # Syntax highlighting style for code
    code_folding: show       # Add "Show/Hide Code" buttons (options: show, hide, none)
    code_download: true      # Adds a button to download the source .Rmd file
    df_print: paged          # Displays data frames as interactive, paged tables
    fig_width: 8             # Default figure width in inches
    fig_height: 6            # Default figure height in inches
    keep_md: true            # Saves a .md file during rendering
---



















































































# Executive Summary

This project aimed to develop a highly accurate machine learning model to classify human movements (classe) using data sourced from wearable sensors. After extensive data preprocessing, several algorithms (XGBoost, Random Forest, SVM, k-NN) were trained and compared using a 5-fold cross-validation scheme. The *XGBoost (Gradient Boosting) model* was selected as the optimal choice due to its superior performance across key metrics, particularly its low LogLoss and fewer total misclassifications. The final model achieved an estimated out-of-sample accuracy of approximately 99.93%. The report confirms that the model is highly stable, reliable, and driven by key sensor readings and an engineered temporal feature (num_window).

# 1. Introduction

## 1.1 Background                       

Physical activity monitoring is a growing field with applications in healthcare, fitness tracking, and rehabilitation. Wearable sensors provide a rich, continuous stream of data detailing human movement, but the challenge lies in accurately translating this raw data into meaningful classifications of specific activities. 

This project addresses this challenge through qualitative activity recognition, shifting the focus of wearable technology from tracking activity volume to evaluating the quality of movement execution. By applying machine learning to accelerometer data from the Weight Lifting Exercises Dataset, the study classifies bicep curl techniques into correct and incorrect forms. This research ultimately demonstrates how predictive modeling can provide automated, real-time feedback in sports and rehabilitation to enhance performance and reduce the risk of injury.

## 1.2 Objective 

The objective of this project is to develop a robust predictive model that objectively assesses the sensor inputs and classifies the quality of exercise performance using data from accelerometers on the belt, forearm, arm, and dumbbell. The model is designed to predict the "classe" variable, which features five distinct levels, each corresponding to a different execution manner: 

- Class A: Exactly according to the specification (correct execution)
- Class B: Throwing the elbows to the front (incorrect)
- Class C: Lifting the dumbbell only halfway (incorrect)
- Class D: Lowering the dumbbell only halfway (incorrect)
- Class E: Throwing the hips to the front (incorrect)

The ultimate goal is to deploy an objective feedback system for real-world use, assisting users in maintaining proper form and mitigating the risk of injury.

## 1.3 Scope and Limitations

The scope of this analysis was limited to the provided datasets. The analysis focused strictly on structured, table-based machine learning algorithms (caret package in R). A key limitation was the need to rely entirely on cross-validation metrics for out-of-sample performance estimation, as the true labels for the testing set were unknown at the time of development.

# 2. Methodology 

## 2.1 Data Sources and Overview 

The project utilizes accelerometer data from the belt, forearm, arm, and dumbbell of six participants. Two primary datasets were used:

- Training Set (**[pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)**): Contains 19,622 observations and 160 variables for model development and cross-validation.

- Test Set (**[pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)**): Contains 20 observations and 160 variables for final out-of-sample prediction and model validation.

- The data for this project come from the Weight Lifting Exercises  **[Dataset](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har)**page.

For more details, visit the University of California, Irvine **[Machine Learning Repository](https://archive.ics.uci.edu/)**.

## 2.2 Data Preprocessing and Cleaning  

Rigorous preprocessing was essential to handle data quality issues: Weight Lifting Exercises Dataset

- **Handling Missing Values:** Columns containing a significant number of missing values (represented as NA, #DIV/0!, or blank strings) were removed entirely.
- **Removal of Near-Zero Variance Features:** Features with minimal variation were identified and removed.
- **Exclusion of Identifier Columns:** Non-predictive meta-data columns, such as user_name and cvtd_timestamp, were excluded.

The cleaned training dataset maintained 19,622 observations but was reduced to 58 relevant features, a structure mirrored in the cleaned test dataset (20 observations, 58 variables). 

## 2.3 Analysis of Class Distribution

A visualization of the frequency of each exercise class (see *Figure 1 in Appendix A*) revealed a relatively balanced distribution:

- **Most Frequent Class:** classe A (5,580 observations).
- **Least Frequent Class:** classe D (3,216 observations).

This balanced distribution is crucial, as it validates that accuracy is a reliable performance metric and confirms that the model's high performance is genuine, not simply a result of class bias.

# 3. Results and Findings

## 3.1 Model Selection and Methodology

To identify the most effective algorithm for predicting the manner of exercise, a comparative evaluation was conducted across four distinct machine learning models:

- **XGBoost**, selected for its exceptional predictive accuracy on structured data.
- **Random Forest**, included for its high stability and robustness against overfitting.
- **SVM**, chosen for its effectiveness in high-dimensional spaces, a key characteristic of our sensor data.
- **kNN**, used as a non-parametric baseline to gauge the dataset's separability.

By evaluating a diverse set of algorithms, the procedure ensures that the final model selection is based on objective, comparative performance rather than arbitrary choice.

## 3.2 Cross-Validation and Hyperparameter Tuning

A 5-fold cross-validation scheme was employed to estimate model performance and tune hyperparameters. This approach, facilitated by the caret package, provided a robust and reliable estimate of performance on unseen data. A grid search was performed for each algorithm to find the optimal set of hyperparameters. 

## 3.3 Model Performance Comparison

Based on the aggregated cross-validation results summarized in *Table 1 (Appendix B)*, clear performance differences among the models were revealed. The model evaluation metrics indicate that both the XGBoost and Random Forest models delivered exceptional performance, achieving nearly identical mean accuracy scores (0.999300 and 0.999200, respectively) and excellent Mean Area Under the Curve (AUC) values of approximately 1.0.  

Despite similar accuracy levels, the XGBoost model demonstrated superior predictive confidence, evidenced by a significantly lower Mean LogLoss value (0.002500) compared to the Random Forest model (0.039500). The distributions of these performance metrics, visualized in the *boxplots in Figure 2 (Appendix A*) confirm that while the accuracy distributions for both models were comparable, the XGBoost model consistently maintained a tighter and substantially lower LogLoss distribution across all cross-validation folds. In contrast, the Support Vector Machine (SVM) and k-Nearest Neighbors (kNN) models were determined to be suboptimal and unsuitable for deployment with this high-dimensional dataset.

To further analyze classification precision, the *confusion matrices for the models were examined (see Figure 3 in Appendix A).* These heatmaps provided a detailed, class-by-class breakdown of performance. The XGBoost model recorded a total of 12 misclassifications across all cross-validation folds, fewer than the 21 total errors produced by the Random Forest model, indicating a slightly higher degree of classification precision in XGBoost model.

## 3.4 Estimated Out-of-Sample (OOS) Error and Model Selection

### 3.4.1  Estimated Out-of-Sample (OOS) Error

To evaluate generalization, Out-of-Sample (OOS) error (\(1-\text{Accuracy}\)) was estimated using 5-fold cross-validation and an isolated 30% hold-out set (see *Table 2, Appendix B*).

Key Findings:

- **Top Performers:** XGBoost and Random Forest achieved exceptional predictive power with OOS error rates below 0.10%.
- **Underperformers:** SVM recorded a significantly higher error of 8.75%, while k-NN failed to generalize with a 65.50% error rate, rendering distance-based models ineffective for this dataset.

### 3.4.2  Optimal Model Selection

XGBoost was selected as the final model for the 20-case prediction task. While Random Forest performed well, XGBoost demonstrated superior precision and stability, evidenced by:

- **Minimal Error:** A negligible 0.06% OOS error rate (representing only 12 misclassifications compared to Random Forest's 21).
- **Statistical Superiority:** A significantly lower Mean LogLoss (0.0025 vs. 0.0395).

With an anticipated accuracy of approximately 99.9%, XGBoost provides the most robust solution for distinguishing between correct form (Class A) and specific technical errors (Classes B–E).

# 4. Discussion and Interpretation

## 4.1 Interpretation of Findings

Based on the superior metrics and visual confirmation via confusion matrices, the XGBoost model was selected as the optimal choice. The feature importance ranking provided clear evidence for this decision:

- **Dominance of num_window:** The top feature was an engineered temporal feature, num_window (*Figure 6, Appendix A*). The visualization of this feature's distribution across classes (see *Figure 4 in Appendix A*) showed clean, non-overlapping decision boundaries, which the model exploited for high accuracy.

- **Importance of sensor data:** Direct sensor readings (roll_belt, magnet_dumbbell_y, pitch_forearm) were consistently ranked in the top 20 features by Gain.

## 4.2 Model Consistency and Robustness Analysis

To ensure the stability of the final metrics, the *confusion matrix for each of the cross-validation folds* was visualized (see *Figure 5 in Appendix A).* The resulting heatmaps consistently demonstrated exceptional stability across all folds, with concentrated high counts strictly along the main diagonal. This visual uniformity confirms minimal misclassification rates regardless of the specific data subset used for training.

## 4.3 Implications and Trade-offs

The high performance of Gradient Boosting confirmed that simple linear relationships were insufficient to explain the sensor data. The ability of XGBoost to model complex interactions justified selecting it over the slightly simpler Random Forest model. The performance gain outweighed any minor loss of inherent model simplicity, as interpretability tools can still provide robust explanations for the predictions.

# 5. Summary and Conclusion

A highly accurate and robust machine learning model was successfully developed and validated to predict the "manner of exercise" (classe) using wearable sensor data. A comprehensive, systematic workflow involved the training and rigorous comparison of four distinct algorithms, culminating in the selection of XGBoost as the optimal predictive solution. The key findings and conclusions derived from this process are summarized below:

- **Superior Ensemble Performance:** XGBoost and Random Forest decisively outperformed the alternative models, confirming the suitability and power of ensemble tree methods for high-dimensional sensor data analysis.

- **XGBoost's Predictive Confidence:** A deeper analysis utilizing the logLoss metric revealed XGBoost's superiority over Random Forest. A significantly lower logLoss value (0.0025 vs. 0.0395) indicates a higher degree of predictive confidence and precision, establishing XGBoost as the more reliable model.

- **Methodological Validation:** The discrepancy between perfect *in-sample metrics* (Accuracy = 1.0) and slightly lower cross-validated metrics (Accuracy = 0.9993) validates the critical importance of employing cross-validation. This difference underscores the necessity of obtaining an honest *out-of-sample* performance estimation over potentially optimistic *in-sample error rates*.

- **Reliability of Final Predictions:** The optimal XGBoost model was utilized to generate final predictions for the unseen test data. The model's proven high accuracy, robustness, and consistent behavior observed across cross-validation folds provide a strong, evidence-based foundation for the reliability of these predictions.

In conclusion, the developed XGBoost model represents a highly effective and trustworthy solution for this predictive challenge. The insights gained from interpreting its feature importance and the established confidence in its performance confirm its readiness for real-world application.

# 6. Recommendations and Future Work

- **Model Deployment:** The developed XGBoost model is recommended for immediate deployment due to its high accuracy and confirmed robustness in classifying exercise execution. Focus efforts on wrapping the existing R model into a robust API to facilitate real-time inference in the target application. 

- **Future Research:** Investigate *deep learning approaches*, such as recurrent neural networks (RNNs) or LSTMs. These architectures specialize in time-series data and may capture more nuanced temporal patterns, potentially leading to further performance improvements.


\newpage


# References

1. Velloso, E., Bulling, A., Gellersen, H., Ugulino, W., & Fuks, H. (2013, March). Qualitative Activity Recognition of Weight Lifting Exercises. In Proceedings of the 4th International Conference in Cooperation with SIGCHI on Augmented Human (AH '13). ACM.
  
2. Velloso, E., Bulling, A., Kuflik, T., Concejero, J., Nova, F., & Bianchi, A. (2014, September). Qualitative Activity Recognition of Weight Lifting Exercises. In Proceedings of the 2014 International Symposium on Wearable Computers (ISWC). ACM.

3. Pontifical Catholic University of Rio de Janeiro, Department of Informatics. (n.d.). HAR related publications. Retrieved from web.archive.org

# Appendices

## Appendix A: Figures

![](index_files/figure-html/plotting_figure_1-1.png)<!-- -->

![](index_files/figure-html/plotting_figure_2-1.png)<!-- -->

![](index_files/figure-html/plotting_figure_3-1.png)<!-- -->

![](index_files/figure-html/plotting_figure_4-1.png)<!-- -->

![](index_files/figure-html/plotting_figure_5-1.png)<!-- -->

![](index_files/figure-html/plotting_figure_6-1.png)<!-- -->

## Appendix B: Tables


Table: **Table 1:** Comparison of Performance Metrics

|        Model| Accuracy|  Kappa| logLoss|    AUC|
|------------:|--------:|------:|-------:|------:|
|      XGBoost|   0.9994| 0.9992|  0.0027| 1.0000|
| RandomForest|   0.9991| 0.9988|  0.0392| 1.0000|
|          SVM|   0.9102| 0.8861|  0.3353| 0.9895|
|          kNN|   0.3430| 0.1669| 12.9071| 0.6348|

<table class="table table-striped table-hover table-condensed table-responsive" style="margin-left: auto; margin-right: auto;">
<caption>Table 2: Estimated Out-of-Sample Error Comparison</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> Model </th>
   <th style="text-align:right;"> Mean CV Accuracy </th>
   <th style="text-align:right;"> Validation Accuracy </th>
   <th style="text-align:right;"> Expected OOS Error </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> XGBoost </td>
   <td style="text-align:right;"> 0.9993 </td>
   <td style="text-align:right;"> 0.9994 </td>
   <td style="text-align:right;"> 0.0006 (0.06%) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Random Forest </td>
   <td style="text-align:right;"> 0.9992 </td>
   <td style="text-align:right;"> 0.9991 </td>
   <td style="text-align:right;"> 0.0009 (0.09%) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> SVM </td>
   <td style="text-align:right;"> 0.9131 </td>
   <td style="text-align:right;"> 0.9125 </td>
   <td style="text-align:right;"> 0.0875 (8.75%) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> k-NN </td>
   <td style="text-align:right;"> 0.3441 </td>
   <td style="text-align:right;"> 0.3450 </td>
   <td style="text-align:right;"> 0.6550 (65.50%) </td>
  </tr>
</tbody>
</table>



