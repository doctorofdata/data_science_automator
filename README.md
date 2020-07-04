# AI Pipeline for Building Data Science Models - 

## By: ZXS

Pipeline for building a series of models on a given dataset for obtaining preliminary data science results before further investigation. 

---

*Requirements:*
1) PATH of desired CSV file
2) PATH for output directory
3) List of categorical variables for transformation to numeric for modeling
4) Column name for dependent variable
5) Precompiled list of desired models, including necessary parameters

*Outputs:*
*Per model-*
* Confusion matrix for performance on validation data
* Classification report for predictions
* ROC for model saved to output_directory
