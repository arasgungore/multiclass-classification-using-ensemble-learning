# multiclass-classification-using-ensemble-learning

Two ensemble models made from ensembles of LightGBM and CNN for a multiclass classification problem. The training set contains 62,5k samples and 108 features and a target "y" integer corresponding to the class of the sample. The test set contains 150k samples and 108 features. There are a total of 9 classes and the goal is to minimize the categorical cross-entropy loss (log loss) and maximize the accuracy of our predictions.



## Run on Terminal

**a) Run CNN ensemble model**
```sh
jupyter nbconvert --to notebook --execute cnn_ensemble.ipynb
```

**b) Run LightGBM ensemble model**
```sh
jupyter nbconvert --to notebook --execute lightgbm_ensemble.ipynb
```



## Task Description and Guidance

### Task Overview

The goal of the machine learning task is to predict the target values for the test set using the model trained on the training set. The predictions should be in the format of a tabular dataset with 150,000 rows and 9 columns, where each column represents the predicted probability for one of the 9 classes.


### Required Format

The expected format for the predictions is a tabular dataset with 150,000 rows and 9 columns, where each column corresponds to one of the classes (c1 to c9). The cells in each column should contain the predicted probability for the respective class.

```plaintext
c1   c2   c3   c4   c5   c6   c7   c8   c9
0.1  0.2  0.3  0.0  0.4  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.8  0.0  0.0  0.0  0.1  0.1
...
```


### Modeling and Recommendations

1. The dataset is tabular with no time series features and does not contain categorical variables.

2. Consider using a supervised learning approach such as Decision Trees or Neural Networks to start the modeling process.

3. For feature engineering, instead of PCA, consider using non-linear encoding methods like UMAP or t-SNE to capture complex relationships in the data. The resulting embeddings can be added as additional features.

4. Utilize an 80/20 train/test split for the dataset, and consider employing k-fold cross-validation for model evaluation.

5. In stacking ensemble models, including lower-performing models like Decision Trees, Neural Networks, or Random Forest can contribute to overall performance.

6. In terms of time efficiency, LightGBM models may be more suitable than xgboost for parameter search in decision tree models.

7. For stacking ensemble models, consider using SVM logistic regression models, as they can be beneficial for second-level modeling.

8. If further competitive scores are desired, explore Neural Network models and ensemble the predictions of all models.

9. When stacking models, focus on the uniqueness of predictions as well as their accuracy.

10. For stacking, a second-level LightGBM model can be effective.

Feel free to experiment with the suggestions above and evaluate the results in your work.



## Author

👤 **Aras Güngöre**

* LinkedIn: [@arasgungore](https://www.linkedin.com/in/arasgungore)
* GitHub: [@arasgungore](https://github.com/arasgungore)
