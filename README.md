# multiclass-classification-using-ensemble-learning

Two ensemble models based on CNN and LightGBM for a multiclass classification problem. The training set contains 62,5k samples and 108 features and a target "y" integer corresponding to the class of the sample. The test set contains 150k samples and 108 features. There are a total of 9 classes and the goal is to minimize the categorical cross-entropy loss (log loss) and maximize the accuracy of our predictions.



## Run on Terminal

**a) Run CNN ensemble model**
```sh
jupyter nbconvert --to notebook --execute cnn_ensemble.ipynb
```

**b) Run LightGBM ensemble model**
```sh
jupyter nbconvert --to notebook --execute lightgbm_ensemble.ipynb
```



## Author

👤 **Aras Güngöre**

* LinkedIn: [@arasgungore](https://www.linkedin.com/in/arasgungore)
* GitHub: [@arasgungore](https://github.com/arasgungore)
