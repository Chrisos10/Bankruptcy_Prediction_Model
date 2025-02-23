# Overview
The project in this repository aims to develop an ML-driven bankruptcy prediction model that can enable entrepreneurs to proactively identify and mitigate financial risks.

# The Dataset
The dataset used in this project includes various financial ratios and metrics that serve as features to predict the likelihood of bankruptcy.

This dataset is valuable for training machine learning models to identify patterns in financial data that could indicate a company's risk of bankruptcy, aiding investors, entrepeneurs, and financial institutions in making informed decisions.

Link to the dataset: https://www.kaggle.com/code/ahmedtronic/company-bankruptcy-prediction

# Problem Statement
“How can the insights from existing bankruptcy prediction models, be leveraged to develop a machine learning solution, aiming to predict and mitigate the risk of bankruptcy, enabling entrepreneurs to make timely interventions and informed decisions?


# Classic ML Model Findings
The classical model used random forest classifier. It did well in predicting bankruptcy with an accuracy of 97%. However, it performs far better in predicting the negative class with a precision of 98%, a recall of 99%, and an f1 -score of 98%. When predicting the positive class, these metrics drop drastically, where the precision falls to 56%, a recall of 0.32, and an f1-score of 41%. This was due to the dataset used which had a huge class imbalance, making the model learn far better from the negative class data than it did on the positive class.

Another remark is the effect of the hyperparameters tuning which is slightly. Changed the accuracy of the model from 96% to 97%. The change was slight because the data did not much complexity as it turned out, except for the class imbalance challenge only.

          precision    recall  f1-score 

       0       0.98      0.99      0.98 
       1       0.56      0.32      0.41 

       accuracy                    0.97      

# Neural Networks Findings
## Summary Table

| Training Instance  | Optimizer Used  | Reguralizer Used  | Epochs  | Early Stopping  | Number Of Layers  | Learning Rate  |Accuracy  | Recall  | F1 Score | Precision|
|--------            |--------         |--------           |-------- |--------         |--------           |--------        |--------  |-------- |--------  |--------  |
| Instance 1         | None            | None              | 10      | no              | 5                 | Default        | 95%      | 0.18    | 0.25     | 0.25     |
| Instance 2         | adam            | l2                | 20      | yes             | 5                 | 0.0005         | 96%      | 0.25    | 0.35     | 0.57     |
| Instance 3         | RMSPROP         | l2                | 20      | yes             | 5                 | 0.0001         | 95%      | 0.11    | 0.19     | 0.62     |
| Instance 4         | nadam           | l2                | 20      | yes             | 5                 | 0.0002         | 96%      | 0.13    | 0.23     | 0.66     |
| Instance 5         | adagrad         | l1_l2             | 30      | yes             | 5                 | 0.001          | 95%      | 0.06    | 0.12     | 0.75     |


## Discusion Of Findings
The neural networks were built while iterating different parameters to compare their impacts on performance. In the first instance where no parameters where set or regularized, the model showed a good performance on the training sets but when it came to new data, it showed poor generalization of the information mainly because it almost memorized the training data instead of identifying trends. It had an accuracy of 95%, a precision of 0.3462, a recall of 0.2093, and an F1 Score of 0.2609. The low The value of recall is also a result of the dataset which had low positives and so the computations of recall always tend to be low for all models.


The other instances tried different combinations of optimizers, regularizers, early stopping and learning rates in an attempt to improve the model performance in addition to identifying their impacts on models. The best resulting model had the following metrics:


Accuracy: 0.9609
Precision: 0.5789
Recall: 0.2558
F1 Score: 0.3548


It shows a higher accuracy and precision but the recall is low per the reason identified above(Low positive data in the dataset). It had 512 neurons which were able to identify the trends. It also used adam as an optimizer and applied a dropout of 0.2 on the hidden layers, an l2 regularizer with a rate of 0.05 which carefully regularized the weights allowing more generalization and reduced overfitting.


**Different Optimizers**: Since optimizers control how weights are updated during training, turns out that Adam optimizer had the most accurate updates of weights and biases than RMSprop, adagrad or even nadam that I used on other instances. This is concluded based on the facts that regularizers did not change much and so optimizers are accountable.
# A Neural Network VS A Classic ML Model
The classic ML model performed better than the neural network. This can be seen from a difference in metrics where the classic ml has a slightly higher accuracy of 97% while the NN had 96%. Also, the precision was much higher and other metrics like f1-score and recall although they were both affected by the high imbalance in the dataset of the positive and the negative classes.

Some NNs also presented risks of overfitting and they might be too smart for these type of classifications where the complexity can not be termed to be that high.
This lead to a conclusion that **model1** in the saved modls directory was the best model in this project.

# Link To Video
Discussion Video Link: https://drive.google.com/file/d/1uywMFtgc2QO_NXcjZr8MAVoaXkOF90Ob/view?usp=sharing
# Notebook Usage & Loading The Best Model
There are various ways to access the notebook
One of them is to use the notebook through google drive via the following commands;
- clone the repo using;
```
git clone Repo URL
```
Upload the notebook to google drive and then open the notebook. 

To laoad the best model(model1.pkl);
Follow these commands
```
import pickle
```
```
with open("model_random_classifier.pkl", "rb") as f:
    model = pickle.load(f)

```