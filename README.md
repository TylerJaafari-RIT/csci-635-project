# Twitter Sentiment Analysis: DistilBERT vs Traditional Models
Semester project for CSCI-635: Introduction to Machine Learning at RIT, Fall 2025.


# Abstract
In this project, we investigated the relative performance of the DistilBERT Large Language Model against four traditional machine learning models. We used two basic models (Linear Support Vector Machine, K Nearest Neigbors) and two ensemble models utilizing decision trees (AdaBoost, Random Forest) to classify tweets based on TF-IDF matrices . We found that the best DistilBERT model (trained with a learning rate of $\eta=2\times 10^-6$) performed on par with the best Random Forest Model and slightly worse than the best K Nearest Neigbors model did, while all three of these models outperformed the Linear Support Vector Machine and AdaBoost models. Furthermore, the K Nearest Neighbors algorithm was much faster to train and fit than DistilBERT, and Random Forest was faster than either leading to a preference for using those two models for evaluating sentiments towards known entities. However, it is likely that DistilBERT or similar models would be more flexible for extending this problem to entities outside of the training set, making it still a reasonable option for general applications while Random Forest and K Nearest Neigbors would be better to train on-demand for a known entity.


# Contributors
### Liam Hainsworth
* Initial data processing and TF-IDF including data balancing
* LLM training and evaluation
    * BERT-process-data
    * BERT-train
    * BERT-evaluate

### Tyler Jaafari
* Refactored and optimized data processing code and TF-IDF
    * dataprocess.py
    * balance_data.py
* Traditional models
    * KNN
    * SVC
    * AdaBoost
    * RandomForest

# Running the Project
### Preparing Data
To prepare the data, run
```[bash]
$ cd code/
$ python3 data_processing.py
$ python3 balance_data.py
```
to clean the data files and create the train, validation, and test data files as well as the balanced training data file in the `data/clean` folder. 

### Traditional Models
Each of the traditional models is contained within a jupyter notebook in the `code/` directory. To run any of them, open the corresponding notebook and select the `Run all cells` option. The four notebooks are:
* `code/KNN.ipynb`
* `code/SVC.ipynb`
* `code/RandomForest.ipynb`
* `code/AdaBoost.ipynb`

### DistilBERT
To run the DistilBERT model, first run all cells in the `BERT-data-process.ipynb` notebook using jupyter to create the labeled data in the `data/bert-ds-labeled` folder.

Then to train the model, run all cells in the `BERT-train.ipynb` notebook using jupyter. As a warning, this took 30 minutes per epoch on a highly accelerated Google Colab instance, so it is recommended using the provided best model in `models/2neg6-checkpoint-90018` to evaluate performance unless re-training is absolutely necessary. The evaluation notebook is already programmed to use this by default.

To evaluate the trained DistilBERT model, run all cells in the `BERT-evaluate.ipynb` notebook using jupyter. By default, this uses the provided best model in `models/2neg6-checkpoint-90018`, but this can be changed by modifying the `model_name` variable in the "Prediction/Load pretrained model" section of the notebook. 

The last section of the `BERT-evaluate` notebook also plots the loss over epoch for the best training run, but uses the copy-and pasted trainer output (from the table displayed while running) - the `loss` variable in the "Plots" section needs to be changed to plot for a different run.