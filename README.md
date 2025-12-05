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