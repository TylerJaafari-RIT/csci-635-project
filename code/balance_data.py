import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

train = pd.read_csv("../data/clean/twitter_training_clean.csv")
val = pd.read_csv("../data/clean/twitter_validation_clean.csv")
test = pd.read_csv("../data/clean/twitter_test_clean.csv")

labels = ["Negative", "Neutral", "Positive", "Irrelevant"]


ros = RandomOverSampler(random_state=0)
train_resampled, tmp = ros.fit_resample(train,train["Sentiment"])
#val_resampled, tmp = ros.fit_resample(val,val["Sentiment"])

## Balance

fig, ax = plt.subplots(1,4,num='Post-Balancing Distribution')
fig.suptitle("Distributions")

train_counts = train["Sentiment"].value_counts()
ax[0].bar(labels, train_counts)
ax[0].tick_params(axis='x',labelrotation=90)
ax[0].set_title("Train")

train_counts = train_resampled["Sentiment"].value_counts()
ax[1].bar(labels, train_counts)
ax[1].tick_params(axis='x',labelrotation=90)
ax[1].set_title("Train (Oversampled)")

val_counts = val["Sentiment"].value_counts()
ax[2].bar(labels, val_counts)
ax[2].tick_params(axis='x',labelrotation=90)
ax[2].set_title("Validation")

test_counts = test["Sentiment"].value_counts()
ax[3].bar(labels, test_counts)
ax[3].tick_params(axis='x',labelrotation=90)
ax[3].set_title("Test")

plt.tight_layout()
plt.draw()

train_resampled.to_csv("../data/clean/twitter_train_balanced.csv")


plt.show()