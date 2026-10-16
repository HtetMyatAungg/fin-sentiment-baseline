import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from scipy.stats import bootstrap
df = pd.read_csv('Data/Sentences_50Agree.txt', encoding='latin-1', sep='@', header=None, names=['sentence', 'label'])

df.drop_duplicates(inplace=True)

# Split the dataset into training and testing sets
X = df['sentence']
y = df['label']

plt.figure(figsize=(8, 6))
# Plot the distribution of labels
total = len(y)
neutral, positive, negative = y.value_counts()
plt.bar(['Neutral', 'Negative', 'Positive'], [neutral/total, negative/total, positive/total])
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Sentences (%)')
plt.xticks(rotation=0)
plt.savefig('label_distribution.png')


stratified_split = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y)

x_val, x_test, y_val, y_test = train_test_split(stratified_split[1], stratified_split[3], test_size=0.5, random_state=42, stratify=stratified_split[3])
x_train, y_train = stratified_split[0], stratified_split[2]
with open('train.csv', 'w') as f:
    f.write('sentence,label\n')
    for sentence, label in zip(x_train, y_train):
        f.write(f'"{sentence}",{label}\n')

with open('test.csv', 'w') as f:
    f.write('sentence,label\n')
    for sentence, label in zip(x_test, y_test):
        f.write(f'"{sentence}",{label}\n')

with open('val.csv', 'w') as f:
    f.write('sentence,label\n')
    for sentence, label in zip(x_val, y_val):
        f.write(f'"{sentence}",{label}\n')

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(x_train)
X_test_tfidf = vectorizer.transform(x_test)
X_val_tfidf = vectorizer.transform(x_val)

model = LogisticRegression(random_state=0).fit(X_train_tfidf, y_train)
""" macro_f1 = f1_score(y_test, model.predict(X_test_tfidf), average='macro')
macro_f1_val = f1_score(y_val, model.predict(X_val_tfidf), average='macro')
plt.figure(figsize=(8, 6))
# Plot the macro F1 scores for test and validation sets
plt.bar(['Test Set', 'Validation Set'], [macro_f1, macro_f1_val])
plt.title('No Validation Tuning Loop Macro F1 Scores')
plt.ylabel('Macro F1 Score')
plt.savefig('macro_f1_scores.png')"""

pipeline = make_pipeline(vectorizer, LogisticRegression(random_state=0))
scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring='f1_macro')

y_pred = model.predict(X_test_tfidf)


def calculate_macro_f1(y_test, y_pred):
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    return macro_f1

bootstrap_result = bootstrap((y_test, y_pred), calculate_macro_f1, method='percentile', confidence_level=0.95, paired=True, vectorized=False, n_resamples=1000, random_state=42)

print(bootstrap_result.confidence_interval, bootstrap_result.standard_error)

