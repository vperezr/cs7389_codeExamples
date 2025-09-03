# 📘 NLP Baselines 
# ✨ Theme: "One Baseline Does Not Fit All"
# 🔧 Dependencies: datasets, sklearn, nltk, transformers 
!pip install datasets scikit-learn nltk --quiet

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import random
import numpy as np
import nltk
nltk.download('punkt')

# ------------------------------
# 🧪 Task 1: Sentiment Classification (IMDb subset)
# ------------------------------
print("\n🧪 Task 1: Sentiment Classification")
imdb = load_dataset("imdb", split="train[:1000]")  # small sample for speed
texts = imdb["text"]
labels = imdb["label"]

# Baseline 1: Majority Class
majority = max(set(labels), key=labels.count)
preds_majority = [majority] * len(labels)
print("Majority Class Baseline Accuracy:", accuracy_score(labels, preds_majority))

# Baseline 2: Random Guess
random_preds = [random.choice([0,1]) for _ in labels]
print("Random Baseline Accuracy:", accuracy_score(labels, random_preds))

# Baseline 3: BOW + Logistic Regression
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)
clf = LogisticRegression(max_iter=300).fit(X, labels)
preds_lr = clf.predict(X)
print("BOW + Logistic Regression Accuracy:", accuracy_score(labels, preds_lr))

# ------------------------------
# 🧪 Task 2: Question Answering (SQuAD lite-style task)
# ------------------------------
print("\n🧪 Task 2: Question Answering")
squad = load_dataset("squad", split="train[:100]")  # small for demo

# Baseline 1: Return First Sentence
from nltk.tokenize import sent_tokenize
first_sent_answers = [sent_tokenize(context)[0] if sent_tokenize(context) else "" 
                      for context in squad["context"]]

# Evaluate rough overlap
def simple_match(pred, ref):
    return 1 if ref.lower() in pred.lower() else 0

accuracy = np.mean([simple_match(pred, ref) 
                    for pred, ref in zip(first_sent_answers, squad["answers"])])
print("First Sentence Baseline QA Accuracy:", accuracy)

# ------------------------------
# 🧪 Task 3: Textual Entailment (SNLI subset)
# ------------------------------
print("\n🧪 Task 3: Textual Entailment")
snli = load_dataset("snli", split="train[:1000]")
snli = snli.filter(lambda x: x['label'] != -1)  # remove ambiguous

X = [p + " " + h for p, h in zip(snli["premise"], snli["hypothesis"])]
y = snli["label"]  # 0 = entailment, 1 = neutral, 2 = contradiction

# Baseline 1: Predict "entailment" always (label 0)
y_ent = [0] * len(y)
print("Always Entailment Accuracy:", accuracy_score(y, y_ent))

# Baseline 2: BOW + Logistic Regression
X_vec = vectorizer.fit_transform(X)
clf_entail = LogisticRegression(max_iter=300).fit(X_vec, y)
y_pred = clf_entail.predict(X_vec)
print("BOW + Logistic Regression Accuracy:", accuracy_score(y, y_pred))
print("Macro F1 Score:", f1_score(y, y_pred, average='macro'))

# ------------------------------
# 🧠 Discussion Prompts
print("\n💬 Discussion:")
print("- Which baseline was surprisingly strong? Which was useless?")
print("- How does task structure change what counts as a reasonable baseline?")
print("- What are the dangers of comparing new models only to weak baselines?")
print("- How would you construct a fair, informative baseline for your own project?")

