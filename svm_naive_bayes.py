import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# load data
df = pd.read_csv("clean_data.csv", encoding="cp1252")

# just grab the cols we need
df = df[["Narrative Text", "Event Label"]].dropna()

X = df["Narrative Text"]
y = df["Event Label"]

# split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,y, test_size=0.30,random_state=42, stratify=y
)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp,test_size=0.50, random_state=42,stratify=y_temp
)

tuning_results = []
final_results = []

# -------------------
# Naive Bayes model
# -------------------
print("...running first model (naive bayes)")

best_nb_model = None
best_nb_dev_acc = 0
best_nb_params = None

count = 0

for max_features in [10000,20000]:
    for ngram_range in [(1, 1),(1, 2)]:
        for min_df in [1,2]:
            for alpha in [0.1,0.5, 1.0]:
                count += 1
                print("naive bayes progress:", count)


                nb_model = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        stop_words="english",
                        max_features=max_features,
                        ngram_range=ngram_range,
                        min_df=min_df
                    )),
                    ("clf",MultinomialNB(alpha=alpha))
                ])

                nb_model.fit(X_train,y_train)

                train_pred = nb_model.predict(X_train)
                dev_pred=nb_model.predict(X_dev)

                train_acc = accuracy_score(y_train,train_pred)
                dev_acc = accuracy_score(y_dev, dev_pred)

                tuning_results.append({
                    "model": "Naive Bayes",
                    "max_features": max_features,
                    "ngram_range": str(ngram_range),
                    "min_df": min_df,
                    "alpha": alpha,
                    "train_accuracy": train_acc,
                    "dev_accuracy": dev_acc
                })

                if dev_acc > best_nb_dev_acc:
                    best_nb_dev_acc = dev_acc
                    best_nb_model = nb_model
                    best_nb_params = {
                        "max_features": max_features,
                        "ngram_range": ngram_range,
                        "min_df": min_df,
                        "alpha": alpha
                    }

nb_train_acc = accuracy_score(y_train,best_nb_model.predict(X_train))
nb_test_acc = accuracy_score(y_test, best_nb_model.predict(X_test))

final_results.append({
    "model": "Naive Bayes",
    "best_params": str(best_nb_params),
    "train_accuracy": nb_train_acc,
    "dev_accuracy": best_nb_dev_acc,
    "test_accuracy": nb_test_acc
})

# SVM
print("...running second model (svm)")

best_svm_model = None
best_svm_dev_acc = 0
best_svm_params = None

count = 0

for max_features in [10000, 20000]:
    for ngram_range in [(1,1), (1, 2)]:
        for min_df in [1, 2]:
            for c_value in [0.1,1.0,5.0]:

                count += 1
                print("  svm progress:", count)

                svm_model = Pipeline([
                    ("tfidf",TfidfVectorizer(
                        stop_words="english",
                        max_features=max_features,
                        ngram_range=ngram_range,
                        min_df=min_df
                    )),
                    ("clf", LinearSVC(C=c_value))
                ])

                svm_model.fit(X_train, y_train)

                train_acc = accuracy_score(y_train, svm_model.predict(X_train))
                dev_acc = accuracy_score(y_dev,svm_model.predict(X_dev))

                tuning_results.append({
                    "model": "SVM",
                    "max_features": max_features,
                    "ngram_range": str(ngram_range),
                    "min_df": min_df,
                    "C": c_value,
                    "train_accuracy": train_acc,
                    "dev_accuracy": dev_acc
                })

                if dev_acc > best_svm_dev_acc:
                    best_svm_dev_acc = dev_acc
                    best_svm_model = svm_model
                    best_svm_params = {
                        "max_features": max_features,
                        "ngram_range": ngram_range,
                        "min_df": min_df,
                        "C": c_value
                    }

svm_train_acc = accuracy_score(y_train, best_svm_model.predict(X_train))
svm_test_acc = accuracy_score(y_test,best_svm_model.predict(X_test))

final_results.append({
    "model": "SVM",
    "best_params": str(best_svm_params),
    "train_accuracy": svm_train_acc,
    "dev_accuracy": best_svm_dev_acc,
    "test_accuracy": svm_test_acc
})

# results
tuning_results_df = pd.DataFrame(tuning_results)
final_results_df = pd.DataFrame(final_results)

print("\nFinal results:")
print(final_results_df)

tuning_results_df.to_csv("hyperparameter_tuning_results.csv",index=False)

from sklearn.metrics import classification_report


print("\nNaive Bayes Report")
nb_final_preds = best_nb_model.predict(X_test)
print(classification_report(y_test, nb_final_preds))

print("\n SVM Detailed Report")
svm_final_preds = best_svm_model.predict(X_test)
print(classification_report(y_test, svm_final_preds))