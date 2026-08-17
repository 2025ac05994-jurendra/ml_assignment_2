from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def build_logistic_regression():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_decision_tree():
    return DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_knn():
    return KNeighborsClassifier(
        n_neighbors=25,
        weights="distance",
        n_jobs=-1,
    )


def build_naive_bayes():
    return GaussianNB()


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

CLASSIFIERS = {
    "Logistic Regression": (build_logistic_regression, "logistic_regression"),
    "Decision Tree": (build_decision_tree, "decision_tree"),
    "kNN": (build_knn, "knn"),
    "Naive Bayes": (build_naive_bayes, "naive_bayes"),
    "Random Forest": (build_random_forest, "random_forest"),
}


def build(model_name):
    if model_name not in CLASSIFIERS:
        raise KeyError(f"Unknown model {model_name!r}. Choose from {list(CLASSIFIERS)}.")
    return CLASSIFIERS[model_name][0]()


def file_stem(model_name):
    return CLASSIFIERS[model_name][1]


def all_file_stems():
    return {name: stem for name, (_, stem) in CLASSIFIERS.items()}
