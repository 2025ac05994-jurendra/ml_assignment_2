import train_decision_tree
import train_knn
import train_logistic_regression
import train_naive_bayes
import train_random_forest

MODEL_MODULES = [
    train_logistic_regression,
    train_decision_tree,
    train_knn,
    train_naive_bayes,
    train_random_forest,
]

CLASSIFIERS = {
    module.MODEL_NAME: (module.build_classifier, module.FILE_STEM)
    for module in MODEL_MODULES
}


def build(model_name):
    if model_name not in CLASSIFIERS:
        raise KeyError(f"Unknown model {model_name!r}. Choose from {list(CLASSIFIERS)}.")
    return CLASSIFIERS[model_name][0]()


def file_stem(model_name):
    return CLASSIFIERS[model_name][1]


def all_file_stems():
    return {name: stem for name, (_, stem) in CLASSIFIERS.items()}
