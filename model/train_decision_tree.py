from sklearn.tree import DecisionTreeClassifier

from data_prep import RANDOM_STATE, train_and_save

MODEL_NAME = "Decision Tree"
FILE_STEM = "decision_tree"


def build_classifier():
    return DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    train_and_save(MODEL_NAME, build_classifier(), FILE_STEM)
