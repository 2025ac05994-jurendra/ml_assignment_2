from sklearn.linear_model import LogisticRegression

from data_prep import RANDOM_STATE, train_and_save

MODEL_NAME = "Logistic Regression"
FILE_STEM = "logistic_regression"


def build_classifier():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    train_and_save(MODEL_NAME, build_classifier(), FILE_STEM)
