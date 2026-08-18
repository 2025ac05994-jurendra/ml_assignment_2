from sklearn.ensemble import RandomForestClassifier

from data_prep import RANDOM_STATE, train_and_save

MODEL_NAME = "Random Forest"
FILE_STEM = "random_forest"


def build_classifier():
    return RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    train_and_save(MODEL_NAME, build_classifier(), FILE_STEM)
