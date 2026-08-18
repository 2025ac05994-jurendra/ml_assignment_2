from sklearn.neighbors import KNeighborsClassifier

from data_prep import train_and_save

MODEL_NAME = "kNN"
FILE_STEM = "knn"


def build_classifier():
    return KNeighborsClassifier(
        n_neighbors=25,
        weights="distance",
        n_jobs=-1,
    )


if __name__ == "__main__":
    train_and_save(MODEL_NAME, build_classifier(), FILE_STEM)
