from sklearn.naive_bayes import GaussianNB

from data_prep import train_and_save

MODEL_NAME = "Naive Bayes"
FILE_STEM = "naive_bayes"


def build_classifier():
    return GaussianNB()


if __name__ == "__main__":
    train_and_save(MODEL_NAME, build_classifier(), FILE_STEM)
