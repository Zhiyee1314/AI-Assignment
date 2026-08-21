"""Compatibility entry point. Shared preprocessing requires joint retraining."""

from train_all_models import main


if __name__ == "__main__":
    print("Training ANN, SVM and KNN together to keep preprocessing identical.")
    main()
