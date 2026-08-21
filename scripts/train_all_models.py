"""Train ANN, SVM and KNN, then generate the feature-ablation CSV report."""

from training_pipeline import ABLATION_PATH, COMPARISON_PATH, MODELS_DIR, train_and_analyse


def main():
    comparison, ablation = train_and_analyse()
    print("\nModel comparison:\n", comparison.to_string(index=False))
    print("\nFeature ablation:\n", ablation.to_string(index=False))
    print(f"\nSaved models to: {MODELS_DIR}")
    print(f"Saved comparison report to: {COMPARISON_PATH}")
    print(f"Saved ablation report to: {ABLATION_PATH}")


if __name__ == "__main__":
    main()

