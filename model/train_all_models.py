import json

import pandas as pd

from classifiers import CLASSIFIERS, all_file_stems
from data_prep import (METRIC_NAMES, MODEL_DIR, RANDOM_STATE, build_pipeline, evaluate,
                       export_test_csv, get_train_test, report, save_pipeline, write_schema)


def main():
    X_train, X_test, y_train, y_test = get_train_test()
    print(f"Training rows : {len(X_train):,}  ({y_train.mean():.2%} positive)")
    print(f"Test rows     : {len(X_test):,}  ({y_test.mean():.2%} positive)")
    print(f"Predictors    : {X_train.shape[1]}")

    results = {}

    for model_name, (build_classifier, file_stem) in CLASSIFIERS.items():
        pipeline = build_pipeline(build_classifier(), X_train)
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        report(model_name, metrics)
        save_pipeline(pipeline, file_stem)

        results[model_name] = metrics

    print("\nArtifacts")
    print("-" * 9)
    export_test_csv(X_test, y_test)
    write_schema(X_train, all_file_stems())

    (MODEL_DIR / "metrics.json").write_text(json.dumps({
        "dataset": "UCI Bank Marketing (bank-full.csv)",
        "test_rows": int(len(y_test)),
        "positive_rate": round(float(y_test.mean()), 4),
        "dropped_features": ["duration"],
        "random_state": RANDOM_STATE,
        "models": {name: {k: round(float(v), 4) for k, v in row.items()}
                   for name, row in results.items()},
    }, indent=2))
    print("  saved -> metrics.json")
    print("  saved -> schema.json")

    table = pd.DataFrame(results).T[METRIC_NAMES]
    print("\nComparison table")
    print("=" * 72)
    print(table.round(4).to_string())
    print("=" * 72)
    print(f"\nBest MCC: {table['MCC'].idxmax()} ({table['MCC'].max():.4f})")


if __name__ == "__main__":
    main()
