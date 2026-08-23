from __future__ import annotations

import json
from pathlib import Path

from src.data import audit_raw_data, clean_transactions, download_dataset, load_raw_transactions, save_json
from src.evaluation import add_relative_segment_labels, summarize_clusters, validate_segment_output
from src.features import build_customer_features, prepare_clustering_matrix
from src.model import cluster_stability, evaluate_cluster_counts, fit_final_kmeans

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


def main() -> None:
    raw_dir = DATA_DIR / "raw"
    processed_dir = DATA_DIR / "processed"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    workbook = download_dataset(raw_dir)
    raw = load_raw_transactions(workbook)
    raw_audit = audit_raw_data(raw)

    clean, cleaning_report = clean_transactions(raw)
    customers = build_customer_features(clean)
    matrix, transformed, _, preprocessing_metadata = prepare_clustering_matrix(customers)

    selection = evaluate_cluster_counts(matrix)
    model, labels = fit_final_kmeans(matrix, selection.selected_k)
    validate_segment_output(customers, labels)

    stability = cluster_stability(matrix, selection.selected_k, labels)
    summary = add_relative_segment_labels(summarize_clusters(customers, labels))

    assignments = customers.copy()
    assignments["cluster"] = labels

    clean.to_csv(processed_dir / "clean_transactions.csv", index=False)
    assignments.to_csv(RESULTS_DIR / "customer_segments.csv", index=False)
    selection.diagnostics.to_csv(RESULTS_DIR / "cluster_diagnostics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "cluster_summary.csv", index=False)

    verification = {
        "verification_pass": True,
        "source": "UCI Machine Learning Repository - Online Retail",
        "raw_audit": raw_audit,
        "cleaning": cleaning_report,
        "customer_feature_rows": int(len(customers)),
        "selected_k": int(selection.selected_k),
        "selected_silhouette": float(
            selection.diagnostics.loc[
                selection.diagnostics["k"].eq(selection.selected_k), "silhouette"
            ].iloc[0]
        ),
        "cluster_stability": stability,
        "preprocessing": preprocessing_metadata,
        "cluster_centres_scaled": model.cluster_centers_.tolist(),
        "output_files": [
            "results/customer_segments.csv",
            "results/cluster_diagnostics.csv",
            "results/cluster_summary.csv",
        ],
        "limitations": [
            "KMeans imposes distance-based partitions and does not prove that natural customer segments exist.",
            "RFM summarizes transaction behaviour and does not capture demographics, channel exposure or profit margin.",
            "Segment names are relative descriptions of this dataset, not universal customer personas.",
            "The analysis is descriptive and should not be interpreted as causal evidence for marketing actions.",
        ],
    }
    save_json(verification, RESULTS_DIR / "verification.json")
    save_json(raw_audit, RESULTS_DIR / "raw_data_audit.json")
    save_json(cleaning_report, RESULTS_DIR / "cleaning_report.json")

    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
