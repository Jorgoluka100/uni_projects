"""Pinned source configuration for the Olist analytics project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FILE_TABLES = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "payments",
    "olist_order_reviews_dataset.csv": "reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "category_translation",
}

EXPECTED_FILE_SHA256 = {
    "olist_customers_dataset.csv": "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
    "olist_geolocation_dataset.csv": "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6",
    "olist_order_items_dataset.csv": "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    "olist_order_payments_dataset.csv": "4f713964f2815dbbaa40b9488268c55aac3627bfce5aa96cf58d1f3616de3cc0",
    "olist_order_reviews_dataset.csv": "0dff69f6fed33a13648020198ea94d7ae12afbdd4904186c6cd904e27a3e1ccd",
    "olist_orders_dataset.csv": "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    "olist_products_dataset.csv": "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
    "olist_sellers_dataset.csv": "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
    "product_category_name_translation.csv": "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278",
}


@dataclass(frozen=True)
class ProjectConfig:
    dataset_url: str = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "olistbr/brazilian-ecommerce?datasetVersionNumber=7"
    )
    dataset_version: int = 7
    archive_sha256: str = "d521eb1d4a8b6dae030aa429380787261d3b04cd95bee0f43f18cb9cb18ffebb"
    data_dir: Path = Path("data/olist_v7")
    archive_path: Path = Path("data/olist_brazilian_ecommerce_v7.zip")
    database_path: Path = Path("artifacts/ecommerce.duckdb")
    output_dir: Path = Path("artifacts")
    complete_month_start: str = "2017-01-01"
    complete_month_end: str = "2018-09-01"

    def validate(self) -> None:
        if self.dataset_version != 7:
            raise ValueError("This project is verified against Olist dataset version 7")
        if self.complete_month_start >= self.complete_month_end:
            raise ValueError("complete_month_start must be before complete_month_end")
