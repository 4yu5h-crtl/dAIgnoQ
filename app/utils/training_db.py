import sqlite3
from pathlib import Path
from typing import Dict, Optional

from dAIgnoQ.app import config


DB_PATH = config.DATA_DIR / "training_runs.db"


def init_training_db(db_path: Optional[Path] = None) -> Path:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_name TEXT,
                model_family TEXT,
                dataset_size INTEGER,
                synthetic_images INTEGER,
                augmentation_applied INTEGER,
                train_metric REAL,
                val_metric REAL,
                checkpoint_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS generated_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_family TEXT,
                checkpoint_path TEXT,
                count_generated INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return path


def insert_training_run(
    model_family: str,
    dataset_size: int,
    checkpoint_path: str,
    train_metric: Optional[float] = None,
    val_metric: Optional[float] = None,
    synthetic_images: int = 0,
    augmentation_applied: bool = False,
    disease_name: Optional[str] = None,
):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO training_runs
            (disease_name, model_family, dataset_size, synthetic_images, augmentation_applied, train_metric, val_metric, checkpoint_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                disease_name or config.CURRENT_DISEASE,
                model_family,
                dataset_size,
                synthetic_images,
                1 if augmentation_applied else 0,
                train_metric,
                val_metric,
                checkpoint_path,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def insert_generation_run(model_family: str, checkpoint_path: str, count_generated: int):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO generated_samples (model_family, checkpoint_path, count_generated)
            VALUES (?, ?, ?)
            """,
            (model_family, checkpoint_path, count_generated),
        )
        conn.commit()
    finally:
        conn.close()
