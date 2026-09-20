from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

V4: Path = ROOT / "data" / "raw" / "megares_db_maintenance" / "database_files" / "v4"
FASTA: Path = V4 / "megares_database_v4.00.fasta"
ANNOT: Path = V4 / "megares_annotations_with_clusters_v4.00.csv"
OUT: Path = ROOT / "data" / "processed"

ARTIFACTS: Path = ROOT / "artifacts"
