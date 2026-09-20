from __future__ import annotations

import pandas as pd
from Bio import SeqIO

from mee.common import ANNOT, FASTA, OUT
from mee.translate import Translation, classify


def parse_fasta() -> pd.DataFrame:
    """
    Parse the FASTA file and return a DataFrame with the relevant information.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed information from the FASTA file.
    """

    rows: list[dict[str, any]] = []
    for record in SeqIO.parse(str(FASTA), "fasta"):
        parts: list = record.description.split("|")
        if len(parts) < 5:
            raise ValueError(f"Unexpected FASTA header format: {record.description}")

        rows.append(
            {
                "header": record.description,
                "meg_id": parts[0],
                "type": parts[1],
                "class": parts[2],
                "mechanism": parts[3],
                "group": parts[4],
                "requires_snp": len(parts) >= 6
                and parts[5] == "RequiresSNPConfirmation",
                "nt_seq": str(record.seq),
                "nt_len": len(record.seq),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """
    Main function to parse the FASTA file, classify the sequences, and save the results to a CSV file.
    """

    OUT.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame = parse_fasta()
    print(f"Parsed {len(df)} FASTA records")
    assert len(df) == 11506, f"Expected 11506 records, but got {len(df)}."

    annotations: pd.DataFrame = pd.read_csv(ANNOT)
    annotations = annotations.rename(columns={"Can be confirmed": "can_be_confirmed"})
    annotations = annotations.drop(
        columns=["type", "class", "mechanism", "group"]
    )  # already in header

    df = df.merge(annotations, on="header", how="left", validate="one_to_one")

    missing: int = df["Cluster_ID"].isna().sum()
    if missing:
        print(f"Warning: {missing} records had no annotation-CSV match.")

    results: list[Translation] = [
        classify(row.nt_seq, row.mechanism, row.group) for row in df.itertuples()
    ]

    df["aa_seq"] = [translation.aa_seq for translation in results]
    df["translation_status"] = [translation.status for translation in results]
    df["trimmed_bases"] = [translation.trimmed_bases for translation in results]
    df["internal_stops"] = [translation.internal_stops for translation in results]
    df["had_trailing_stop"] = [translation.had_trailing_stop for translation in results]
    df["aa_len"] = df["aa_seq"].str.len()

    print("\ntranslation status:")
    print(df["translation_status"].value_counts().to_string())
    print(f"\nrequires_snp: {df['requires_snp'].sum()}")

    df.to_parquet(OUT / "megares_tidy.parquet", index=False)

    # embeddable subset
    proteins: pd.DataFrame = df[df["translation_status"] == "ok"].reset_index(drop=True)
    proteins.to_parquet(OUT / "megares_proteins.parquet", index=False)
    print(f"\nembeddable subset: {len(proteins)}")

    excluded: pd.DataFrame = df[df["translation_status"] != "ok"]
    excluded[
        [
            "meg_id",
            "type",
            "class",
            "mechanism",
            "group",
            "translation_status",
            "internal_stops",
            "nt_len",
        ]
    ].to_csv(OUT / "excluded.csv", index=False)
    print(f"excluded subset: {len(excluded)}")

    print("\naa length distribution:")
    print(proteins["aa_len"].describe().to_string())
    over: int = (proteins["aa_len"] > 1022).sum()
    print(f"sequences over 1022 aa (will be truncated by ESM-2): {over}")


if __name__ == "__main__":
    main()
