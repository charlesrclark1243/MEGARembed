import re
from dataclasses import dataclass

from Bio.Seq import Seq

NONCODING_REGEX: re.Pattern = re.compile(
    r"16S|23S|rRNA|ribosomal_RNA|\brrs\b|\brrl\b|ribosomal_subunit", re.IGNORECASE
)

BACTERIAL_TABLE: int = 11
MIN_AA_LEN: int = 30  # anything shorter is almost certainly not a real ORF


@dataclass
class Translation:
    aa_seq: str
    status: str  # ok | noncoding_excluded | internal_stop | too_short | empty
    trimmed_bases: int
    internal_stops: int
    had_trailing_stop: bool


def is_noncoding(mechanism: str, group: str) -> bool:
    """
    Determine if a given mechanism and group indicate a noncoding sequence.

    Args:
        mechanism (str): The mechanism string to check.
        group (str): The group string to check.

    Returns:
        bool: True if the sequence is noncoding, False otherwise.
    """

    return bool(NONCODING_REGEX.search(f"{mechanism} {group}"))


def translate_cds(nt: str) -> Translation:
    """
    Translate a nucleotide sequence into an amino acid sequence, handling internal stops and trimming.

    Args:
        nt (str): The nucleotide sequence to translate.

    Returns:
        Translation: The translated amino acid sequence and associated information.
    """

    seq: Seq = Seq(nt.strip().upper())
    trimmed: int = len(seq) % 3
    if trimmed:
        seq = seq[: len(seq) - trimmed]

    if len(seq) == 0:
        return Translation(
            aa_seq="",
            status="empty",
            trimmed_bases=trimmed,
            internal_stops=0,
            had_trailing_stop=False,
        )

    aa: str = str(seq.translate(table=BACTERIAL_TABLE))

    had_trailing_stop: bool = aa.endswith("*")
    core: str = aa[:-1] if had_trailing_stop else aa
    internal_stops: int = core.count("*")

    if internal_stops:
        return Translation(
            aa_seq=core.split("*")[0],
            status="internal_stop",
            trimmed_bases=trimmed,
            internal_stops=internal_stops,
            had_trailing_stop=had_trailing_stop,
        )

    if len(core) < MIN_AA_LEN:
        return Translation(
            aa_seq=core,
            status="too_short",
            trimmed_bases=trimmed,
            internal_stops=0,
            had_trailing_stop=had_trailing_stop,
        )

    return Translation(
        aa_seq=core,
        status="ok",
        trimmed_bases=trimmed,
        internal_stops=0,
        had_trailing_stop=had_trailing_stop,
    )


def classify(nt: str, mechanism: str, group: str) -> Translation:
    """
    Classify a nucleotide sequence based on its translation and noncoding status.

    Args:
        nt (str): The nucleotide sequence to classify.
        mechanism (str): The mechanism string to check for noncoding status.
        group (str): The group string to check for noncoding status.

    Returns:
        Translation: The translated amino acid sequence and associated information.
    """

    if is_noncoding(mechanism, group):
        return Translation(
            aa_seq="",
            status="noncoding_excluded",
            trimmed_bases=0,
            internal_stops=0,
            had_trailing_stop=False,
        )

    return translate_cds(nt)
