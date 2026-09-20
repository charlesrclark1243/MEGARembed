from Bio import Align
from Bio.Align import substitution_matrices


def setup_aligner() -> Align.PairwiseAligner:
    """
    Set up a pairwise aligner with specific scoring parameters.

    Returns:
        Align.PairwiseAligner: Configured pairwise aligner.
    """

    aligner: Align.PairwiseAligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    return aligner


def align_sequences(
    aligner: Align.PairwiseAligner, seq1: str, seq2: str
) -> Align.Alignment:
    """
    Align two sequences using the provided aligner.

    Args:
        aligner (Align.PairwiseAligner): The aligner to use for alignment.
        seq1 (str): The first sequence to align.
        seq2 (str): The second sequence to align.

    Returns:
        Align.Alignment: The resulting alignment object.
    """

    return aligner.align(seq1, seq2)[0]


def compute_score(aligner: Align.PairwiseAligner, aln: Align.Alignment) -> tuple[float]:
    """
    Compute the alignment score for a given alignment.

    Args:
        aligner (Align.PairwiseAligner): The aligner used for scoring.
        aln (Align.Alignment): The alignment object.

    Returns:
        float: The computed alignment score.
    """

    score: float = aln.score

    a, b = aln[0], aln[1]
    matches: int = sum(1 for x, y in zip(a, b) if x == y and x != "-")
    aln_length: int = len(a)

    pct_identity: float = 100.0 * matches / aln_length if aln_length > 0 else 0.0

    return score, pct_identity
