# MEGARes Embedding Explorer

An interactive explorer for the [MEGARes 4.0](https://www.meglab.org/megares/) antimicrobial
resistance gene database in protein language model space. Paste a gene, see its nearest neighbours
by ESM-2 embedding similarity, find it on a 2D map of the whole database, and align it against any
neighbour — all locally, with sub-millisecond search.

> **Exploratory research and education tool — not clinical or diagnostic.**

---

## What it does

- **Semantic search.** Paste a protein *or* nucleotide sequence. Nucleotide input is detected and
  translated automatically. Returns the nearest MEGARes genes by cosine similarity over ESM-2
  embeddings, with their full resistance ontology.
- **Interactive map.** All 10,814 indexed proteins projected to 2D with UMAP, coloured by any
  ontology level. Your query is placed on the same map. Box- or lasso-select regions to inspect them.
- **Click to inspect.** Select any gene on the map to see its annotation, sequence, and *its* nearest
  neighbours — no model call needed, since its embedding is already indexed.
- **Pairwise alignment.** BLOSUM62 alignment of your query against any neighbour, with percent
  identity and score.
- **Resistance-flag awareness.** Genes marked `RequiresSNPConfirmation` confer resistance only via a
  specific variant, never by presence. They are carried end to end and badged everywhere they appear.

## How it works

```
MEGARes v4 FASTA          11,506 nucleotide gene entries
                                  |
  01_parse_translate.py    translate (bacterial code, table 11), exclude rRNA
                           and internal-stop entries, join ontology + clusters
                                  v
                           10,814 clean protein sequences
                                  |
  02_embed.py              ESM-2 mean-pooled embeddings (fp16, length-sorted
                           batching, OOM backoff)          -> embeddings_650M.npy
                                  v
  03_build_index.py        L2-normalize + IndexFlatIP      -> megares_650M.faiss
  04_umap.py               UMAP (cosine, seed 42)          -> umap_coords + reducer
                                  v
  app/main.py              Streamlit: loads artifacts only, never recomputes
```

Everything expensive happens offline. The app only *loads* precomputed artifacts, so cold start is
a few seconds and a query is a single forward pass plus a sub-millisecond index lookup.

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU (see
[Limitations](#limitations)).

```bash
# 1. dependencies
uv sync

# 2. verify the GPU (checks compute capability and runs a real matmul)
uv run scripts/gpu_check.py

# 3. get the data: download the MEGARes v4.0.0 archive and extract it so that
#    data/raw/megares_db_maintenance/database_files/v4/ exists.
#    Use the pinned release, not a git clone of main — the checksums in
#    data/checksums.txt refer to the v4.0.0 files.

# 4. build the artifacts (~10 minutes total on an RTX 5080)
uv run scripts/01_parse_translate.py
uv run scripts/02_embed.py       -s 650M
uv run scripts/03_build_index.py -s 650M
uv run scripts/04_umap.py        -s 650M

# 5. run it
uv run streamlit run app/main.py --server.address localhost
```

Use `-s 150M` throughout for a faster, lighter pass (640-dim embeddings, ~1 minute to embed).

There is also a CLI:

```bash
uv run scripts/query.py -s 650M -k 10 -q MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDKLGARVGYIELDLNSG
```

## Project layout

```
src/mee/            reusable logic, shared by the CLI and the app
  common.py           paths + artifact loading (bottom of the import stack)
  translate.py        nucleotide -> protein with explicit exclusion rules
  embed.py            ESM-2 loading and mask-aware mean pooling
  search.py           query preparation, FAISS search, ontology join
  project.py          UMAP coords, reducer, query projection
  alignment.py        Biopython pairwise alignment
scripts/            one numbered script per pipeline stage, plus query.py
app/main.py         the Streamlit UI
```

Scripts stay thin — argument parsing, I/O, progress. All logic lives in `src/mee/` so the CLI and
the app share one implementation and cannot drift apart.

## Design decisions worth noting

- **Translation is explicit and logged.** MEGARes ships nucleotide sequences only. 235 rRNA-based
  determinants are excluded (no protein product, so a protein language model is meaningless for
  them) and 457 entries with internal stop codons are excluded as likely frame/strand issues. Every
  exclusion is written to `data/processed/excluded.csv` with a reason.
- **Pooling excludes special tokens.** Attention-mask-aware mean over residue tokens only, with
  BOS/EOS/pad zeroed. Verified by checking that a sequence embedded alone matches the same sequence
  embedded inside a padded batch (agreement to 4e-4, fp16 noise).
- **Length-sorted batching with a token budget.** Protein lengths range 40–2,890 residues. Sorting
  by length and capping batches by padded tokens cuts wasted compute by ~63% versus fixed batches,
  with an OOM backoff that halves and retries for the long tail.
- **Row alignment is a hard invariant.** Row *i* of the embedding matrix is row *i* of the protein
  table is row *i* of the UMAP coordinates. Every stage asserts it, because a silent off-by-one here
  returns confidently wrong genes.
- **Artifacts are keyed by model size,** so 150M and 650M results never collide and a query cannot
  be run against an index built by a different model.

## Limitations

- **The map is for browsing, not ranking.** UMAP 2D distances are not metric, and projecting a new
  query into a frozen manifold distorts. FAISS over the full embeddings is the source of truth.
  (Measured: 10-NN same-class agreement is 88.5% in 2D versus 90.3% in the full 1280-d space.)
- **This is not a better gene identifier than alignment.** Benchmarked against MMseqs2 under a
  homology-aware, cluster-disjoint protocol, sequence alignment recovered MEGARes ontology labels
  more accurately at every level. What embeddings uniquely provide here is a continuous, browsable
  map and fast lookup — not accuracy. Use BLAST or MMseqs2 to identify a gene.
- **A GPU is currently required.** The device defaults to `cuda` in the app. The CLI accepts
  `--device cpu`, and CPU inference is usable for single queries with the 150M model.
- **Not clinical.** Nothing here should inform diagnosis or treatment.

## Data and licensing

MEGARes 4.0 is published by the
[Microbial Ecology Group](https://github.com/Microbial-Ecology-Group/MEGARes_db_maintenance).
SHA-256 checksums of the exact source files used are recorded in `data/checksums.txt`.

**Licensing is unsettled, so read this before redistributing anything derived from it.** The source
distribution ships a GPL-3.0 `LICENSE` file, but its own README lists the license as
"[LICENSE TO BE ADDED]" and states that "an aggregated resource cannot be licensed more permissively
than its most restrictive source allows." MEGARes v4 aggregates CARD v4.0.0, ResFinder 4.7.1,
PointFinder 4.7.1, NCBI AMRFinderPlus 4.0 and BacMet 2.0, each with its own terms.

This repository redistributes none of it: `data/` and `artifacts/` are gitignored and built locally
from your own copy of the database. Only `data/checksums.txt` is tracked. Anyone publishing the
derived artifacts — embeddings, translated sequences, a prebuilt container — should check each
source database's terms first.

The database contains 11,506 accessions across 4 types, 63 classes, 275 mechanisms and 1,609 gene
groups; 10,814 translate cleanly to protein and are indexed here, of which 552 carry the
`RequiresSNPConfirmation` flag.

## Stack

Python 3.12 · PyTorch 2.14 (cu130) · transformers 5.17 · ESM-2 (`facebook/esm2_t33_650M_UR50D`) ·
faiss-cpu 1.15 · umap-learn 0.5.12 · Biopython 1.88 · Streamlit 1.64 · Plotly · uv
