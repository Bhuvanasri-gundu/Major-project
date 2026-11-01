# Dataset Directory: mHealth Literature Corpus

This directory holds the raw biomedical literature extraction files downloaded from the National Center for Biotechnology Information (NCBI) PubMed repository.

---

## Primary Dataset File

- **`pubmed_mhealth_2015_2025_proposed_final.xls`**:
  - **Format**: CSV format stored with `.xls` extension for Excel interoperability.
  - **Date Range**: January 1, 2015 – December 31, 2025.
  - **Total Raw Records**: 14,905 PubMed records.
  - **Screened Relevant Records**: 12,642 peer-reviewed articles meeting strict mobile health criteria.
  - **Fields**:
    - `pmid`: Unique PubMed identifier.
    - `title`: Article title.
    - `abstract`: Full unstructured publication abstract.
    - `journal`: Publishing scientific journal name.
    - `publication_year`: Year of publication.
    - `doi`: Digital Object Identifier.
    - `mesh_terms`: Medical Subject Headings indexed by the National Library of Medicine.

---

## Data Governance & Reproducibility

- The raw file is strictly read-only and preserved as the immutable baseline.
- Data validation, cleaning, and preprocessing steps are fully reproducible via:
  - Notebook 01: `notebooks/01_pubmed_data_extraction_and_cleaning.ipynb`
  - Notebook 02: `notebooks/02_data_validation.ipynb`
  - Python Module: `src/data_validation.py`
  - Cleaned outputs are persisted to `outputs/mhealth_candidate_cleaned.csv` and `outputs/mhealth_screened_corpus.csv`.
