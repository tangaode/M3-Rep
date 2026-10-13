# Input bundle

M3-Rep reads one compressed NumPy archive. The archive contains the two ID arrays, three drug-by-disease label matrices and the feature matrices used by the model.

Required arrays:

- `drug_ids`, `disease_ids`
- `train_labels`, `all_labels`, `test_labels`
- `kernel__disease_gene`, `kernel__disease_phenotype`, `kernel__disease_name`, `kernel__disease_ontology`
- `kernel__disease_definition`, `kernel__disease_clinical`, `kernel__drug_gene`, `kernel__drug_chem`
- `kernel__disease_llm`, `kernel__disease_causal`, `kernel__disease_causal_tfidf`
- `static__direct_target`, `static__disease_gene_count`, `static__disease_phenotype_count`
- `static__max_gene_neighbor`, `static__max_name_neighbor`
- `static__definition_available`, `static__clinical_available`, `static__physchem_available`

Label matrices use shape `(number_of_drugs, number_of_diseases)`. `train_labels` contains the treatment records used to construct training features. `all_labels` contains every known treatment record and is used only to prevent known positives from being sampled as negatives. `test_labels` contains the held-out records used for evaluation.

The kernel matrices are non-negative, square, and aligned to the corresponding drug or disease ID array. Mechanism kernels are supplied as precomputed similarities; the repository does not store API credentials or external raw databases.
