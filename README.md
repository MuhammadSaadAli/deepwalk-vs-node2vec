# DeepWalk with a gHypEG Configuration Model

A notebook that compares two random-walk node embedding methods on small
community-labelled graphs:

- **Enhanced DeepWalk**: random walks biased by the expected edge weights of a
  gHypEG configuration model.
- **Node2Vec**: second-order biased walks (`p = 1`, `q = 0.5`).

Both walk sets are embedded with Word2Vec (skip-gram) and evaluated on a binary
node classification task.

## Contents

- `Deepwalk_configuration_Unified.ipynb`: the full pipeline

## Datasets

Run **one** of the two loader cells in Step 1. Both produce the same variables
(`G`, `adj`, `labels`, `name_map`), so the rest of the notebook is unchanged.

| Dataset | Source | Labels |
|---|---|---|
| Zachary's Karate Club | `networkx.karate_club_graph()` | Club (`Mr. Hi` = 0, `Officer` = 1) |
| American College Football | Girvan & Newman (2002), downloaded from `www-personal.umich.edu/~mejn/netdata/football.zip` | Two largest conferences, relabelled 0 and 1 |

The football loader needs internet access. It sends a browser-like
`User-Agent`, because the server rejects Python's default one with a 403.

## Pipeline

1. **Load dataset**: builds the adjacency matrix and binary labels.
2. **gHypEG model**: computes the propensity matrix `Xi_ij = k_i * k_j`
   (diagonal set to 0) and the expected weights `E[X_ij] = m * Xi_ij / M`.
   The expected weights are then masked to observed edges, so walks stay on real
   edges. The degree-preservation property (Corollary 2) is checked numerically.
3. **Enhanced DeepWalk walks**: the next node is sampled in proportion to the
   masked `E[X_ij]`.
4. **Node2Vec walks**: return parameter `p` and in-out parameter `q` bias the
   second-order walk.
5. **Frequency analysis**: node visit counts and frequencies against degree.
6. **Embeddings**: Word2Vec skip-gram, 32 dimensions, window 5, 10 epochs.
7. **5-fold cross-validation**: stratified, logistic regression.
8. **Train/test split**: 70/30 stratified split with logistic regression,
   random forest and an RBF SVM.
9. **Prediction changes**: nodes where the two methods disagree.
10. **Plots**: accuracy and F1 comparison, confusion matrices, 2D PCA of the
    embeddings.

## Default parameters

| Parameter | Value |
|---|---|
| `NUM_WALKS` (walks per node) | 10 |
| `WALK_LENGTH` | 20 |
| `P`, `Q` (Node2Vec) | 1.0, 0.5 |
| `EMBEDDING_DIM` | 32 |
| `WINDOW` | 5 |
| `EPOCHS` | 10 |
| `TEST_SIZE` | 0.3 |
| `SEED`, `RANDOM_STATE` | 42 |

## Requirements

Python 3.8+ and:

```
numpy
networkx
gensim
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install numpy networkx gensim matplotlib seaborn scikit-learn
```

## Usage

1. Open the notebook in Jupyter or Google Colab.
2. Run the imports cell.
3. In Step 1, run **only one** dataset cell (Karate Club or Football).
4. Run the remaining cells in order.

To change the experiment, edit the parameters at the top of the walk,
embedding and classification cells (see the table above).

## Notes

- The football graph is reduced to its two largest conferences so that the
  binary classification setup applies. Node IDs are relabelled to `0..n-1`,
  and the team names are kept in `name_map`.
- Results are seeded, but Word2Vec with `workers=2` is not fully
  deterministic, so scores can vary slightly between runs.
- The masked expected weights are proportional to `k_i * k_j` on edges, so the
  Enhanced DeepWalk transition probability reduces to
  `k_j / sum of neighbour degrees`. It favours high-degree neighbours.

## References

- Girvan, M. and Newman, M. E. J. (2002). Community structure in social and
  biological networks. *PNAS*, 99(12).
- Perozzi, B., Al-Rfou, R. and Skiena, S. (2014). DeepWalk: Online learning of
  social representations. *KDD*.
- Grover, A. and Leskovec, J. (2016). node2vec: Scalable feature learning for
  networks. *KDD*.
- Zachary, W. W. (1977). An information flow model for conflict and fission in
  small groups. *Journal of Anthropological Research*, 33(4).
