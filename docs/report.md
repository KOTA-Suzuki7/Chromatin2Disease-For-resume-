# Methods and preliminary results

## Problem and motivation

The project classifies 3D chromosome traces from untreated versus acutely
CTCF-depleted mouse embryonic stem cells. CTCF helps organize loop anchors and
multiway chromatin hubs, but individual chromosome conformations are highly
variable. The central modeling decision is therefore to classify a set of ten
same-condition traces rather than require one noisy trace to determine the
condition alone.

The repository focuses on interpretability. The model uses one biological
contact-message-passing round and a final additive readout in which every
locus has an exact signed contribution to the graph logit.

## Dataset and processing

The source is the Hafner et al. ORCA dataset deposited in the 4D Nucleome Data
Portal. Six FOF-CT core files contain three experimental run groups for each
condition. Each row supplies a trace identifier, genomic interval, and measured
3D coordinate for one detected locus in the mouse chromosome 6 Hoxa region.

The canonical panel contains 73 loci. Following the paper's 50% coverage
criterion, traces with fewer than 37 observed loci are excluded. The retained
dataset contains 11,488 variable-size graphs: 5,283 untreated and 6,205
CTCF-depleted.

Two nonadjacent observed loci receive an undirected edge when their Euclidean
distance is below the paper's 238-nm chromosome 6 contact threshold. The
processed dataset contains 2,035,993 undirected edges, with a mean of 177.2
edges per graph.

Each node has a 73-way locus identity, a binary indicator for the paper's five
chromosome 6 CTCF-binding sites, normalized contact degree, and normalized
radial position. Missing loci are omitted. No edge features are used.

Traces are randomly split 70/15/15, stratified by condition and experimental
run group, before bags are constructed. Each bag contains ten traces with the
same condition and run group. Bags are balanced across the two classes within
each run group. The fixed test partition contains 144 bags and 1,440 graphs.

## Model and loss

For node `i`, one GraphConv layer performs separate linear transformations of
the node and the sum of its contact neighbors, followed by ELU and 10% dropout:

\[
h_i=\operatorname{ELU}(W_{self}x_i+W_{neighbor}\sum_{j\in\mathcal N(i)}x_j+b).
\]

The hidden dimension is eight. Final attention is normalized within each graph:

\[
\alpha_i=\frac{\exp(q^Th_i+b_a)}{\sum_{k\in V_g}\exp(q^Th_k+b_a)}.
\]

With signed node score `s_i=w_s^Th_i`, the graph and bag logits are

\[
z_g=b_g+\sum_i\alpha_i s_i,
\qquad
z_B=\frac{1}{10}\sum_{g\in B}z_g.
\]

For a minibatch of `M` bags, binary cross-entropy with logits is

\[
\mathcal L=-\frac{1}{M}\sum_{m=1}^{M}
[y_m\log\sigma(z_m)+(1-y_m)\log(1-\sigma(z_m))].
\]

Adam uses learning rate 0.003 and weight decay 0.0001. The checkpoint with the
highest validation ROC-AUC is selected, with early stopping after 12 epochs
without improvement.

## Preliminary results

Across three initialization seeds, individual-graph test ROC-AUC was
0.637 ± 0.002. Averaging ten graph logits raised bag-level ROC-AUC to
0.828 ± 0.007. Bag-level average precision was 0.831 ± 0.006 and accuracy was
0.722 ± 0.014. The improvement supports the hypothesis that CTCF depletion
causes subtle but recurrent structural changes that become clearer across
multiple chromosome traces.

The learned graph logit was reconstructed exactly from graph bias plus signed
node contributions in all three runs. Several loci were emphasized consistently,
but the strongest-attention locus also has a condition-associated observation
rate difference. Consequently, the current interpretation and performance
must be treated as preliminary until missingness is controlled.

## Limitations and next checks

1. Evaluate whether performance persists after matching or masking loci with
   condition-associated missingness.
2. Use run-group-held-out evaluation to assess generalization across
   experimental batches.
3. Group traces by biological cell if reliable cell identifiers become
   available.
4. Compare learned signed contributions with the paper's reported CTCF-binding
   site hubs without treating supplied CBS annotations as independent
   rediscovery.
