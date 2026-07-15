#!/usr/bin/env python3
"""
select_cluster_representatives.py
=================================
Principled, training-data-only channel reduction via correlation clustering.

METHOD (defensible in a thesis, no test labels involved anywhere):
  1. Compute pairwise Pearson |r| between channels on NORMAL TRAINING data
     (already done: channel_correlation_analysis.py -> channel_correlation_matrix.csv).
  2. Hierarchical agglomerative clustering (average linkage) on the distance
     d(i,j) = 1 - |r(i,j)|. Clustering on |r| (not r) means strongly
     ANTI-correlated channels (e.g. ch61, r=-0.98 with Cluster A) are
     automatically recruited into their functional cluster.
  3. Cut the dendrogram at a distance threshold t (default 0.30, i.e. channels
     with average |r| >= 0.70 are considered redundant duplicates).
  4. From each cluster of size >= min_size, select:
       - the MEDOID: the channel with maximum mean |r| to its cluster members
         (the most central, most representative channel), and optionally
       - the COMPLEMENT: the member with minimum |r| to the medoid
         (preserves intra-cluster relationship information, which matters for
         relationship-breakdown anomaly detection, at the cost of 1 channel).
  5. Emit ready-to-paste RUN_SPECS.

This replaces brute-force subset sweeps with a single-parameter method
(the dendrogram cut t), whose sensitivity can be reported as t in {0.2,0.3,0.4}.

Related literature to cite: feature agglomeration / correlation-based feature
clustering (e.g. scikit-learn FeatureAgglomeration), minimum-redundancy
feature selection (mRMR, Peng et al. 2005), and cluster-medoid selection.

Usage:
    python select_cluster_representatives.py
    python select_cluster_representatives.py --corr_csv results_longrun/channel_correlation/channel_correlation_matrix.csv
    python select_cluster_representatives.py --cut 0.30 --min_size 3 --per_cluster 2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_csv",
                    default=("results_longrun/channel_correlation/"
                             "channel_correlation_matrix.csv"))
    ap.add_argument("--cut", type=float, default=0.30,
                    help="dendrogram cut distance t; channels with avg |r| >= 1-t merge")
    ap.add_argument("--min_size", type=int, default=3,
                    help="ignore clusters smaller than this (isolated/noise channels)")
    ap.add_argument("--per_cluster", type=int, default=2, choices=[1, 2],
                    help="1 = medoid only, 2 = medoid + least-correlated complement")
    ap.add_argument("--out_csv", default="cluster_representatives.csv")
    args = ap.parse_args()

    corr = pd.read_csv(args.corr_csv, index_col=0)
    corr = corr.loc[corr.index, corr.index]          # ensure square & aligned
    A = corr.abs().to_numpy()
    np.fill_diagonal(A, 1.0)
    A = np.clip((A + A.T) / 2.0, 0.0, 1.0)           # symmetrise, clip fp noise

    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=args.cut, criterion="distance")

    chans = corr.index.tolist()
    rows, run_channels = [], {"medoid": [], "pair": []}

    for cl in sorted(set(labels)):
        members_idx = [i for i, l in enumerate(labels) if l == cl]
        members = [chans[i] for i in members_idx]
        size = len(members)

        sub = A[np.ix_(members_idx, members_idx)]
        # medoid: max mean |r| to other members (or itself if singleton)
        if size == 1:
            medoid, complement, cohesion = members[0], None, 1.0
        else:
            mean_r = (sub.sum(axis=1) - 1.0) / (size - 1)
            mi = int(np.argmax(mean_r))
            medoid = members[mi]
            cohesion = float(mean_r[mi])
            ci = int(np.argmin(sub[mi]))
            complement = members[ci]

        keep = size >= args.min_size
        rows.append({"cluster": cl, "size": size,
                     "medoid": medoid, "medoid_mean_abs_r": round(cohesion, 3),
                     "complement": complement,
                     "medoid_complement_abs_r":
                         (round(float(A[members_idx[mi], members_idx[ci]]), 3)
                          if size > 1 else None),
                     "kept": keep,
                     "members": ",".join(members)})
        if keep:
            run_channels["medoid"].append(medoid)
            run_channels["pair"].append(medoid)
            if args.per_cluster == 2 and complement and complement != medoid:
                run_channels["pair"].append(complement)

    df = pd.DataFrame(rows).sort_values(["kept", "size"],
                                        ascending=[False, False])
    print(df.to_string(index=False))
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")

    def to_nums(cols):
        return sorted(int(c.split("_")[-1].replace("ch", "")) for c in cols)

    med = to_nums(run_channels["medoid"])
    pai = to_nums(set(run_channels["pair"]))
    print(f"\ncut t={args.cut}  (channels with avg |r| >= {1-args.cut:.2f} merged)")
    print(f"clusters kept (size >= {args.min_size}): {int(df.kept.sum())}")
    print(f"\nmedoids only          ({len(med)} ch): {med}")
    print(f"medoid + complement   ({len(pai)} ch): {pai}")

    cut_tag = str(args.cut).replace('.', '')
    print("\nAdd to RUN_SPECS:")
    print(f'    {{"run": "clust_med_t{cut_tag}_{len(med)}ch",')
    print(f'     "features": ch_list({med}),')
    print(f'     "group": "cluster_representatives"}},')
    print(f'    {{"run": "clust_pair_t{cut_tag}_{len(pai)}ch",')
    print(f'     "features": ch_list({pai}),')
    print(f'     "group": "cluster_representatives"}},')


if __name__ == "__main__":
    main()
