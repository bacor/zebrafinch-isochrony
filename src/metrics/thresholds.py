"""All thresholds used by the metrics, in one place."""

# A label or cluster is isolated when its nearest neighbour is further than
# this: Cohen's d for labels, the Mahalanobis distance for clusters.
ISOLATION_D = 3

# Clusters with fewer than this share of a bird's dyads are left out of the
# isochrony scores of its patterns.
MIN_CLUSTER_SHARE = 0.05

# A single dyad (x, y) is isochronous when |r - 0.5| < RHO (relative
# criterion, r = x / (x + y)) or |x - y| < TAU ms (absolute criterion).
RHO = 0.05
TAU = 20
