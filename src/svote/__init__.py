"""SVote — Support-Sentence Voting for self-consistency.

Stage 1 (CPU): weighted voting on support sentences from N ballots, with
  bridging constraints to favor multi-paragraph evidence.
Stage 2 (LLM): re-infer the answer on a context filtered by the voted
  support set.

This differs from existing SC variants (SC / CISC / RVSC / RASC / JESC) which
all vote on the *output* (answer or rank). SVote votes on the *input* — the
context filter — which is a previously unutilized signal axis.
"""
from .voting import vote_supports, select_supports  # noqa: F401
