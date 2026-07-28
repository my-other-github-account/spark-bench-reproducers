# Kasa and wedge doctrine

Kasa is recovery-only, never a scheduler. Use a configured power client only after proving that the target node is the failed identity, that no healthy owner or storage appliance shares the selector, and that the action cannot interrupt an accepted artifact write. Credentials and private selectors do not belong in receipts.

A wedge is a stalled worker with no advancing receipt, not merely a long kernel. Before takeover: (1) compare two progress snapshots, (2) prove the expected process/GPU owner is absent or non-advancing, (3) seal the newest valid checkpoint and binding hashes, and (4) establish a single-writer lease. Resume from the newest binding-matched checkpoint. Refuse a binding mismatch. Keep two rolling checkpoints. A checkpoint write failure is log-and-continue; an artifact-binding failure is fail-closed.

After recovery or takeover, verify hostname/identity independently, run the first-scored-layer gate, require monotonic progress, and release the node only after process/GPU emptiness plus terminal readback. Never power-cycle the control node, storage, or fabric casually.
