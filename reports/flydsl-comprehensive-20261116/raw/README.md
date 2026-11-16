# Raw benchmark records

The archives retain every native/vLLM sample, status, command ledger, and log from both GPUs. Files are unchanged from the measured runs.

- [GPU 0](gpu0.tar.gz): 2,054 files.
- [GPU 1](gpu1.tar.gz): 2,059 files.

Extract from the parent report directory using `tar -xzf raw/gpu0.tar.gz` and `tar -xzf raw/gpu1.tar.gz`. This restores the original paths used by the aggregators. All six published CSV/JSON result files were recomputed from these archives and matched byte-for-byte before publication. See [reproduction commands](../REPRODUCE.md).
