# PR #1374 benchmark results

[Read the full arbitrary-shape benchmark report](flydsl-comprehensive-20261116/REPORT.md), including all per-shape results, regressions, raw timing archives, and reproduction commands.

The three-pass sweep covers 1,842 unique shapes on two Radeon AI PRO R9700 GPUs. It measures FlyDSL `3ed2ca5`, before the later E5M2-only cleanup. The results branch preserves that source revision and adds only benchmark documentation and artifacts; it does not alter the PR branch.

The current PR uses E5M2-folded weights. See the [paired E4M3/E5M2 format comparison](flydsl-weight-format-20261116/REPORT.md) for the measurements supporting that change. The full-sweep numbers remain associated with their original E4M3-folded revision.
