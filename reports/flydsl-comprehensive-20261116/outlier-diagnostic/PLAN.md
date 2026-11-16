# Supplemental outlier diagnostic

Run only after the full native and GPU0 vLLM measurements finish. This is separate from all main sweep aggregates. No production kernel or selector edits.

Use the unchanged native benchmark with `--gpu 0 --m 1 8 --nk 528x8192 1040x4096 --native-fp8 --cache-policy auto --configs configs.json`, three independent processes/passes, and the same source runtime/ROCm/AMD SMI environment as the native sweep. Save every pass JSON, log, command, and exit status. Each pass should have 12 rows: four shapes × ALU, LUT, native FP8.

Question: does explicitly switching the register decoder between ALU and LUT explain the stable M8 regressions? The normal M8 selector uses LUT for N528/K8192 and ALU for N1040/K4096. Keep default selection unchanged. The baseline data and diagnostic data must not be mixed. This experiment can assess the decoder choice; it cannot establish an ISA-level cause of residual M1/M8 scaling.
