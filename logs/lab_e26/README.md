# E26 lab run (NF4 half; 2 x RTX A4000)

Driver output of the three runs on the shared lab server, in order:

- `run_attempt1.log` (`launch/e26_lab_run.sh`, 2026-10-30 22:43): the PyTorch install timed out; uv's HTTP
  timeout was raised to 600 s.
- `run.log` (`launch/e26_lab_run.sh`, 23:44): setup, then NF4 and the 16-bit reference ran out of GPU memory
  part-way (6 NF4 seeds and 1 bf16 seed finished).
- `resume.log` (`launch/e26_lab_resume.sh`, 00:53 → 01:26): one process per seed, finished seeds skipped; all 10
  seeds of both precisions done.

The outputs are in `runs/E26/a4000/`. The per-process logs stayed on the lab server.
