# Measured numbers behind the paper

Regenerate everything here with `./paper_numbers.sh` from the repository root
(or one section at a time: `render`, `resolution`, `hetero`, `rtx`, `grid`).
Each file starts with the commit, host and GPU it was produced on, so a figure
in the manuscript can be traced to a run rather than to someone's terminal.

| File | Feeds |
|---|---|
| `render_512.csv` | Full-frame render at 512², all four paths — the FRep Designer columns of the cross-system render table. `ms` is wall-clock per frame; `build_ms` and `run_ms` separate the one-time setup from the recurring dispatch. |
| `render_scaling.csv` | The same at 400/800/1280 — the CPU-vs-GPU table and the speed-up-by-resolution figure. |
| `hetero_tilequeue.csv` | CPU_IR and GPU_GLSL sharing one tile queue at 800² over 128-px cells: throughput and measured energy for each device alone (under the guided policy, its best in this harness) and composed under each queue policy — the plain queue (`hetero_plain`), the fastest-takes-the-tail rule (`hetero_tail`) and the speed-weighted guided chunking (`hetero_guided`, the default). Medians of five interleaved rounds; each run lasts at least 500 ms so NVML resolves it. |
| `rtx_instances.csv`, `rtx_instances_raw.txt` | GPU_RTX vs CPU_IR on a grid of N separable spheres at 1024², with NVML/RAPL energy — the instance-scaling table and figure. Two constructions of the same structure: `member` (one bottom-level structure and one intersection shader per union member, `FREP_RTX_NO_INSTANCING=1`) and `proto` (one per shared prototype, placed by instance transforms). Every point is three runs in separate processes with the driver's shader cache off; the raw file keeps all of them, the CSV their medians. The CPU column is from the same runs. |
| `grid_193.csv` | Raw SDF-evaluation throughput on the shared 193³ grid — the FRep Designer column of the cross-system grid table. |

The other systems (HyperFun, libfive, mpr) are measured by `run_all.sh`, which
writes `../results.csv`; those numbers are unaffected by changes to FRep
Designer's backends and are not regenerated here.

## Two things worth knowing before reading the numbers

**Separate build from run.** Every path pays a one-time cost (LLVM JIT, GLSL to
SPIR-V, NVPTX, or the RT pipeline plus acceleration structure) and a recurring
per-frame cost. Conflating them once made the ray-tracing path look an order of
magnitude slower than it is. `render_*.csv` keeps the two in separate columns.

**Whole-package energy drifts more.** `energy_xsys.csv` integrates the
package and the card around each system's binary for seconds without an idle
subtraction; four runs on two days moved every system's figure by a third or
more (FRep 103–158 J, libfive 193–398, HyperFun 33–39 kJ) with the times
nearly unchanged. Report it as a median with its range, not as one number,
and take it on a settled machine (the script sleeps before the floor).

**Throughput baselines drift.** GPU clock and thermal state move between runs,
so an isolated baseline can differ by tens of percent from one round to the
next. `hetero_tilequeue.csv` interleaves the three configurations and reports
medians for that reason; a combined figure slightly above the sum of separately
measured parts is that drift, not super-additivity.
