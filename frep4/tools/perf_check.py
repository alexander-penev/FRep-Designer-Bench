#!/usr/bin/env python3
"""
perf_check.py — performance regression detector for FRep Designer.

Compares the JSON output of `frep_bench --json` against the committed
baseline in `tools/perf_baseline.json` and fails (exit 1) if any
measurement is more than `--threshold` × the baseline value.

Typical CI use:

    ./build/frep_bench --json > /tmp/current.json
    python3 tools/perf_check.py /tmp/current.json

For environments where the GPU path is significantly noisier than the
CPU path (e.g. software Vulkan via Mesa llvmpipe in GitHub-hosted CI
runners), use per-kind thresholds:

    python3 tools/perf_check.py /tmp/current.json \\
        --cpu-threshold 2.0 --gpu-threshold 3.0

Sample output:

    Scene                                       Current   Baseline   Ratio
    ────────────────────────────────────────────────────────────────────────
    CPU Simple (1 sphere) @ 400x300              73 ms      71 ms    1.03×  OK
    CPU Heavy (MeshSDF + CSG) @ 800x600         186 ms     180 ms    1.03×  OK
    ...
    GPU Simple (1 sphere) @ 400x300              19 ms      17 ms    1.12×  OK
    ...
    ✓ No regressions detected (CPU 2.0×, GPU 3.0×)

If a row exceeds the threshold, that line is flagged with ✗ and the
script exits 1. CI workflows can wire that into a required-check.

Why no scipy/numpy: this script must run on a stock Ubuntu image
without `pip install` — single-file stdlib only.

Updating the baseline: after intentional performance changes, regenerate
the file with:

    ./build/frep_bench --json > tools/perf_baseline.json
    git add tools/perf_baseline.json
    git commit -m "perf: update baseline after <reason>"

Or invoke the CMake convenience target:

    cmake --build build --target perf-update-baseline
"""

import argparse
import json
import sys
from pathlib import Path


def load_run(path: Path) -> dict:
    """Read a JSON file produced by `frep_bench --json`. We accept partial
    inputs gracefully — older baselines may lack some scenes; missing keys
    just skip the corresponding row in the report."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_rows(rows: list[dict]) -> dict[tuple[str, str], float]:
    """Convert [{scene, size, ms}, ...] to {(scene, size): ms}."""
    return {(r["scene"], r["size"]): r["ms"] for r in rows}


def compare(current: dict, baseline: dict,
            thresholds: dict[str, float]) -> bool:
    """Print a report. Returns True if all measurements are within their
    respective per-kind thresholds, False otherwise.

    `thresholds` maps "cpu" / "gpu" -> ratio. This lets noisier categories
    (e.g. software Vulkan under llvmpipe in CI) use a more lenient ratio
    than the well-behaved JIT path."""
    all_ok = True

    print(f"{'Scene':<48}{'Current':>10}{'Baseline':>12}{'Ratio':>10}  Status")
    print("─" * 92)

    for kind in ("cpu", "gpu"):
        cur_rows = index_rows(current.get(kind, []))
        base_rows = index_rows(baseline.get(kind, []))
        threshold = thresholds[kind]

        for key, ms_cur in sorted(cur_rows.items()):
            scene, size = key
            ms_base = base_rows.get(key)
            label = f"{kind.upper()} {scene} @ {size}"

            if ms_base is None:
                # New scene that wasn't in baseline — informational only,
                # don't fail.
                print(f"{label:<48}{ms_cur:>7.0f} ms{'(new)':>12}{' --':>10}  NEW")
                continue

            # Guard against tiny times where ratios are noisy: anything
            # under 5 ms is exempted from the threshold check (noise floor
            # of the timer).
            if ms_base < 5.0:
                ratio_str = "noise"
                status = "OK"
            else:
                ratio = ms_cur / ms_base
                ratio_str = f"{ratio:.2f}×"
                if ratio > threshold:
                    status = f"✗ REGRESS (>{threshold}×)"
                    all_ok = False
                else:
                    status = "OK"

            print(f"{label:<48}{ms_cur:>7.0f} ms{ms_base:>9.0f} ms"
                  f"{ratio_str:>10}  {status}")

    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("current", type=Path,
        help="JSON file from `frep_bench --json`")
    ap.add_argument("--baseline", type=Path,
        default=Path(__file__).with_name("perf_baseline.json"),
        help="Baseline JSON to compare against "
             "(default: tools/perf_baseline.json)")
    ap.add_argument("--threshold", type=float, default=2.0,
        help="Default maximum allowed ratio current/baseline (default: 2.0). "
             "Used for both CPU and GPU rows unless --cpu-threshold or "
             "--gpu-threshold override.")
    ap.add_argument("--cpu-threshold", type=float, default=None,
        help="Override threshold for CPU rows. Use when the GPU path is "
             "noisier than the CPU path (e.g. software Vulkan under "
             "llvmpipe in CI).")
    ap.add_argument("--gpu-threshold", type=float, default=None,
        help="Override threshold for GPU rows. llvmpipe variance in "
             "GitHub-hosted CI runners typically warrants ~3.0 here.")
    args = ap.parse_args()

    if not args.current.exists():
        print(f"error: {args.current} not found", file=sys.stderr)
        return 2
    if not args.baseline.exists():
        print(f"error: baseline {args.baseline} not found "
              f"(run `frep_bench --json > {args.baseline}` to create one)",
              file=sys.stderr)
        return 2

    cur = load_run(args.current)
    base = load_run(args.baseline)

    thresholds = {
        "cpu": args.cpu_threshold if args.cpu_threshold is not None
                                  else args.threshold,
        "gpu": args.gpu_threshold if args.gpu_threshold is not None
                                  else args.threshold,
    }

    ok = compare(cur, base, thresholds)
    print()
    if ok:
        if thresholds["cpu"] == thresholds["gpu"]:
            print(f"✓ No regressions detected (threshold {thresholds['cpu']}×)")
        else:
            print(f"✓ No regressions detected "
                  f"(CPU {thresholds['cpu']}×, GPU {thresholds['gpu']}×)")
        return 0
    print(f"✗ Performance regressions detected over the configured threshold")
    return 1


if __name__ == "__main__":
    sys.exit(main())
