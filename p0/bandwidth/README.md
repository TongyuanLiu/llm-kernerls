# Vector add bandwidth

`C = A + B` on an RTX 3090 Ti (theoretical DRAM bandwidth 1008 GB/s), CUDA 13.4.
Explicit `cudaMalloc` + `cudaMemcpy`; only the kernel is timed with `cudaEvent`
(3 warmup runs, median of 20). Bytes moved = 12 × N (read A, read B, write C).

```bash
nvcc -O2 -arch=sm_86 vector_add.cu -o vector_add
for k in $(seq 20 28); do ./vector_add $k; done
```

| N | Bytes (12 × N) | Kernel time (median) | Bandwidth | Utilization |
|---|---|---|---|---|
| 2^20 | 13 MB | 0.016 ms | 768.0 GB/s | 76.2% |
| 2^21 | 25 MB | 0.032 ms | 792.8 GB/s | 78.6% |
| 2^22 | 50 MB | 0.059 ms | 847.4 GB/s | 84.1% |
| 2^23 | 101 MB | 0.114 ms | 885.6 GB/s | 87.9% |
| 2^24 | 201 MB | 0.223 ms | 901.9 GB/s | 89.5% |
| 2^25 | 403 MB | 0.443 ms | 908.1 GB/s | 90.1% |
| 2^26 | 805 MB | 0.882 ms | 913.5 GB/s | 90.6% |
| 2^27 | 1611 MB | 1.756 ms | 917.1 GB/s | 91.0% |
| 2^28 | 3221 MB | 3.502 ms | 919.8 GB/s | 91.3% |

All runs: `Max error: 0`.

## ncu cross-check (N = 2^26)

```bash
ncu --section SpeedOfLight --launch-skip 3 --launch-count 1 ./vector_add
```

| | cudaEvent | ncu (default clock lock) | ncu (`--clock-control none`) |
|---|---|---|---|
| Time | 881 µs | 878.6 µs | 880.1 µs |
| DRAM utilization | 90.7% | 93.1% | 93.0% |

Timing agrees within 0.3%. Clock locking makes no difference here (DRAM at
10.24 GHz either way), and cache flushing doesn't matter because 768 MiB of data
is far larger than the 6 MB L2. The remaining ~2.4 points most likely come from
ncu using its own peak and counting actual DRAM bytes rather than the
theoretical 12 × N (not yet verified with `dram__bytes.sum`).

## Why small N gets lower bandwidth

Each launch carries a fixed overhead of a few µs (launch, ramping up to and
draining all SMs, event timing). At 2^20 the whole kernel is only ~16 µs, so
that overhead is a large fraction; at 2^28 (3.5 ms) it is negligible. Even 2^20
(12.6 MB) exceeds the L2, so these are still DRAM numbers.
