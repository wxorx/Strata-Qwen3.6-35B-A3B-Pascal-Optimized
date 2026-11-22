# Strata Qwen3.6 35B-A3B — Pascal Optimized

An experimental source fork of Strata's Qwen3.6 engine, tuned for the **NVIDIA GeForce GTX 1050 Ti Mobile 4 GB** (`sm61`). It runs the 35B-A3B MoE model with CPU/GPU expert execution, an adaptive GPU expert cache, MTP drafting, and a long-prompt layer-major prefill path.

This repository contains source code and reproducible Windows build/run scripts. It does **not** commit the model, binaries, DLLs, benchmark logs, or release archives. Tagged builds are compiled by GitHub Actions and attached separately on the [Releases page](https://github.com/SirDebugALot/Strata-Qwen3.6-35B-A3B-Pascal-Optimized/releases).

## Performance on the GTX 1050 Ti Mobile

Same machine, model, prompt token IDs, 25,088-token effective context, single stream, greedy sampling, and 1,000 generated tokens:

| Engine | 8K prefill | 8K decode | 16K prefill | 16K decode |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp b11524 | 113.47 t/s | 7.88 t/s | 105.12 t/s | 7.16 t/s |
| Original Strata Qwen3.6 branch, Pascal compatibility build | 141.75 t/s | 15.00 t/s | 106.76 t/s | 13.63 t/s |
| **Strata Qwen3.6 Pascal Optimized** | **256.80 t/s** | **24.53 t/s** | **210.98 t/s** | **21.40 t/s** |

The optimized build improved:

- 8K: **+81.2% prefill / +63.6% decode** over the original compatibility build, and **+126.3% / +211.3%** over llama.cpp.
- 16K: **+97.6% prefill / +57.1% decode** over the original compatibility build, and **+100.7% / +198.8%** over llama.cpp.

Test system: Intel Core i7-8750H, 48 GB RAM, Windows 11, NVIDIA driver 577.00. The optimized executable used CUDA 12.9; llama.cpp used its official CUDA 12.4 Windows package. The optimized figures average four runs at each prompt size; each comparison row averages two runs.

The untouched original commit does not build for `sm61` or import all expert tensor types in this GGUF. Its measured row therefore uses the smallest required Pascal/import compatibility changes while retaining the original inference path. These are throughput measurements, not model-quality scores; the fixed-length document answers exposed factual or structural review flags across the tested engines.

## Performance on the Quadro P1000 (Linux)

Same machine, requests through each server's `/v1/chat/completions` with `temperature=0` and
`cache_prompt=false`, single stream, on a **Quadro P1000 4 GB** (`sm_61`) under Ubuntu 24.04.
llama.cpp rows use this machine's own production profiles with `llama.cpp` b12387 built for CUDA
(`--n-cpu-moe 37`, q8_0 KV, FlashAttention, batch 4096/2048, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`,
196K context); the MTP row adds `--spec-type draft-mtp --spec-draft-n-max 2` on the MTP artifact.
Strata runs `./BUILD-PASCAL-SM61.sh` (`SQ_ENABLE_PREFILL_MMQ=ON`, `SQ_ENABLE_MMQ_KQUANTS=ON`) and
`./START-OPTIMIZED-PASCAL.sh` on the same MTP artifact, published profile, 25,088-token context,
6 CPU threads. Documents are tokenized from this repository's own text: 8K ≈ 7.9K prompt tokens,
16K ≈ 16.1K prompt tokens. Prefill rows answer 8 tokens; decode rows generate 256 tokens after the
same document; Chat decode is three thinking-mode prompts (code, Japanese, math) at 384 tokens,
six requests averaged. The plain llama.cpp row uses the non-MTP twin of the same quant.

| Engine | 8K prefill | 8K decode | 16K prefill | 16K decode | Chat decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp b12387 (default profile, no drafts) | 182.65 t/s | 11.30 t/s | 158.45 t/s | 10.15 t/s | 13.09 t/s |
| llama.cpp b12387 (`draft-mtp`, strongest tested) | 155.25 t/s | 11.14 t/s | 140.53 t/s | 9.39 t/s | 13.61 t/s |
| **Strata Qwen3.6 Pascal Optimized** | **183.70 t/s** | **20.80 t/s** | 154.50 t/s | **17.40 t/s** | **26.43 t/s** |

Against the strongest llama.cpp configuration on this card, the optimized build improves token
generation by **+87% after 8K, +85% after 16K, +94% on chat** — about **2x** — while prefill is
level with the plain profile and **+18% / +10%** against the MTP profile (draft layers cost
llama.cpp prefill throughput). At 16K, Strata's prefill trails the plain profile slightly; its
expert-cache and KV budgets are tighter at 4 GB than the 1050 Ti reference runs' because the P1000
keeps a larger share of the model resident.

Test system: Intel Core i7-8700T (6C/12T, AVX2), 62 GB RAM, Ubuntu 24.04, Quadro P1000 4 GB,
NVIDIA driver 580.196.07, CUDA Toolkit 12.6.85. Each row is one measured pass (chat decode: 6
requests; engine-level repeats of the profile varied by under 0.2%). These are throughput
measurements, not model-quality scores; `--selfcheck-layer-major` passes exactly (KL 0, identical
argmax across continuation, MTP verification, rollback and prefix reuse) for the Q4_K MMQ path on
this hardware, and `docs/UNSLOTH_Q4.md`-style native expert quantization is kept throughout.

## Model

Download the exact tested model from Hugging Face:

- Repository: [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
- File: [`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/main/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf)
- Tested SHA-256: `36f9ec0e4c775f6efd3a61c1ea76f0875128469c3d012e3d9e2495a90e7a7150`

The file is about 14.1 GB. Git LFS is not required for this source repository because the GGUF is not included.

## Windows build for Pascal `sm61`

Requirements:

- Windows 10 or 11
- Visual Studio 2022 Build Tools with Desktop C++
- CUDA Toolkit 12.x with Pascal support
- CMake, Ninja, and Git on `PATH`
- An AVX2-capable CPU

Run:

```bat
BUILD-PASCAL-SM61.bat
```

The script checks out the pinned llama.cpp dependency at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, configures the Qwen engine with `SQ_ENABLE_PREFILL_MMQ=ON`, and writes:

```text
build-sm61\strata-qwen35moe.exe
```

No prebuilt executable is committed.

## Linux build for Pascal `sm_61`

The engine needs no source changes on Linux: its CMake already carries GCC/POSIX branches and the
C++/CUDA files are `#ifdef _WIN32`-guarded. Tested on Ubuntu 24.04 with gcc 13, CMake 3.28 and CUDA
Toolkit 12.6 on a Quadro P1000 (`sm_61`).

```sh
./BUILD-PASCAL-SM61.sh     # fetches the same pinned llama.cpp, builds build-sm61/strata-qwen35moe
./START-OPTIMIZED-PASCAL.sh ~/.llama.cpp/models/Qwen3.6-35B-A3B-UD-Q4_K_M/MTP/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

`BUILD-PASCAL-SM61.sh` adds `-DSQ_ENABLE_MMQ_KQUANTS=ON`: a new opt-in gate that also compiles
llama.cpp's `q4_k/q5_k/q5_1/q6_k` MMQ template instances behind the `STRATA_MMQ_KQUANTS` branches
already present in `prefill_mmq.cu`. Without them the layer-major prefill refuses K-quant expert
GGUFs (`fatal: layer-major prefill requires MMQ coverage for every main and MTP layer`), so only the
IQ2/IQ3/Q2_K/Q3_K family — the `UD-IQ3_XXS` artifact measured above — can use the optimized path.
The gate defaults to OFF in CMake, so the Windows build and its released binary are unchanged.
`--selfcheck-layer-major` passes exactly (KL 0, identical argmax through continuation, MTP
verification, rollback and prefix reuse) with the Q4_K MMQ path.

`START-OPTIMIZED-PASCAL.sh` creates the `.venv` on first run, builds nothing, and starts the same
launcher with the same engine options and environment as `START-OPTIMIZED-PASCAL.bat` (default context
25000). It binds **0.0.0.0** so the web UI and the OpenAI/Anthropic APIs are reachable on the LAN;
pass `--host 127.0.0.1` to keep it local-only. Measurements for this card are tabulated under
[Performance on the Quadro P1000 (Linux)](#performance-on-the-quadro-p1000-linux). The published
profile itself is the tuned setting for the P1000: CUDA graphs, oversubscribed CPU threads
(`--threads 12` measured 4.1 t/s), a CPU-side LM head, larger MMQ tiles, a bigger cache reserve and
stronger cache adaptation all measured neutral or worse on this hardware.

#### Long contexts and KV overflow (4 GB cards)

`--max-context` is clamped to the model's own 262,144-token context. KV costs ~21.5 KiB per token
in fp16, so the P1000 holds roughly **65K tokens of KV on the device** after evicting the expert
cache; normal (25K-profile) speeds apply below that. Beyond it the engine no longer stops: the
attention KV moves wholesale to **pinned host RAM** (the kernels keep a UVA device pointer, so
results are unchanged; every KV access now crosses PCIe). Measured on this machine: a 30K-token
request stays on-device (~15 t/s decode), a 50K request stays on-device with the expert cache
evicted (~11 t/s), a 75K request overflows to host RAM (~45–50 t/s prefill, ~2–3 t/s decode) and
completes. The status line marks the mode with `KV on host RAM (overflow mode)`. For sustained
work above ~65K tokens the llama.cpp unified-memory configuration remains the faster choice; the
overflow mode exists so an oversized request finishes instead of ending the engine.

Images are not supported by this engine: its `--serve` answers `GENI` with an error, the server
reports `"images": false`, and no `mmproj` file is loaded.

## Prebuilt Windows release

The `Windows Pascal release` GitHub Actions workflow builds the same `sm61` target with CUDA Toolkit 12.9. A pushed `v*` tag creates a GitHub Release containing a ZIP and its SHA-256 checksum. The ZIP contains the engine, required cuBLAS and Visual C++ runtime DLLs, the optimized launcher, and the web/API server. It does not contain the GGUF model.

After extracting a release, run `SETUP-RUNTIME.bat` once and then:

```bat
START-OPTIMIZED-PASCAL.bat "D:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
```

See [BINARY-RELEASE.md](BINARY-RELEASE.md) for requirements and complete instructions.

## Run the optimized profile

Create the Python environment once:

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the server with the GGUF path:

```bat
START-OPTIMIZED-PASCAL.bat "D:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
```

The launcher extracts the tokenizer from the GGUF on first use, writes only local ignored state under `.local-data` and `runtime`, and serves the OpenAI-compatible API and web UI on [http://127.0.0.1:8080](http://127.0.0.1:8080). Press `Ctrl+C` in its console to stop it.

Optional arguments:

```text
START-OPTIMIZED-PASCAL.bat MODEL.gguf --port 8080 --context 25000 --threads 6
START-OPTIMIZED-PASCAL.bat MODEL.gguf --tokenizer D:\path\to\tokenizer
```

The tested profile assumes one inference stream and a 4 GB Pascal GPU. Display workloads also consume VRAM; start the server after connecting the monitors you intend to use.

## How the optimization works

A 4 GB GPU cannot keep this model's weights, KV cache, expert cache, MTP state, and a large prompt workspace resident at the same time. The central design change is to give the same VRAM to different jobs during **prefill** and **decode**, instead of forcing one static allocation to serve both phases.

### The execution flow

1. **Keep the GGUF experts in their native quantization.** IQ2/IQ3/IQ4 main-model experts and the Q2_K/Q3_K MTP experts are imported without silently expanding the whole expert arena to Q6_K. This reduces host RAM, memory bandwidth, cache bytes, and startup conversion work. Unsupported types fail explicitly instead of being requantized behind the user's back. The CPU path uses AVX2 kernels; the GPU cache keeps the same native payloads.
2. **Loan decode memory to long-prompt prefill.** For a long prompt the engine records the current expert-cache layout, releases that cache, and temporarily releases the decode-only MTP rollback arena and GPU LM head. On this model those last two allocations are about 188 MiB and 515 MiB. The final hidden row is retained, so the large prefill workspace can be destroyed before the head is restored and the first decode logits are calculated.
3. **Change the long-prompt loop order.** Conventional chunk-major execution runs every layer for one chunk before moving to the next chunk. This fork processes one layer across a bounded prompt window in 1,024-token tiles, then advances to the next layer. The default 8,192-token window amortizes expert gathering and dense-weight preparation while keeping activation memory bounded. The engine reduces the window automatically when CUDA free memory is tight.
4. **Multiply packed experts directly.** Routed activations are quantized to Q8_1 and the original packed GGUF expert weights are consumed by llama.cpp-derived DP4A MMQ kernels. Experts are grouped in batches of up to 16, covering all 40 main layers and the MTP layer. This avoids expanding every routed expert to temporary FP16 or Q6 matrices before the multiply.
5. **Restore a decode-oriented layout.** After prefill, the temporary workspace is freed, the LM head and rollback state are recreated, and the GPU expert cache is refilled using routing observed in the prompt. Decode therefore starts with a cache shaped by the current request rather than only a static global profile.
6. **Run CPU and GPU MoE work concurrently.** Cached experts execute on the GPU while misses execute through persistent AVX2 CPU workers. CUDA events and an asynchronous cache copy path replace full-device synchronization. GPU results are allowed to start early, but the final reduction preserves router order.
7. **Use bounded online cache adaptation.** Per-layer expert scores combine a saved profile with live routing, decay by `0.85`, and are reconsidered every four tokens with at most 16 swaps. The policy limits PCIe traffic and eviction churn; cache-hit percentage alone is not treated as the objective because a higher hit rate can still be slower when swaps or non-overlapped transfers dominate.
8. **Reduce speculative-decoding work.** MTP drafts up to three tokens while a probability threshold stops weak draft chains. Draft-vocabulary subsets, last-row-only output work, and explicit verification/rollback handling reduce work that cannot affect accepted tokens. Its rollback allocation is borrowed during prefill and restored before decode.
9. **Tune attention and dense work for `sm61`.** Prefill shares prepared K/V data, uses an AQ6 bank layout for attention, and selects a full-tile cuBLAS algorithm that measured well on Pascal. Tail shapes retain the general path. These choices are runtime-gated so an unsupported architecture does not accidentally use the Pascal-specific dispatch.

The corresponding implementation is concentrated in [engine.cpp](engines/qwen35moe/src/engine.cpp), [prefill.cu](engines/qwen35moe/src/prefill.cu), [prefill_mmq.cu](engines/qwen35moe/src/prefill_mmq.cu), the [native IQ CPU kernels](engines/qwen35moe/src/native_iq/iq_avx2.cpp), and the explicit tested settings in [run_qwen36_pascal_optimized.py](tools/run_qwen36_pascal_optimized.py).

### What transfers to other GPU generations

| Technique | Why it helps | Porting guidance |
| --- | --- | --- |
| Native quantized experts | Avoids conversion, extra resident bytes, and wasted memory bandwidth | Useful on every GPU and CPU, but each GGUF type needs a validated native kernel. |
| Phase-specific VRAM loans | Prefill and decode need different large allocations | Broadly useful on low-VRAM systems. Recalculate the head, rollback, KV, and safety-reserve sizes for the model and GPU. |
| Layer-major bounded windows | Reuses prepared weights across many prompt tokens | Broadly useful for long prompts. Larger GPUs can use larger windows; short prompts should keep a lower-overhead path. |
| Packed MMQ | Reads compressed weights once and uses integer dot products | Pascal benefits from DP4A. Turing, Ampere, Ada, and newer GPUs should benchmark IMMA/Tensor Core or generation-tuned kernels rather than assume the `sm61` launch geometry is best. |
| CPU/GPU expert overlap | Turns otherwise idle CPU cores into useful miss capacity | Useful when CPU compute and PCIe copies can hide behind GPU work. Retune worker count, affinity, and spin/wait thresholds for the host CPU. |
| Cost-aware expert cache | Keeps hot experts resident without unlimited transfers | Useful for any MoE. Capacity, decay, swap budget, and admission policy must reflect PCIe speed and expert size. |
| Prompt-informed cache refill | The prompt often predicts the experts used during its answer | Model-independent in concept. Measure whether prompt and decode routing correlate for the target model and workload. |
| MTP work elimination | Avoids projection and state work that cannot affect accepted drafts | Applies to compatible speculative models, but draft layout, verification, and rollback semantics are model-specific. |
| Attention/dense autotuning | Library defaults are not always best for unusual old or small GPUs | Keep the dispatch gated by architecture and shape, and benchmark full tiles and tails separately. |
| AVX2 native expert kernels | Makes CPU fallback fast enough to overlap usefully | Replace with AVX-512/VNNI or ARM kernels where available; do not compile the whole program for an ISA the user's CPU may lack. |

### Design lessons from the experiments

- **A larger expert cache is not always the best use of VRAM during prefill.** Long prompts touch many experts, so temporarily using that memory for wider prompt work can save more time. Restore the cache for token-by-token decode, where reuse matters more.
- **Larger chunks alone eventually run out of memory.** Layer-major windows amortize preparation without retaining the entire prompt's intermediate state. Persistent KV memory still grows with context, so this does not make context unlimited.
- **Asynchrony needs a measured serialization point.** The engine already uses a copy stream and a multi-slot staging ring. Extra double buffering is useful only if CUDA-event timing shows that compute is waiting for copies; otherwise it only removes cache capacity.
- **Numerical order matters in MoE.** Early layer-major prototypes had large logit divergence even with one tile. Immutable routing bounds plus the original router-order combine restored parity. A fast path should be tested through continuation, MTP verification, rollback, and prefix reuse, not only one prompt forward pass.
- **Quantized KV is a quality decision as well as a memory decision.** INT8 KV did not pass this build's full-model quality gate, so the published profile keeps FP16 KV. Other GPUs can trade the saved KV memory for a larger expert cache only after model-level validation.
- **Optimize elapsed time, not one counter.** Expert hit rate, PCIe bytes, CPU utilization, MTP acceptance, and kernel time describe different costs. Final selection should use end-to-end prefill and decode throughput with the same generated workload.

## Source lineage

This fork starts from [`ansonsi/Strata-Qwen3.6-35B-A3B`](https://github.com/ansonsi/Strata-Qwen3.6-35B-A3B) commit `f6b748c78885ed19369860441660c9015c1a1c5f`, itself derived from [`Niko1221/Strata`](https://github.com/Niko1221/Strata). The packed-MMQ path builds against pinned llama.cpp/ggml source. See [SOURCE-PROVENANCE.md](SOURCE-PROVENANCE.md), [UPSTREAM-README.md](UPSTREAM-README.md), and [LICENSE](LICENSE).

This is an independent experimental fork and is not an official Qwen, Unsloth, llama.cpp, or upstream Strata release.
