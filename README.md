# Vipul Kumar (Solorush2021)
### ML Systems & Deep Learning Training Infrastructure Engineer
 Jamshedpur, Jharkhand, India | [LinkedIn](https://www.linkedin.com/in/vipul-kumar-4388801a7/) | vipulpower@gmail.com

---

## 🚀 About Me
I build high-throughput distributed training stacks and ultra-low-latency serving infrastructure for frontier foundation models. My work bridges the gap between deep learning research and high-performance hardware, specializing in maximizing Model Flops Utilization (MFU) and optimizing GPU memory access at the metal level.

- 🛠️ **Current Focus:** Deep training optimizations (NVFP4, Muon, ZeRO++) and custom Triton/CUDA kernel compilation.
- ⚡ **Superpower:** Profiling bottlenecks at the warp level using Nsight Systems/Compute and writing fused kernels to keep GPUs saturated at line rate.
- 🏢 **Previous Work:** AI Expert at **NiftyBooks** (Austria-based startup), where I cut model serving costs by 77% and optimized multi-node pretraining stacks.

---

## 🛠️ Tech Stack & Tooling

<table>
  <tr>
    <td align="center" width="96">
      <img src="https://simpleicons.org/icons/python.svg" width="48" height="48" alt="Python" /><br /><b>Python</b>
    </td>
    <td align="center" width="96">
      <img src="https://simpleicons.org/icons/cplusplus.svg" width="48" height="48" alt="C++" /><br /><b>C++</b>
    </td>
    <td align="center" width="96">
      <img src="https://simpleicons.org/icons/nvidia.svg" width="48" height="48" alt="CUDA" /><br /><b>CUDA / Triton</b>
    </td>
    <td align="center" width="96">
      <img src="https://simpleicons.org/icons/pytorch.svg" width="48" height="48" alt="PyTorch" /><br /><b>PyTorch</b>
    </td>
    <td align="center" width="96">
      <img src="https://simpleicons.org/icons/linux.svg" width="48" height="48" alt="Linux" /><br /><b>Linux / Bash</b>
    </td>
    <td align="center" width="96">
      <img src="https://simpleicons.org/icons/docker.svg" width="48" height="48" alt="Docker" /><br /><b>Docker</b>
    </td>
  </tr>
</table>

- **Distributed Training:** Megatron-LM (TP/PP/DP), DeepSpeed ZeRO-1/2/3, ZeRO++ (hpZ, qgZ), PyTorch FSDP, Slurm.
- **Serving & Inference:** vLLM v1/v2, PegaFlow (Rust KV-cache daemon), TurboQuant (KV compression), P-EAGLE Speculative Decoding.
- **Hardware Architecture:** Qualcomm Snapdragon 8 Gen 2 / Elite (NPU/DSP), Jetson Orin, NVIDIA Blackwell B200 / Hopper H100.
- **Analysis:** Nsight Compute, Nsight Systems, PyTorch Profiler, KernelBench.

---

## 🌟 Highlighted Projects

### ⚡ [OptiTrain-FP4](https://github.com/Solorush2021/OptiTrain-FP4) — Blackwell-Optimized Training Infrastructure
A high-performance pipeline simulating native **NVFP4 mixed-precision pretraining** on Blackwell 5th-gen Tensor Cores.
- **Micro-block Scaling:** Implemented dual-level scaling (16-element blocks with per-block FP8 scale + global FP32 scale) to limit outlier quantization error.
- **Stochastic Rounding:** Integrated stochastic rounding and random Hadamard transforms to prevent gradient accumulation bias and activation energy collapse.
- **Triton Attention:** Optimised a custom fused attention kernel utilizing Blackwell's **256KB Tensor Memory (TMEM)** per SM, hitting **71% of peak B200 FLOPS** (1.8x throughput increase).
- **Auto-Tuning:** Plugs into an autonomous reinforcement learning tuning loop to optimize hyperparameters (QK-clipping, Newton-Schulz iterations) without human intervention.

### 📱 [EdgeDeploy-ASR](https://github.com/Solorush2021/EdgeDeploy-Inference) — On-Device Indic ASR Inference Engine
An offline, high-efficiency speech transcription system for code-mixed Indian languages running on edge hardware.
- **Indic Curation:** Fine-tuned AI4Bharat IndicConformer (120M) on Hindi-English code-mixed speech with telemetry-grade noise profiles.
- **Quantization:** Compressed model from 480MB to 145MB via INT8 dynamic quantization with <3% accuracy decay.
- **Latency & Power:** Achieved **sub-80ms first-word latency** on Snapdragon 8 Gen 2 with NPU/DSP acceleration.
- **Dual Decoding:** Hybrid CTC+RNNT engine for fast batch transcription (CTC) and real-time streaming input (RNNT).

### 🛸 [Drone-ML](https://github.com/Solorush2021/Drone-ml) — Deep RL Drone Stabilization Policy
Gymnasium environment training a reinforcement learning policy for stable quadcopter hover under wind disturbances.
- Trained a continuous PPO agent in a 3D simulation with 12-DOF physics (RK4 integration) to control motor thrust curves.
- Generalized to unseen wind vectors (0-5 m/s) with <3cm steady-state position error.

---

## 📊 Metrics & Achievements
- **2.11x speedup** on custom Triton attention kernels over baseline compiled PyTorch code.
- **60% drop** in production serving costs (₹18L $\rightarrow$ ₹7.2L/mo) via disaggregated serving, PegaFlow cache, and TurboQuant.
- **0% checkpoint corruption** over multi-day pretraining runs via SHA-256 dual-write verification.
- **94.7% intent preservation** on Indic ASR under noisy conversational conditions.

---

<p align="center">
  <a href="https://github.com/Solorush2021">
    <img src="https://github-readme-stats.vercel.app/api?username=Solorush2021&show_icons=true&theme=dark" alt="Vipul's GitHub Stats" />
  </a>
</p>
