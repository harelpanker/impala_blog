# Wide-EP MoE Serving: Dispatch/Combine + Dual-Batch Overlap (DBO) + DeepEP (NVL72 / GB200)

> **Goal:** Build a systems-level mental model (and a tunable math model) for why Wide-EP becomes attractive at huge token volumes, and how **dispatch+combine**, **DBO (dual-batch overlap)**, and **DeepEP (HT/LL kernels)** fit together.

---

## TL;DR

- A Wide-EP MoE layer executes a **3-stage pipeline**:
  1) **Dispatch** (sparse all-to-all)  
  2) **Expert compute** (straggler rank dominates)  
  3) **Combine** (sparse all-to-all back)

- Without overlap, layer time is approximately a **sum**:
  \[
  t_{\text{no}} \approx t_d + t_c + t_b \approx 2t_d + t_c
  \]

- With DBO, steady-state time approaches a **max**:
  \[
  t_{\text{dbo}} \approx \max(2t_d,\ t_c)
  \]
  So DBO helps when **compute can cover comm**.

- DeepEP provides two kernel families:
  - **LL (low-latency):** reduce fixed overhead \(L\) for small messages (decode-like)
  - **HT (high-throughput):** maximize effective bandwidth \(BW_{\text{eff}}\), including NVLink↔RDMA forwarding (prefill/offline)

- On **GB200 NVL72**, intra-rack NVLink bandwidth is enormous, while scale-out bandwidth is typically limited by **ConnectX-7 / Quantum-2 400 Gb/s** links — so off-rack traffic is the usual bottleneck.

---

## 1) What is Wide-EP in deployment terms?

### Model view (you already know)

Input activations:
\[
X \in \mathbb{R}^{T \times d}
\]

Top-\(k\) routing chooses experts \(g(t)=(e_{t,1},...,e_{t,k})\). MoE output:
\[
Y_t = \sum_{j=1}^{k} \alpha_{t,j} f_{e_{t,j}}(X_t)
\]

### Systems view (the part that matters for serving)

Experts are placed across **EP ranks** (GPUs):
\[
\pi(e) \in \{1,\dots,P\}
\]

So tokens must move to where their experts live:

- **Dispatch:** send token activations to expert owners (sparse all-to-all)
- **Combine:** send expert outputs back to token owners (inverse all-to-all)

---

## 2) Dispatch/Combine payload math (bytes)

Let:

- \(T\) = tokens per step *per rank* (important: per GPU in the EP group)
- \(d\) = hidden size
- \(k\) = top-k experts
- \(s\) = bytes/element (BF16/FP16 = 2; FP8 payload = 1)

**One direction (dispatch OR combine):**
\[
V_{\text{shot}}(T) \approx T \cdot k \cdot d \cdot s
\]

**Round-trip per MoE layer (dispatch + combine):**
\[
V_{\text{layer}}(T) \approx 2Tkd s
\]

If only a fraction \(\rho\) of tokens go **off-rack**, then RDMA-visible bytes:
\[
V_{\text{RDMA}}(T) = 2\rho Tkd s
\]

---

## 3) Time model (the one you actually tune)

### 3.1 Communication time (dispatch+combine)

Model each all-to-all as “latency + bandwidth”:

\[
t_d(T) \approx L_{a2a} + \frac{V_{\text{shot}}(T)}{BW_{\text{eff}}}
\]
\[
t_b(T) \approx t_d(T)
\]

So:
\[
t_{\text{comm}}(T) = t_d + t_b \approx 2\left(L_{a2a} + \frac{Tkd s}{BW_{\text{eff}}}\right)
\]

**DeepEP framing:**

- LL reduces \(L_{a2a}\)
- HT increases \(BW_{\text{eff}}\) (and can add hierarchy/forwarding to do it)

### 3.2 Compute time is straggler-dominated

Let \(c_{\text{tok}}\) be expert compute seconds per routed token on the critical rank, and \(\gamma \ge 1\) be straggler factor (routing skew, imbalance, topology effects):

\[
t_c(T) \approx \gamma \cdot \frac{T k}{P} \cdot c_{\text{tok}}
\]

---

## 4) No-overlap vs DBO

### Baseline (no overlap)

\[
t_{\text{no}}(T) \approx t_{\text{comm}}(T) + t_c(T) \approx 2t_d(T) + t_c(T)
\]

### DBO (steady-state bound)

A clean steady-state model:

\[
t_{\text{dbo}}(T) \approx \max\big(t_{\text{comm}}(T),\ t_c(T)\big)
\]

**Interpretation:** DBO converts a “sum” into a “max”.

---

## 5) The key inequality: when DBO is worth it

DBO provides near-ideal benefit when:

\[
t_c(T) \ge t_{\text{comm}}(T)
\]

Plugging the models:

\[
\gamma \frac{Tk}{P}c_{\text{tok}}
\ge
2\left(L_{a2a} + \frac{Tkd s}{BW_{\text{eff}}}\right)
\]

Solve for a token threshold \(T^\*\):

\[
T \ge
\frac{2L_{a2a}}
{\gamma \frac{k}{P}c_{\text{tok}} - \frac{2kd s}{BW_{\text{eff}}}}
\]

### What this tells you immediately

- Increasing EP width \(P\) reduces compute per rank → overlap becomes harder.
- Improving \(BW_{\text{eff}}\) (DeepEP HT) lowers comm → overlap becomes easier.
- Larger steps \(T\) help by amortizing latency and increasing compute volume.

---

## 6) Choosing DeepEP HT vs LL: exact crossover formula

Model each kernel family as \((L, BW)\).

Pick HT when:

\[
L_{\text{HT}} + \frac{V}{BW_{\text{HT}}} < L_{\text{LL}} + \frac{V}{BW_{\text{LL}}}
\]

Solve for message-size threshold \(V^\*\):

\[
V^\*=
\frac{L_{\text{LL}}-L_{\text{HT}}}
{\frac{1}{BW_{\text{HT}}}-\frac{1}{BW_{\text{LL}}}}
\]

**Practical:** below \(V^\*\) use LL (decode-ish), above \(V^\*\) use HT (prefill/offline-ish).

---

## 7) NVL72 (GB200) network numbers to plug into the model

### Intra-rack (NVLink domain)

- NVLink Switch System inside NVL72 provides extremely high aggregate GPU communication bandwidth (often quoted as **130 TB/s aggregate** across the system).

### Inter-rack / scale-out (RDMA)

- ConnectX-7 supports **400 Gb/s per port**, which is:
  \[
  400\ \text{Gb/s} = 50\ \text{GB/s (raw, one-way)}
  \]
  Effective \(BW_{\text{eff}}\) is typically \( \eta \cdot 50\text{ GB/s}\) with \(\eta \in [0.6,0.9]\) depending on message size, NIC, and topology.

### Consequence

\[
BW_{\text{NVLink}} \gg BW_{\text{RDMA}}
\]

So **off-rack traffic is the bottleneck** unless \(\rho\) is tiny. This is exactly why DeepEP HT emphasizes NVLink↔RDMA forwarding.

---

## 8) “Optimal” configuration: what you can actually optimize

### 8.1 Objective (most common)

Minimize per-layer time:

\[
\min_{\text{kernel},\ \text{chunk},\ \text{DBO}}\ t_{\text{layer}}
\]

with:

- kernel ∈ {LL, HT}
- chunk size \(C\) (or “message coalescing” granularity)
- DBO enabled/disabled and microbatch split

### 8.2 Optimal DBO decision rule (exact)

Enable DBO if:

\[
\max(t_{\text{comm}}(T),t_c(T)) \ll t_{\text{comm}}(T)+t_c(T)
\]

which happens when both stages are non-trivial and overlapable; in practice you check:

\[
t_c(T)\ \text{is within a factor of}\ t_{\text{comm}}(T)
\]

### 8.3 Optimal HT vs LL (exact)

Compute:

- \(V = Tkd s\)
- compare \(t_{\text{LL}}(V)\) vs \(t_{\text{HT}}(V)\)
- pick the smaller

### 8.4 Chunk sizing (overlap-friendly lower bound)

If you chunk comm into blocks of size \(C\):

\[
t_{\text{chunk}}(C)=L+\frac{C}{BW}
\]

If compute per chunk is \(\beta C\), overlap requires:

\[
\beta C \ge L + \frac{C}{BW}
\Rightarrow
C \ge \frac{L}{\beta - 1/BW}
\]

This gives a real “don’t make chunks too tiny” threshold.

---

## 9) Practical measurement plan (do this once, then everything becomes plug-and-play)

Create three microbench numbers on your NVL72 environment:

1) **DeepEP comm fit:** run dispatch+combine with varying payload \(V\), fit:
   \[
   t(V)=2(L + V/BW_{\text{eff}})
   \]
   separately for LL and HT.

2) **Expert compute per routed token:** measure \(c_{\text{tok}}\) for the expert MLP (critical path kernel).

3) **Straggler factor \(\gamma\):** run real routing distributions and measure max-rank load / mean-rank load.

Then:

- Compute \(V^\*\) crossover for LL vs HT
- Compute \(T^\*\) threshold where DBO becomes “near-perfect”
- Decide EP width \(P\) that doesn’t collapse compute per rank below comm

---

## 10) Appendix — Parameter Table (copy/paste)

**Workload params**

- \(d\) = hidden size  
- \(k\) = top-k  
- \(s\) = bytes/elem  
- \(T\) = tokens per step per rank  
- \(P\) = EP ranks  
- \(\rho\) = off-rack routed fraction  
- \(\gamma\) = straggler factor  
- \(c_{\text{tok}}\) = compute seconds per routed token  

**Fabric params**

- \(L_{a2a}\) (LL and HT separately)  
- \(BW_{\text{eff}}\) (LL and HT separately)  

---

## 11) Embedding your interactive widget (optional)

If you host the HTML page (your live visualization) somewhere reachable, you can embed it in Notion or any docs site via an embed block / iframe.

