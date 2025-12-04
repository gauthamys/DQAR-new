# DQAR Presentation Outline
**Target: 5 minutes (~8-10 slides)**
**Course Project Presentation - University of Illinois Chicago**

---

## Slide 1: Title (15 sec)

**Title:** DQAR: Accelerating Diffusion Transformers via Attention Reuse

**Subtitle:** Generative AI Course Project

**Your name:** Gautham Satyanarayana

---

## Slide 2: The Problem (30 sec)

**Title:** Diffusion Transformers are Slow

**Bullets:**
- DiT models generate amazing images but require 50+ denoising steps
- Each step computes attention across ALL 28 transformer layers
- Same expensive computation repeated over and over

**Equation (attention computation):**
```latex
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V
```

**Visual suggestion:** Simple diagram showing "Step 1 → Step 2 → ... → Step 50" with "Attention × 28 layers" at each step

---

## Slide 3: Key Insight (30 sec)

**Title:** Attention Outputs are Temporally Stable

**Bullets:**
- Adjacent timesteps produce similar attention patterns
- Especially true in later denoising stages when image structure is stable
- **Idea:** Cache and reuse attention outputs instead of recomputing

**Visual suggestion:** Two side-by-side attention heatmaps from timestep t and t+1 showing similarity

---

## Slide 4: Why K/V Caching Fails (45 sec)

**Title:** Naive K/V Caching Doesn't Work

**The Problem:**
- LLMs cache K, V and compute with fresh Q - works because tokens are sequential
- In diffusion: each timestep has DIFFERENT noise level
- Fresh Q (current noise) + Stale K,V (old noise) = **temporal mismatch**

**Table to recreate:**
| Component | Source | Noise Level |
|-----------|--------|-------------|
| Q (Query) | Current | t (current) |
| K (Key) | Cached | t-1 (stale) |
| V (Value) | Cached | t-1 (stale) |

**Result:** Visual artifacts and quality degradation

---

## Slide 5: Our Solution (45 sec)

**Title:** Attention Output Caching

**Key idea:** Cache the COMPLETE attention output, not K/V

**Pseudocode (simplified):**
```python
if should_reuse(layer, timestep) AND has_cache(layer):
    return cached_output[layer]
else:
    output = compute_attention(Q, K, V)
    cache[layer] = output
    return output
```

**Benefit:** No mismatch - we reuse the exact output, skipping entire attention computation

---

## Slide 6: When to Reuse? Simple Static Schedule (45 sec)

**Title:** Linear Scheduling with Warmup

**Two parameters control reuse:**
- **Warmup (w):** No reuse for first w% of steps (let structure form)
- **Layer fraction (ℓ):** Max % of layers that can reuse

**Decision rule:**
```
can_reuse(timestep, layer) = (progress > warmup) AND (layer in allowed_set)
```

**Scheduling equation:**
```latex
\text{ReusableLayers}(p) = \begin{cases}
0 & \text{if } p < w \\
\left\lfloor \frac{p - w}{1 - w} \cdot \ell \cdot L \right\rfloor & \text{otherwise}
\end{cases}
```

Where:
- p = progress through denoising (0 to 1)
- L = total number of layers (28 for DiT-XL)

**Visual suggestion:** Diagram showing warmup region → progressive reuse zone

---

## Slide 7: Key Results (45 sec)

**Title:** Warmup Dominates Quality, Layers Dominate Speed

**FID Score Table (quality metric - lower is better):**

| Warmup | 33% Layers | 50% Layers | 100% Layers |
|--------|------------|------------|-------------|
| 10% | 23.0 | 34.2 | 67.0 |
| 20% | 18.6 | 26.7 | 46.6 |
| 40% | **8.3** | 14.8 | 26.2 |

**Key findings:**
- 40% warmup → **3× better FID** than 10% warmup (same layers)
- More layers → more speedup but worse quality
- Sweet spot: **40% warmup, 33% layers**

---

## Slide 8: Surprising Finding (30 sec)

**Title:** Which Layers to Reuse? Doesn't Matter!

**Experiment:** Shallow-first (layers 0,1,2...) vs Deep-first (layers 27,26,25...)

**Result:** Identical FID scores across ALL 20 configurations tested (complete tie)

**Implication:**
- Only **how many** layers and **when** (warmup) matters
- NOT **which** layers
- Simple schedule works - no need for complex policies!

---

## Slide 9: Visual Comparison (30 sec)

**Title:** Quality Preserved

**Visual:** 2×2 grid of your benchmark images
- Baseline vs DQAR (class 207 - golden retriever)
- Baseline vs DQAR (class 360 - otter)

**Caption:** "40% warmup, 33% layers: **1.06× speedup, FID 8.3** - visually identical"

---

## Slide 10: Summary & Takeaways (30 sec)

**Title:** Conclusion

**Key contributions:**
1. K/V caching fails for diffusion → use **attention output caching**
2. Simple static schedule works: **warmup + layer fraction**
3. **Warmup dominates quality**, **layer fraction dominates speedup**
4. Which layers to reuse is **irrelevant** (surprising!)

**Recommended config:**
- Quality-focused: 40% warmup, 33% layers → **1.06× speedup, FID 8.3**
- Speed-focused: 30% warmup, 100% layers → **1.16× speedup**

**Memory overhead:** Only 63 MB (3.6% of baseline)

---

## Backup Slides (if questions)

### Backup 1: Pareto-Optimal Configurations

| Config | Warmup | Layers | Speedup | FID |
|--------|--------|--------|---------|-----|
| Quality | 40% | 33% | 1.06× | 8.3 |
| Balanced | 40% | 66% | 1.09× | 18.8 |
| Speed | 30% | 100% | 1.16× | 37.0 |
| Max Speed | 10% | 100% | 1.21× | 67.0 |

### Backup 2: Why Does Warmup Matter? (Theoretical Motivation)

**SNR from the noise schedule:**
```latex
\text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2} = \frac{\alpha_t^2}{1 - \alpha_t^2}
```

- **Early steps (low SNR):** Mostly noise → Structure forming → Need fresh computation
- **Late steps (high SNR):** Mostly signal → Structure stable → Safe to reuse

**This is why warmup helps!** Early steps are critical for quality.

### Backup 3: Future Work - Learned Adaptive Policy

**Implemented but not used in experiments:**

We built an MLP policy that could learn when to reuse:
```latex
p_{\text{reuse}} = \sigma\left(\text{MLP}_\theta\left(H_t, \text{SNR}_t, \|x_t\|_2, t\right)\right)
```

**Attention entropy** measures stability:
```latex
H_t = -\frac{1}{H \cdot T} \sum_h \sum_{i,j} A_t^{(h)}(i,j) \cdot \log\left(A_t^{(h)}(i,j)\right)
```

**Finding:** Simple static schedule works well enough that learned policy wasn't needed for our results. Future work could explore adaptive scheduling.

### Backup 4: Memory Profile

| Metric | Baseline | DQAR | Delta |
|--------|----------|------|-------|
| Peak Memory (MB) | 1734.3 | 1734.3 | **0** |
| Cache Memory (MB) | --- | 63.0 | +63.0 |
| Inference Time (s) | 3.08 | 2.67 | -0.41 |
| Speedup | 1.00× | 1.15× | +15% |

---

## Speaker Notes / Timing

| Slide | Time | Cumulative | Notes |
|-------|------|------------|-------|
| 1 | 0:15 | 0:15 | Quick intro, don't linger |
| 2 | 0:30 | 0:45 | Establish the problem clearly |
| 3 | 0:30 | 1:15 | Key insight - the "aha" moment |
| 4 | 0:45 | 2:00 | Why naive approach fails - important! |
| 5 | 0:45 | 2:45 | Your solution - the core contribution |
| 6 | 0:45 | 3:30 | How scheduling works - simple! |
| 7 | 0:45 | 4:15 | Results - emphasize warmup finding |
| 8 | 0:30 | 4:45 | Surprising null result - memorable! |
| 9 | 0:30 | 5:15 | Visual proof |
| 10 | 0:30 | 5:45 | Wrap up |

**Total: ~5:45** (can cut slide 9 if tight on time)

---

## Equations Summary (Copy-Paste Ready)

### LaTeX Format:

**Attention:**
```latex
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V
```

**Scheduling:**
```latex
\text{ReusableLayers}(p) = \begin{cases}
0 & \text{if } p < w \\
\left\lfloor \frac{p - w}{1 - w} \cdot \ell \cdot L \right\rfloor & \text{otherwise}
\end{cases}
```

**SNR (backup/theory):**
```latex
\text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2} = \frac{\alpha_t^2}{1 - \alpha_t^2}
```

**Entropy (backup/future work):**
```latex
H_t = -\frac{1}{H \cdot T} \sum_h \sum_{i,j} A_t^{(h)}(i,j) \cdot \log\left(A_t^{(h)}(i,j)\right)
```

**MLP Policy (backup/future work):**
```latex
p_{\text{reuse}} = \sigma\left(\text{MLP}_\theta\left(H_t, \text{SNR}_t, \|x_t\|_2, t\right)\right)
```

### Unicode Format (for slides without LaTeX):

**Attention:**
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

**Scheduling:**
```
ReusableLayers(p) = ⌊(p - w)/(1 - w) · ℓ · L⌋  if p ≥ w, else 0
```

**SNR:**
```
SNR(t) = αₜ² / σₜ² = αₜ² / (1 - αₜ²)
```

**Entropy:**
```
Hₜ = -1/(H·T) · Σₕ Σᵢⱼ Aₜ⁽ʰ⁾(i,j) · log(Aₜ⁽ʰ⁾(i,j))
```
