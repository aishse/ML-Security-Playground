# ML Security Playground

An interactive playground for experimenting with adversarial attacks on deep learning models — see how small, carefully chosen pixel perturbations can flip a model's prediction, and visualize exactly what changed.

## Stack

Python · PyTorch · Torchvision · Gradio · NumPy · Matplotlib

## Status: FGSM working, visualization in progress

### Done

- **Inference pipeline** — pretrained ResNet18 classifies uploaded images and returns top-5 ImageNet predictions.
- **Preprocessing architecture** — image pipeline keeps a raw `[0, 1]` pixel tensor (`[1, 3, 224, 224]`) separate from model-specific normalization, so attacks perturb raw pixels rather than normalized ones. Normalization is pulled from `weights.transforms()` rather than hardcoded, so models can be swapped later without rewriting constants.
- **FGSM attack** (`attacks/fgsm.py`) — untargeted, single-step. Since arbitrary uploaded images have no ground-truth label, the model's own top prediction is used as a pseudo-label; the attack increases the loss against that label.
- **Attack evaluation** (`test_fgsm.py`) — three-stage test: baseline prediction → generate adversarial example → re-classify and compare. Attack success = top prediction changed.
- **Perturbation metrics** (`utils/visualization.py`) — `tensor_to_image`, `calculate_perturbation`, `create_heatmap`, `calculate_metrics` (L∞ and L2 norms). Verified at `epsilon = 0.01`: L∞ = 0.01 exactly (as expected) and L2 ≈ 3.77, with perturbation visibly spread across the whole image rather than localized — consistent with how FGSM should behave.

### In progress

- **Side-by-side visualization** — a single Matplotlib figure showing original image / adversarial image / perturbation heatmap, with predictions, confidences, and metrics (epsilon, L∞, L2, attack success) in one glance. First draft (`visualize_attack`) is built and smoke-tested; next step is wiring it into `test_fgsm.py` against a real image and confirming labels/heatmap look right end-to-end.

## Project structure

```
ML-Security-Playground/
├── attacks/
│   └── fgsm.py
├── models/
│   └── model_loader.py
├── utils/
│   ├── preprocess.py
│   ├── predict.py
│   └── visualization.py
├── examples/
│   └── test.jpg
├── app.py
├── test_model.py
├── test_fgsm.py
└── fgsm_heatmap.png
```

## Where this is going

**Phase 5B — Finish the FGSM visualization (current)**
Wire `visualize_attack` into `test_fgsm.py` with real data, confirm it renders correctly, use it as the reference pattern for later attacks.

**Phase 6 — Epsilon robustness sweep**
Run FGSM across a range of epsilon values (0.001 → 0.1), tracking prediction, original-class confidence, attack success, L∞, and L2 at each step. Output a table now, a robustness plot later.

**Phase 7 — Gradio UI**
Upload image → pick attack → set epsilon slider → run → show original/adversarial/heatmap + metrics. `app.py` stays a thin orchestrator; attack logic stays in `attacks/`.

**Future attacks**
- **PGD** — iterative FGSM with projection back into the epsilon ball each step. Start only once the FGSM UI is solid.
- **Carlini & Wagner** — optimization-based, more complex. Start only once FGSM + PGD + visualization + Gradio are all stable.

**Possible defense extension**
Compare clean vs. FGSM accuracy on a standard model vs. an adversarially trained one. Nice-to-have if the core project finishes early — not a blocker.

## Key concepts

- **Epsilon** — max perturbation strength. Smaller = subtler, potentially weaker attack; larger = stronger, more visible.
- **L∞ norm** — largest single pixel/channel change. For FGSM, L∞ ≈ epsilon by construction — a useful sanity check.
- **L2 norm** — perturbation magnitude across the whole image. Many tiny changes can add up to a large L2 even with small L∞.
- **Attack success** — currently defined as "top prediction changed." This is the right definition for an arbitrary-image demo (no ground truth available), but it's not the same as proving a *correct* prediction became *incorrect*.

## Immediate next task

Finish the side-by-side Matplotlib visualization (original / adversarial / perturbation heatmap + metrics) and validate it in `test_fgsm.py` before starting PGD.
