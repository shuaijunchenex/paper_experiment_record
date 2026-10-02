# Source audit before the experiment

This directory implements a **synthetic RBLA aggregation experiment**, not a training run.

## Repository instructions and available implementation

- Searched the project recursively, including hidden files, for `AGENTS.md`; none was found. No `H:/AGENTS.md` was present. The project `.agents` directory contained no instruction files.
- This repository is an experiment/results archive. The available Python files and notebook code contain plotting and synthetic canonicalization routines, but no training-side RBLA aggregator, rank-assignment implementation, or LoRA forward implementation. The archived entrypoint imports `entries.run_all`; that package is absent here.
- Existing uncommitted changes were present in several figure directories at the start. They were left untouched. All work for this task is in this new directory.

## RBLA aggregation

Read [`aggregation_rbla.yaml`](../../experiment/06_nlp_dirichlet_rank_correct/supporting_configs/yamls/aggregation/aggregation_rbla.yaml): `method: rbla`, `pad_mode: nan`, `is_wbab: False`, and canonicalization disabled. Its comments explicitly describe mask-normalized padding and preservation of original slot ordering. The corresponding [`aggregation_sp_plus.yaml`](../../experiment/06_nlp_dirichlet_rank_correct/supporting_configs/yamls/aggregation/aggregation_sp_plus.yaml) enables singular-value canonicalization with `pad_mode: nan`.

`core.aggregate_rbla` implements exactly the supplied per-index rule: for zero-based slot s, include clients with rank > s; normalize their original p_i by their participation-weight sum; average B columns and A rows separately. This agrees with the archived mask-normalization description. An independent padded-array/einsum implementation verifies the formula with nonuniform client weights. A separate check demonstrates that the resulting BA differs from the weighted average of dense client BA updates.

No training implementation is available to verify its actual weighting, dtype, or scaling line-by-line; therefore this report does **not** claim bitwise equivalence to a training-side runtime. There is no observed conflicting aggregation rule to explain beyond this provenance limitation.

## Rank assignment

Read the archived MRPC [rank configuration](../../experiment/06_nlp_dirichlet_rank_correct/supporting_configs/yamls/rank_distribution/het_rank_mrpc_dirichlet_0_4_10_correct.yaml), which maps data volume to a rank ratio. Its training implementation is not included. This experiment uses the user's fixed ranks and uniform weights, with no data-volume association. R=16 and R=64 ranks use half-up rounding of the R=32 proportions; the final client reaches R in both cases. No inference about rank-data mismatch is made.

## LoRA scaling

Read [`base_lora_inference.yaml`](../../experiment/06_nlp_dirichlet_rank_correct/supporting_configs/yamls/lora_inference/base_lora_inference.yaml), which contains `scaling: 0.5` and `alpha: 0`, and [`transformer_tinybert_lora.yaml`](../../experiment/06_nlp_dirichlet_rank_correct/supporting_configs/yamls/nn_model/transformer_tinybert_lora.yaml), which contains `lora_r: 8` and `lora_alpha: 16`. Those configuration values are not enough to establish actual training-side effective scaling without its forward code.

The synthetic path never loads a training LoRA layer or those configuration values. Its effective update is exactly BA, with multiplier 1 for every client and redistributed factor pair. The reused NumPy canonicalization function likewise has no alpha/r or other scaling. This explicitly establishes **synthetic effective scaling=1** while leaving the participation-normalized B/A averaging rule unchanged. The Gaussian entry standard deviations define the synthetic input distribution, not an extra LoRA forward multiplier.

## Reused SP+ implementation and figure style

`reference_sp_plus.py` freezes the `compact_sp_plus` function verbatim from [`../broadcast_reconstruction/plot_figure.py`](../broadcast_reconstruction/plot_figure.py), line 120 at the time of inspection. `provenance.json` stores source and function SHA-256 hashes and the repository HEAD. The source was already locally modified by the user; freezing only this function makes this experiment independent of later edits to that figure script.

The reused function performs two thin QRs, one core SVD, and square-root singular-value factor splitting. Its returned full-width factors retain tiny numerical singular directions; no true-rank truncation is applied, so all requested adapter prefixes, including numerically rank-deficient examples, remain valid. This experiment requires adapter width R < min(m,n); overcomplete adapters are outside its declared scope.

The old generator constructs an ordered spectrum and applies a random rotation. It is **not** reused. Its error helper divides by norm+epsilon; it is **not** reused either. The new measurement code explicitly identifies zero/unreliable normalization and preserves all absolute errors. No method error is replaced with the spectral optimum.

The red `#C44E52` prefix, blue `#4C72B0` SP+, black dashed reference, Times New Roman font, hidden top/right spines, and embedded PDF fonts follow the project's existing broadcast/excess-error figures. New figures use pointwise medians and IQRs, not the previous mean/std default. Actual client ranks are marked, and axis limits are data-driven without a ceiling of one.

## PDF review tooling

The configured PDF skill version was absent; the available `pdf/26.909.12148/skills/pdf/SKILL.md` was read. Its operation marker ran successfully using the installed Codex Node runtime. Scientific plots are authored with Matplotlib. The QA script uses Poppler when present and PyMuPDF rendering as an available local fallback; PDF font embedding, page text, and all rendered plots are checked separately. No external service receives experiment data.
