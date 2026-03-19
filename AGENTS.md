# AGENTS.md

## Project Overview
This repository is for implementing a mature, experiment-ready object detection codebase for a custom research model designed for small object detection in foggy and complex weather conditions.

The target architecture is:

- AMFE-Backbone (Asymmetric Multi-branch Feature Enhancement Backbone)
- AMF-Neck (Attention-guided Multi-scale Fusion Neck)
- Ultralytics YOLO Detect Head

This project is research-oriented, but the implementation standard must be engineering-grade, stable, modular, readable, and testable.

---

## Primary Goal
Implement a production-quality experimental codebase that can:

1. Build the full model:
   - AMFE-Backbone
   - AMF-Neck
   - Ultralytics Detect Head

2. Run forward passes correctly with expected tensor shapes.

3. Integrate with the Ultralytics training / validation / loss / export pipeline as much as possible.

4. Support later training on public object detection datasets.

5. Be maintainable and easy to debug.

---

## Hard Constraints

### Architecture Constraints
You MUST follow the architecture specification from the project documents.

Do NOT redesign the model on your own.

Do NOT replace the detection head with a custom head.

Do NOT change the core paper naming unless absolutely necessary for code consistency.

The model must follow this high-level structure:

Input
→ LEM
→ DPS Stem
→ Shared Shallow Feature
→ MSB + ADB + LGCB
→ MBFM
→ F3 / F4 / F5
→ AMF-Neck
→ N3 / N4 / N5
→ Ultralytics YOLO Detect Head

---

### Detect Head Constraints
The detection head is NOT a research innovation in this project.

Therefore:

- Do NOT redesign the YOLO Detect Head.
- Do NOT replace Ultralytics Detect with another detection head.
- Do NOT introduce a custom loss design unless explicitly required later.
- Do NOT change the prediction principle of the head.

It is acceptable to do interface adaptation only, including:
- channel matching
- model registration
- configuration wiring
- connecting N3 / N4 / N5 into Detect

---

### Engineering Constraints
The code must be mature and conservative.

Always prefer:
- correctness
- readability
- modularity
- testability
- compatibility with Ultralytics
- clear failure modes

Avoid:
- clever but fragile code
- unnecessary abstraction
- hidden shape assumptions
- silent fallback logic
- undocumented magic constants

---

## Required Module Names
Please preserve these names in code, comments, and docs as much as possible.

### Backbone
- AMFEBackbone
- LEM
- DPSStem
- MSB
- ADB
- DEB
- LGCB
- MBFM
- CDG

### Neck
- AMFNeck
- CAF
- TDSF
- BURF
- SPG
- DPG
- MCA

If a Python class/function name must be adapted to style conventions, keep the paper name in comments and docstrings.

---

## Architecture Specification

### 1. AMFE-Backbone
AMFE-Backbone means Asymmetric Multi-branch Feature Enhancement Backbone.

Its internal logic must follow this idea:

- LEM performs lightweight low-level enhancement without image restoration.
- DPS Stem replaces aggressive early downsampling and should preserve more small-object detail.
- MSB is the main semantic branch based on ResNet50-style semantic feature extraction.
- ADB is the auxiliary detail branch for local detail compensation.
- LGCB provides lightweight global context enhancement from deep semantic features.
- MBFM fuses semantic, detail, and context features into final backbone outputs.
- Final backbone outputs must be:
  - F3: stride 8
  - F4: stride 16
  - F5: stride 32

### 2. AMF-Neck
AMF-Neck means Attention-guided Multi-scale Fusion Neck.

Its internal logic must follow this idea:

- Accept F3 / F4 / F5 from the backbone.
- First perform channel alignment.
- Then perform top-down selective fusion.
- Then perform bottom-up refinement fusion.
- Output:
  - N3: stride 8
  - N4: stride 16
  - N5: stride 32

Top-down and bottom-up logic must be treated differently.

Do NOT collapse the neck into a plain FPN/PAN clone.

### 3. Detect Head
Use Ultralytics YOLO Detect Head directly.

No structural redesign.

---

## Design Principles

### Principle 1: Clear role separation
Backbone should focus on producing stronger single-scale features.

Neck should focus on better cross-scale information fusion.

Detect head should remain stable and reused.

### Principle 2: Lightweight where possible
This project is not intended to become a huge heavy architecture.

Do not add extra heavy attention blocks everywhere.

Do not duplicate functionality already provided by another stage.

### Principle 3: Conservative integration
Reuse Ultralytics components whenever reasonable:
- trainer
- validator
- loss
- predictor
- export path
- config structure

Only customize what is necessary for the new backbone and neck.

---

## Expected Implementation Style

### Code Quality
All new modules must have:
- clear docstrings
- type hints where practical
- comments for tensor shape assumptions
- readable variable names
- explicit input/output behavior

### Shape Safety
Shape assumptions must be guarded.

Add assertions or explicit checks where appropriate for:
- stride consistency
- feature list lengths
- channel compatibility
- spatial alignment before add / fuse

### Minimal Surprise
If a module expects specific feature order, make it explicit.

Do not rely on undocumented implicit contracts.

---

## Testing Requirements
Testing is mandatory.

At minimum, add tests for:

1. Module-level forward pass
   - LEM
   - DPSStem
   - DEB
   - LGCB
   - MBFM
   - TDSF
   - BURF

2. Backbone integration
   - input image tensor
   - output F3 / F4 / F5 shapes

3. Neck integration
   - input F3 / F4 / F5
   - output N3 / N4 / N5 shapes

4. Full model smoke test
   - forward pass through backbone + neck + detect head

5. Minimal training integration smoke test
   - one synthetic batch
   - forward
   - loss
   - backward

Tests must be lightweight and runnable without real dataset downloads.

Use synthetic tensors where possible.

---

## Repository and File Change Policy

### Before making large changes
For any non-trivial task, first produce a plan.

The plan should include:
- files to create
- files to modify
- implementation order
- possible risks
- test plan

### After each implementation phase
Always summarize:
- what files changed
- what public APIs were introduced
- what assumptions were made
- what tests were added
- what remains unfinished

### If there is a conflict
If the architecture spec conflicts with the existing repository structure:
- do NOT silently improvise
- clearly state the conflict
- propose the smallest safe solution

---

## Environment Expectations
Assume the target environment should be stable and reproducible.

Preferred baseline:
- Python 3.10
- PyTorch stable version
- CUDA-compatible environment when available
- Ultralytics installed from a controlled version

When adding dependencies:
- keep them minimal
- justify why they are needed
- avoid obscure libraries
- prefer PyTorch / torchvision / Ultralytics ecosystem tools

---

## Preferred Directory Intent
You may adapt to the existing repository, but prefer a structure like:

- models/
  - backbone/
  - neck/
  - common/
  - detector/
- tests/
- configs/
- docs/

If the repository already has an Ultralytics-compatible structure, integrate cleanly instead of duplicating framework code.

---

## What NOT to Do

Do NOT:
- redesign the whole project architecture without being asked
- replace Ultralytics Detect head
- introduce a Transformer-based full rewrite
- invent new research modules not in the project spec
- rewrite Ultralytics internals unless necessary
- change too many subsystems at once
- skip tests
- use placeholder code and present it as finished
- hardcode dataset paths
- assume internet access during runtime
- silently ignore shape mismatches

---

## What To Do First
When assigned a large task, do this first:

1. Read this file.
2. Read the project specification document(s).
3. Inspect the repository structure.
4. Produce an implementation plan.
5. Wait for approval if the task explicitly asks for planning first.
6. Then implement in small, reviewable phases.

---

## Task Execution Priorities
When making tradeoffs, prioritize in this order:

1. Correctness
2. Compatibility with project architecture
3. Stability
4. Testability
5. Readability
6. Performance optimization
7. Extra elegance

---

## Output Expectations
For every meaningful task, produce:

1. A concise summary of what was done
2. Changed files
3. Key implementation decisions
4. Tests added or updated
5. Any unresolved issues or risks

Do not claim the code is complete if any critical part is still missing.

Be explicit and honest about unfinished work.

---

## Final Reminder
This repository is intended to support a real research experiment.

The implementation must be robust enough that a researcher can:
- inspect the modules clearly
- run the model
- debug failures
- train with confidence
- extend experiments later

Favor mature engineering over flashy novelty.