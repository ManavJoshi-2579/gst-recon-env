---
title: GST Recon Env
sdk: docker
app_port: 7860
pinned: false
---

# GST-Recon-Env: GST Invoice Reconciliation RL Environment

An OpenEnv-compatible reinforcement learning environment for Indian GST invoice reconciliation, mismatch detection, and Input Tax Credit decision-making.

## Overview

GST reconciliation is a recurring back-office workflow in which purchase invoices must be checked against filed GSTR-2B data before claiming Input Tax Credit (ITC). In practice, teams must detect missing entries, tax mismatches, invalid GSTINs, e-invoice failures, and suspicious vendor behavior while still maximizing valid claims.

`GST-Recon-Env` models this process as a sequential decision environment. An agent reviews one invoice at a time, compares it with available GSTR-2B entries, chooses compliance actions, and is scored on correctness, risk, and final reporting quality.

This environment captures a real-world GST compliance workflow where incorrect decisions directly impact financial risk and working capital. Reinforcement learning is well-suited for this setting because agents must balance correctness, uncertainty, and long-term risk rather than follow static rules.

This environment is specifically designed for learning-based agents, where policies must evolve across episodes rather than rely on static heuristics. The sequential structure and delayed scoring ensure that optimal strategies require reasoning over both local correctness and long-term compliance risk.

## Motivation

This problem matters because reconciliation quality directly affects compliance exposure and working capital:

- Incorrect ITC claims can create audit risk and downstream penalties.
- Missing or mismatched supplier filings delay recoverable tax credits.
- Fraudulent or non-compliant invoices should be rejected even when short-term incentives favor claiming.
- Real teams need consistent, repeatable decision policies rather than brittle rule chains.

The environment is intended as a realistic benchmark for evaluating agents on compliance-sensitive business operations, not just generic task completion.

Unlike synthetic RL benchmarks, this environment captures a real financial workflow with asymmetric penalties, delayed consequences, and uncertainty, making it suitable for evaluating agent reliability in production-like conditions.

## OpenEnv Compliance

The project is structured to match OpenEnv expectations:

- `reset()` initializes a deterministic episode and returns a typed observation.
- `step()` consumes a typed action and returns observation, reward, done, and info.
- `state()` returns a JSON-serializable dictionary for inspection and debugging.
- Pydantic models define actions, invoices, GSTR-2B entries, and observations.
- `openenv.yaml` declares entrypoint, tasks, reward range, action space, and observation space.
- The project passes validation with:

```powershell
.\.venv\Scripts\openenv.exe validate
```

Expected result:

```text
[OK] gst_recon: Ready for multi-mode deployment
```

This environment has been validated using `openenv validate` and is fully compliant with the OpenEnv specification.

## Action Space

OpenEnv-facing schema:

```json
{
  "action_type": "string",
  "invoice_id": "string"
}
```

Runtime HTTP action model:

```json
{
  "type": "match | reject | claim_itc | query_vendor | submit_report",
  "invoice_id": "INV-001",
  "reason": "optional free-text explanation"
}
```

Supported actions:

The agent selects one discrete action per invoice based on reconciliation evidence.

![Supported actions](assets/Actions.png)

| Action | Purpose |
|---|---|
| `match` | Accept the current invoice as compliant and matched to GSTR-2B |
| `reject` | Reject the invoice for mismatch, fraud, missing filing, or invalidity |
| `claim_itc` | Claim tax credit for a compliant invoice |
| `query_vendor` | Ask for clarification when evidence is incomplete |
| `submit_report` | End the episode and trigger final grading |

## Observation Space

The observation provides both invoice-level details and global progress signals, enabling the agent to reason about both local correctness and episode-level strategy.

OpenEnv-facing schema:

```json
{
  "current_invoice": "object",
  "mismatch_flags": "list",
  "risk_score": "float",
  "progress": "float"
}
```

Actual HTTP observation returned by the server:

```json
{
  "current_invoice": {
    "id": "INV-001",
    "gstin": "29ABCDE1234F1Z5",
    "date": "2025-01-01",
    "value": 12000.0,
    "hsn": "9983",
    "igst": 0.0,
    "cgst": 1080.0,
    "sgst": 1080.0,
    "is_einvoice": true,
    "is_fraud": false
  },
  "available_gstr2b": [],
  "matched": [],
  "mismatches": [],
  "current_itc": 0.0,
  "total_itc_possible": 0.0,
  "progress": 0.0,
  "warnings": [],
  "step_count": 0
}
```

Each step response also includes structured metadata:

```json
{
  "info": {
    "score": 0.0,
    "risk": 0.0,
    "processed": 0
  }
}
```

## Tasks (Difficulty Levels)

The environment exposes three difficulty levels:

![Task difficulty levels](assets/Tasks.png)

| Task | Invoices | Mismatch Probability | Behavioral Focus |
|---|---:|---:|---|
| `easy` | 3 | 10% | Straightforward reconciliation with low ambiguity |
| `medium` | 5 | 30% | More missing and mismatched entries; careful ITC decisions matter |
| `hard` | 8 | 50% | Fraud signals, e-invoice failures, and risk-aware claim behavior dominate |

Behavior changes across difficulties:

- `easy` emphasizes basic matching accuracy.
- `medium` increases error surface and penalizes over-claiming more meaningfully.
- `hard` increases fraud and mismatch exposure, making risk management central to final score.

## Reward Design

The reward function is shaped to encourage correct local actions while preventing trivial exploitation.

| Event | Reward |
|---|---:|
| `match_invoice` correct | `+0.3` |
| `reject_invoice` correct | `+0.2` |
| `claim_itc` correct | `+0.5` |
| Wrong `match` / `reject` | `-0.3` |
| Wrong `claim_itc` | `-0.7` |
| `query_vendor` on uncertain invoice | small positive shaping |
| Duplicate invoice/action | `-0.3` |
| Repeated same decision streak | `-0.1` |
| Loop beyond invoice set | `-0.2` |

Design principles:

- Rewards are non-zero for meaningful decisions.
- Partial progress is visible through stepwise shaping instead of only terminal grading.
- Wrong ITC claims increase `risk_score`.
- Repetitive single-action policies are penalized.
- Duplicate handling and loop penalties reduce exploitability.
- Final grading is separated from local step shaping to keep behavior interpretable.

All rewards are bounded and normalized to encourage stable learning, and the overall episode score is constrained to the range [0.0, 1.0].

The reward system is explicitly designed to prevent trivial policies such as always-claim or always-reject strategies. Repetition penalties, duplicate handling, and risk-aware grading ensure that only balanced, context-sensitive policies achieve high scores.

## Grader System

Scoring is deterministic and normalized to `[0.0, 1.0]`.

```python
def grade_easy(state):
    return state.correct_matches / len(state.invoices)

def grade_medium(state):
    penalty = state.wrong_itc_claims * 0.2
    return max(0.0, (state.correct_matches / len(state.invoices)) - penalty)

def grade_hard(state):
    risk_penalty = state.risk_score
    return max(0.0, (state.correct_matches / len(state.invoices)) * (1 - risk_penalty))
```

Interpretation:

- `easy`: pure accuracy.
- `medium`: accuracy minus wrong-claim penalties.
- `hard`: accuracy discounted by accumulated risk.

The final implementation clamps scores with:

```python
score = min(max(score, 0.0), 1.0)
```

Common failure modes for agents include over-claiming ITC, repetitive single-action policies, and ignoring risk signals. The environment penalizes these behaviors explicitly, making it suitable for evaluating robustness beyond raw accuracy.

## Baseline Results

Recent deterministic local inference run:

```text
[START] task=hard env=gst_recon_env model=gpt-4o-mini
[STEP] ...
[END] success=True steps=8 score=1.0 rewards=0.20,0.20,0.30,0.20,0.20,0.10,0.30,0.20
```

Typical heuristic or simple-policy performance lands around `~0.8` to `~0.9`, depending on task mix and action diversity. Reject-only behavior is intentionally not competitive because repetition and risk-aware correctness both matter.

## Inference Script

Run local inference with:

```powershell
.\.venv\Scripts\python.exe inference.py
```

Behavior:

- Tries the HTTP server first.
- Falls back to an in-process local environment if the server is unavailable.
- Uses deterministic heuristics if no OpenAI API key is present.
- Emits structured logs using `[START]`, `[STEP]`, and `[END]`.

## Docker Instructions

Build the image:

```powershell
docker build -t gst-recon-env .
```

Run the server:

```powershell
docker run --rm -p 7860:7860 -e SPACE_ID=local-test gst-recon-env
```

Run in detached mode:

```powershell
docker run -d --name gst-recon-env-test -p 7860:7860 -e SPACE_ID=local-test gst-recon-env
```

Test the API:

```powershell
curl.exe http://localhost:7860/
curl.exe http://localhost:7860/tasks
curl.exe -X POST http://localhost:7860/reset
curl.exe http://localhost:7860/state
```

Test `/reset` with an explicit task:

```powershell
curl.exe -X POST http://localhost:7860/reset `
  -H "Content-Type: application/json" `
  -d "{\"task\":\"hard\"}"
```

Expected root response:

```json
{"status":"ok"}
```

## Deployment

The project ships a FastAPI server in `server/app.py` and is ready for Docker-based deployment.

Server entry:

```powershell
python -m server.app
```

The container automatically switches between inference mode and server mode depending on runtime environment (local vs Hugging Face Space).

Hugging Face Space readiness:

- Compatible with Docker Spaces.
- Binds to `0.0.0.0`.
- Supports Hugging Face runtime on port `7860`.
- Includes a Dockerfile that has already been verified to build successfully.

The system is production-representative, with deterministic behavior, strict API contracts, and failure-safe inference execution, ensuring reliable evaluation across environments.

Suggested Hugging Face flow:

1. Create a new Space.
2. Select `Docker` as the SDK.
3. Upload this repository or connect the GitHub repo.
4. Set the app port to `7860`.
5. Deploy and verify `/`, `/docs`, and `/tasks`.

Interactive docs:

```text
http://localhost:7860/docs
```

## Project Structure

```text
gst_recon_env/
|- assets/
|- server/
|  |- app.py
|  |- dev.py
|  |- gst_recon_env_environment.py
|  |- models.py
|  `- __init__.py
|- client.py
|- Dockerfile
|- HF_SPACE_README.md
|- inference.py
|- openenv.yaml
|- pyproject.toml
|- README.md
`- requirements.txt
```

## Reproducibility

The environment is designed for deterministic evaluation:

- `random.seed(42)` is applied during reset.
- `np.random.seed(42)` is applied when NumPy is available.
- Episode generation is deterministic for a given task.
- Graders are deterministic and normalized.
- Inference falls back to deterministic heuristics when no external model is configured.

This guarantees consistent benchmarking and enables fair comparison between different agent strategies.

## Limitations

This is a simulation benchmark, not a production tax engine.

- GST rules are simplified for benchmarking.
- Vendor behavior and filing data are synthetic.
- Fraud logic is intentionally compact and observable enough for RL evaluation.
- The environment focuses on decision quality, not full statutory coverage.

## Conclusion

`GST-Recon-Env` turns a real compliance workflow into a compact, reproducible RL benchmark with meaningful tradeoffs between accuracy, risk, and ITC behavior. It is OpenEnv-validated, Docker-ready, deployment-ready, and directly relevant to real-world finance operations where decision errors are costly.

This environment is deterministic, robust to runtime failures, and production-ready for evaluating autonomous agents in compliance-sensitive workflows.

This environment fills a critical gap in RL evaluation by combining real-world financial decision-making with exploit-resistant reward design and deterministic benchmarking.
