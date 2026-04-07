# GST-Recon-Env

GST-Recon-Env is an OpenEnv benchmark for Indian GST invoice reconciliation. Agents review purchase invoices against GSTR-2B entries, identify mismatches and fraud signals, claim eligible Input Tax Credit (ITC), and submit a final compliance report.

The environment is designed for reinforcement learning and agent evaluation: rewards are state-driven, task graders are deterministic, and repeated single-action strategies are penalized.

## Highlights

- OpenEnv validation passes: `openenv validate`
- Deterministic task graders for easy, medium, and hard modes
- State-driven step logic: rewards use the current invoice, not the submitted `invoice_id`
- Risk-aware ITC dynamics: wrong ITC claims increase `risk_score` and reduce final score
- Anti-exploit scoring: always-reject and repeated-action policies are penalized
- Structured step metadata: `score`, `risk`, and `processed`
- FastAPI server mode plus local inference fallback

## Tasks

![alt text](image.png)		
Task 	Invoices 	Mismatch Rate	Focus
Easy	3	0.1	Basic invoice and GSTR-2B matching
Medium	5	0.3	Mismatches, missing entries , ITC care 
 Hard	8	0.5	Fraud signals ,e -invoice failures , risk-aware claims 



## Actions

| Action | Correct Reward | Wrong Reward | Notes |
|--------|----------------|--------------|-------|
| `match` | `+0.3` | `-0.3` | Correct when invoice fully matches GSTR-2B and is compliant |
| `reject` | `+0.2` | `-0.3` | Correct for fraud, missing, mismatched, or non-compliant invoices |
| `claim_itc` | `+0.5` | `-0.7` | Wrong claims increase `risk_score` |
| `query_vendor` | shaped | shaped | Useful for uncertain invalid invoices |
| `submit_report` | accuracy based | risk penalized | Ends the episode |

Additional penalties:

- Duplicate invoice/action: `-0.3`
- Repeated same decision pattern: `-0.1`
- Loop beyond invoice set: `-0.2`
- Dominant single-action strategy: final score penalty

## Deterministic Graders

The final score is task-specific:

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

In the implementation, these are exposed as `grade_easy()`, `grade_medium()`, `grade_hard()`, and `_calculate_grader_score()` on `GSTReconEnv`.

## Observation

Each observation contains:

- `current_invoice`
- `available_gstr2b`
- `matched`
- `mismatches`
- `current_itc`
- `total_itc_possible`
- `progress`
- `warnings`
- `step_count`

HTTP step responses also include:

```json
{
  "info": {
    "score": 0.0,
    "risk": 0.0,
    "processed": 1
  }
}
```

## Quickstart

Install dependencies:

```powershell
uv sync
```

Validate OpenEnv readiness:

```powershell
.\.venv\Scripts\openenv.exe validate
```

Expected:

```text
[OK] gst_recon: Ready for multi-mode deployment
```

Run inference locally. If no server is running, `inference.py` falls back to the local environment:

```powershell
.\.venv\Scripts\python.exe inference.py
```

Run server mode:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then in another terminal:

```powershell
.\.venv\Scripts\python.exe inference.py
```

## Current Validation Snapshot

Recent local checks:

```text
inference.py score: ~0.9
reject-only policy max score in sample: 0.46
task mismatch rates: easy ~0.09, medium ~0.27, hard ~0.43
openenv validate: [OK] gst_recon: Ready for multi-mode deployment
```

These checks show that the baseline can score strongly while a lazy always-reject strategy is not competitive.

## Deployment

The project exposes the required server entry point:

```toml
[project.scripts]
server = "server.app:main"
```

For Hugging Face or OpenEnv deployment, validate first, then launch the FastAPI server with the command above.
