---
title: "GST-Recon-Env: GST Invoice Reconciliation RL Environment"
emoji: ":page_facing_up:"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# GST-Recon-Env

GST-Recon-Env is a Docker-based OpenEnv and FastAPI environment for GST invoice reconciliation, mismatch detection, and ITC decision-making.

## Endpoints

- `/`
- `/docs`
- `/tasks`
- `/state`
- `/reset`
- `/step`

## Notes

- This Space exposes an API, not a traditional frontend page.
- The root endpoint returns JSON health status.
- Interactive API testing is available at `/docs`.
