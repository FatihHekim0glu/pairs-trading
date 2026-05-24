---
title: pairs-trading
emoji: ""
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# pairs-trading on Hugging Face Spaces

This Space mirrors the Streamlit demo for the `pairs-trading` project. The
container is built from `deploy/hf-spaces/Dockerfile` and serves the app on
port 7860.

## Local preview

```bash
docker build -t pairs-trading-space -f deploy/hf-spaces/Dockerfile .
docker run --rm -p 7860:7860 pairs-trading-space
```

Once the container is running, open <http://localhost:7860> in a browser.

## Configuration

Runtime settings are read from `PAIRS_*` environment variables (see
`pairs.RuntimeSettings`). To pin a deterministic seed for a deployment, set
`PAIRS_DEFAULT_SEED`. To restrict to cached data only, set `PAIRS_OFFLINE=1`.

## Notes

- The image installs the `app` optional dependency group only; tests and docs
  tooling are intentionally excluded to keep the image small.
- Application code (`app/streamlit_app.py`) is owned by the app agent and is
  not provided by this directory.
