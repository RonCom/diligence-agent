# diligence-agent

A tool-calling agent that answers diligence questions about marketed drugs from public records (FDA Orange Book,
Drugs@FDA and labels, ClinicalTrials.gov, SEC EDGAR, PubMed) and cites a source record for every claim. Models run
locally through Ollama (`gemma4:e4b`).

```bash
uv sync
copy .env.example .env        # then set SEC_USER_AGENT to your name and email
uv run diligence-agent tools  # tool schemas the model sees
uv run diligence-agent tool orange_book '{"drug": "Trikafta"}'
uv run diligence-agent --fake tool ctgov_search '{"intervention": "elexacaftor", "phase": "PHASE3"}'
uv run pytest
```

`--fake` serves recorded responses, so tools, agents and tests run without a network.
