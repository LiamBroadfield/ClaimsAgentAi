# ClaimsAgentAi
My first Ai project. Built within Python, it generates and analyzes claims by message and analyzes urgency using an inexpensive LLM. Eventually, this turned into a project I completed for my company on a different platform. 

# AI-Assisted Auto Claims Triage Pipeline

A prototype pipeline for an auto insurance claims workflow: sorting inbound
messages, classifying urgency, drafting claim briefings, and
scheduling follow-ups — built as a personal project to explore how an LLM
can assist (not replace) a human adjuster's judgment on routine claims work.

This is a **synthetic-data prototype**, not a production system. All claim
data is generated locally.

## Why this exists

I'm an auto insurance claims adjuster exploring a transition into data
analytics / business analysis. This project started as a way to apply that
domain experience to a real technical build — the business rules throughout
(coverage disclosure limits, UMPD logic, total-loss handling, rental
eligibility, etc.) come from real claims-handling practice, generalized and
anonymized for a synthetic dataset.

## Pipeline overview

```
generate_v7.py → synthetic claims + message data (CSV)
↓
step3_claim_briefings.py → LLM-generated claim summaries & FNOL briefings
task_queue_v8.py → per-message urgency/priority classification
followup_scheduler.py → deterministic follow-up scheduling (no LLM — pure business-day logic)
adjuster_queue_split.py → collapses task_queue_v8 output into a simple adjuster-facing view
```

Each script reads/writes CSV files and can be run independently once the
synthetic data exists.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own Azure OpenAI (or
OpenAI-compatible) credentials:

```
AZURE_OPENAI_BASE_URL=https://your-resource-name.services.ai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=your-api-key-here
```

## Usage

Run in order:

```bash
python generate_v7.py # generates the synthetic dataset
python step3_claim_briefings.py # claim-level LLM briefings
python task_queue_v8.py # per-message LLM classification
python followup_scheduler.py # deterministic follow-up scheduling
python adjuster_queue_split.py # adjuster-facing task view
```

`followup_scheduler.py` and `adjuster_queue_split.py` have no LLM cost —
they're pure Python/pandas logic.

## What this demonstrates

- **Prompt engineering for a  business domain** — classification
rules tuned against real adjuster feedback across many iterations
(urgency/priority accuracy, correct disclosure boundaries between insured
and third-party claimants, etc.)
- **Synthetic data generation with internal consistency** — claim_type,
sender identity, and business rules are generated once per claim and
shared across every message on that claim, rather than drawn
independently per row.
- **A real debugging process, not just a working demo.** Notable bugs found
and fixed along the way:
- A shared `random` stream meant adding unrelated fields silently
reshuffled the entire dataset on every regeneration — fixed by
isolating derived fields onto their own seeded RNG.
- `claim_type` was generated independently for the rules table and for
each individual message, so a claim's own messages could disagree with
its official type (and with each other). Fixed by generating it once
per claim and sharing it everywhere downstream.
- A ground-truth label (not the model) was wrong on one message template,
which looked like a classification failure until the label itself was
traced and corrected.
- `is_total_loss` was an independent coin flip unrelated to repair cost
or drivability, producing claims like "total loss, damage below the
deductible" — fixed by deriving total-loss status from repair cost and
drivability instead.

## Known limitations

- Synthetic data only — not validated against real claims volume or
distribution beyond manual spot-checking.
- Fraud/SIU detection logic is deliberately conservative (flags for human
review, never auto-escalates) and not a complete fraud model.
- No live data integration — this is an offline batch pipeline, not a
connected production service.
- Business-day math doesn't account for holidays.

## Tech stack

Python, pandas, OpenAI-compatible chat completions API (tested against
Azure OpenAI).
