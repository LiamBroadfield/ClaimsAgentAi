"""
Task queue (per-message classification) — v3 of the merged pipeline.

This replaces the earlier "classify per claim" approach with "classify per
task/message." The earlier version had a real flaw: with ~9 messages per
claim and ~18% of them being urgent-callback content, ~84% of claims had
at least one such message purely by chance — and since one classification
covered the WHOLE claim, that single message dragged every other routine
task on the same claim into "call_needed" too.

This version judges each actionable message on its own (still using the
claim's status fields as context, so it's not judging blind) — so a claim
with 8 routine messages and 1 callback request now correctly produces 8
email_sufficient tasks and 1 call_needed task, instead of 9 call_needed
tasks.

Everything else built this session carries over:
  - TL/attorney routing is a HARD RULE enforced in code, not model judgment
  - UMPD only ever pertains to the 1st-party insured, never a 3rd party
  - UMPD rental coverage is phrased as state-dependent, not flat yes/no
  - Customers without rental coverage are told about the discounted rate
  - Out-of-network shops never require upfront payment for repairs
  - Attorney representation on auto claims is treated as uncommon
  - A shop supplement implies the customer is already using that shop
  - Missing/unclear intake info defaults toward a preliminary email, not
    automatic urgency
  - "Is my tow covered" questions get answered directly, not referred
  - Every classification includes a party_type (1st_party / 3rd_party) flag

USAGE: run in the same folder as:
    synthetic communications v7.csv
    synthetic claim business rules v7.csv
"""

import json
import random
import pandas as pd
from OpenAI import OpenAI

COMMS_PATH = "synthetic communications v7.csv"
RULES_PATH = "synthetic claim business rules v7.csv"
N_TASKS_TO_CLASSIFY = 5  # sample size
MODEL = "claude-sonnet-4-5"

client = OpenAI(
    base_url=os.environ.get("AZURE_OPENAI_BASE_URL"),
    api_key=os.environ.get("AZURE_OPENAI_BASE_KEY"),
)
random.seed(21)

PRIORITY_ORDER = ["low", "medium", "high", "urgent"]
NO_TASK_CATEGORIES = {"closed_loop_confirmation", "document_dropoff", "photo_submission"}


def fake_phone():
    return f"800-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


def bump_priority(priority: str, levels: int = 1) -> str:
    idx = PRIORITY_ORDER.index(priority) if priority in PRIORITY_ORDER else 0
    return PRIORITY_ORDER[min(idx + levels, len(PRIORITY_ORDER) - 1)]


## logic

SHARED_GUIDANCE = """
Output these separate judgments — do not conflate them:
1. "review_priority": low / medium / high — does an adjuster need to look
   at THIS SPECIFIC MESSAGE soon? Judge the message itself, not the whole
   claim's history.
2. "contact_method": one of "call_needed", "email_sufficient",
   "no_action_needed", "refer_total_loss_adjuster", "refer_attorney_desk".
3. "party_type": "1st_party" or "3rd_party" — which party this message
   concerns.

Be conservative with call_needed — it should be the exception, not the
default. Most messages are routine (a coverage question, a supplement
update, a document follow-up) and should be email_sufficient. Reserve
call_needed specifically for: THIS message explicitly asking for a
callback, clear frustration/escalation language IN THIS message, or a
situation genuinely too complex to resolve in writing. A claim having
other unrelated messages elsewhere is not relevant — judge only what's in
front of you.

Corrections to apply:
- Only classify as refer_attorney_desk if the claim's reassignment_type
  field explicitly says "attorney" — not from message content alone.
  Attorney representation on auto claims is uncommon overall and usually
  tied to an injury, not pure property damage.
- Only use refer_total_loss_adjuster if is_total_loss is explicitly true.
- A non-drivable vehicle alone is NOT a red flag — most are towed from the
  scene already; treat it as neutral information.
- Tow coordination for an in-network shop is a separate team's job, never
  an adjuster call reason. Out-of-network TOW cost is customer-reimbursed
  via email. Out-of-network REPAIR cost is NEVER paid upfront by the
  customer — never imply otherwise, that's false and a legitimate
  complaint if ever told to a customer.
- Rental eligibility: liability claims mean a third party is being paid
  out and DOES qualify for a rental. A first-party insured who is
  attorney-represented but lacks collision/comprehensive coverage likely
  has NO rental available.
- UMPD (uninsured/underinsured motorist property damage) coverage belongs
  ONLY to the first-party insured — never something a third party
  "carries" or lacks. UMPD rental coverage sometimes applies and is
  state-dependent — phrase it as "may be covered depending on your state"
  rather than a flat yes/no.
- If a customer doesn't carry rental coverage, mention the company's
  discounted rental rate as an option.
- A shop submitting a supplement means the customer is already using that
  shop — never suggest shop-routing for these.
- Missing/unclear info should default toward a preliminary retrieval
  email, not automatic high priority or a forced call.
- "Is my tow covered" questions (including ones labeled roadside
  assistance) are answered directly: covered under 1st-party coverage or
  as a claimant, paid directly in-network, reimbursed out-of-network. This
  is email_sufficient, not a referral.
- If the sender is a third-party claimant (TPC), never share coverage or
  deductible details.
"""

TASK_SYSTEM_PROMPT = f"""You are triaging a single inbound message on an auto insurance \
claim for an adjuster's task queue. You'll get the specific message, who \
sent it and how, plus the claim's current status fields for context.
{SHARED_GUIDANCE}
Return ONLY a JSON object, no preamble, no markdown fences:
{{
  "review_priority": "<low, medium, or high>",
  "contact_method": "<call_needed, email_sufficient, no_action_needed, \
refer_total_loss_adjuster, or refer_attorney_desk>",
  "party_type": "<1st_party or 3rd_party>",
  "reasoning": "<one short sentence>"
}}
"""


def classify_task(message_row: pd.Series, rules_row: pd.Series) -> dict:
    lines = [f"MESSAGE from {message_row['sender_type']} via {message_row['channel']}:",
             f'  "{message_row["message_body"]}"',
             "\nCLAIM STATUS FIELDS (for context only):"]
    for field in ["claim_type", "reassignment_type", "is_umpd_claim", "vehicle_drivable",
                  "inspection_method", "has_rental", "is_total_loss", "potential_issue_flag"]:
        if field in rules_row and pd.notna(rules_row[field]) and rules_row[field] != "":
            lines.append(f"  - {field}: {rules_row[field]}")
    if rules_row.get("is_total_loss"):
        lines.append(f"  - total_loss_adjuster_contact: {fake_phone()}")
    if rules_row.get("reassignment_type") == "attorney":
        lines.append(f"  - attorney_desk_contact: {fake_phone()}")

    context = "\n".join(lines)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TASK_SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ],
        temperature=0
        max_tokens=350,
    )
    raw = response.content[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"review_priority": "medium", "contact_method": "email_sufficient",
                "party_type": "1st_party", "reasoning": "PARSE_ERROR — defaulted safely"}


## tasks

def build_base_task_queue(comms: pd.DataFrame) -> pd.DataFrame:
    tasks = []
    for _, msg in comms.iterrows():
        category = msg["true_category"]
        needs_task = (
            msg["requires_fallback_task"]
            or category == "urgent_callback_request"
            or (not msg["auto_respond_eligible"] and not msg["is_referral"]
                and category not in NO_TASK_CATEGORIES)
        )
        if not needs_task:
            continue

        priority = msg["true_urgency"] if msg["true_urgency"] in PRIORITY_ORDER else "low"
        if msg["priority_boost_called_in"]:
            priority = bump_priority(priority)
        if category == "urgent_callback_request":
            priority = "urgent"

        tasks.append({
            "task_id": f"TASK{len(tasks) + 1:05d}",
            "claim_id": msg["claim_id"],
            "source_message_id": msg["message_id"],
            "sender_type": msg["sender_type"],
            "channel": msg["channel"],
            "message_body": msg["message_body"],
            "category": category,
            "heuristic_priority": priority,
            "task_summary": (
                msg["fallback_task_summary"] if msg["requires_fallback_task"]
                else f"{msg['sender_type'].title()} message on {msg['claim_id']}: {category.replace('_', ' ')}"
            ),
        })
    return pd.DataFrame(tasks)


def main():
    comms = pd.read_csv(COMMS_PATH, encoding="utf-8", encoding_errors="replace")
    rules = pd.read_csv(RULES_PATH, encoding="utf-8", encoding_errors="replace")
    rules_lookup = rules.set_index("claim_id")

    base_queue = build_base_task_queue(comms)
    print(f"Base task queue (free, full dataset): {len(base_queue)} tasks "
          f"across {base_queue['claim_id'].nunique()} claims\n")

    sample_tasks = base_queue.sample(
        min(N_TASKS_TO_CLASSIFY, len(base_queue)), random_state=42
    )

    print(f"Running LLM classification on a sample of {len(sample_tasks)} individual "
          f"tasks/messages (of {len(base_queue)} total in queue)...\n")

    classifications = {}
    for i, (idx, task_row) in enumerate(sample_tasks.iterrows()):
        print(f"  Classifying {i + 1}/{len(sample_tasks)}: {task_row['task_id']} "
              f"({task_row['claim_id']})...")
        rules_row = rules_lookup.loc[task_row["claim_id"]]
        result = classify_task(task_row, rules_row)

        ## TL adjusters and repped adjusters handle seperate portions of claim
        
        if rules_row.get("is_total_loss"):
            result["contact_method"] = "refer_total_loss_adjuster"
        elif rules_row.get("reassignment_type") == "attorney":
            result["contact_method"] = "refer_attorney_desk"

        classifications[task_row["task_id"]] = result

    def resolve_row(row):
        tid = row["task_id"]
        if tid in classifications:
            c = classifications[tid]
            return pd.Series({
                "review_priority": c["review_priority"],
                "contact_method": c["contact_method"],
                "party_type": c.get("party_type", "unknown"),
                "reasoning": c["reasoning"],
                "classification_source": "llm_reviewed",
            })
        return pd.Series({
            "review_priority": row["heuristic_priority"],
            "contact_method": "unclassified",
            "party_type": "unknown",
            "reasoning": "",
            "classification_source": "heuristic",
        })

    resolved = base_queue.apply(resolve_row, axis=1)
    task_queue = pd.concat([base_queue, resolved], axis=1)

    task_queue["routed_out_of_handler_queue"] = task_queue["contact_method"].isin(
        ["refer_total_loss_adjuster", "refer_attorney_desk"]
    )

    reviewed = task_queue[task_queue["classification_source"] == "llm_reviewed"]
    print(f"\nLLM-reviewed tasks: {len(reviewed)}")
    print(f"Heuristic-only tasks: {(task_queue['classification_source'] == 'heuristic').sum()}")
    print(f"Routed out of handler queue (TL/attorney referral): "
          f"{task_queue['routed_out_of_handler_queue'].sum()}")

    print("\nContact method breakdown (LLM-reviewed subset only):")
    print(reviewed["contact_method"].value_counts())

    task_queue.to_csv("task queue per message.csv", index=False)
    print("\nSaved to task queue per message.csv")

    return task_queue


if __name__ == "__main__":
    task_queue = main()
