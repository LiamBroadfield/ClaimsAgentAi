"""
Merged step 2 + step 3: one task queue, using step 3's corrected reasoning
to drive step 2's priority and SLA timing — instead of two separate,
disconnected scripts.

How it works:
  1. Build the base task queue from messages, same as step 2 — this part
     is pure Python/pandas, free, and covers the FULL dataset.
  2. Take a SAMPLE of the unique claims that ended up in that queue and
     run them through step 3's claim-level LLM classification (the
     corrected logic: call_needed / email_sufficient / no_action_needed /
     refer_total_loss_adjuster / refer_attorney_desk, plus the missing-info
     and tow-coverage fixes). This is the only part that costs API calls,
     so it's sampled rather than run on everything.
  3. For sampled claims, their tasks get the LLM's review_priority,
     contact_method, and SLA timing appropriate to that method — including
     pulling total-loss/attorney-referral claims OUT of the claim
     handler's active queue entirely, since those shouldn't sit in the
     handler's escalation math at all.
  4. For claims NOT in the sample, tasks keep step 2's original heuristic
     priority — clearly labeled "heuristic" vs "llm_reviewed" so it's
     obvious in the output which rows got the richer treatment. Bump
     N_CLAIMS_TO_CLASSIFY up (or run this again with a different sample)
     once you're happy with how it's working, to cover more of the queue.

USAGE: same folder as synthetic_communications_v7.csv and
       synthetic_claim_business_rules_v7.csv. Run the whole file.

"""

import json
import random
import pandas as pd
from openai import OpenAI

COMMS_PATH = "synthetic communications v7.csv"
RULES_PATH = "synthetic claim business rules v7.csv"
N_CLAIMS_TO_CLASSIFY = 75  # sample size
MODEL = "gpt-4.1-mini"

client = OpenAI(
    base_url=os.environ.get("AZURE_OPENAI_BASE_URL"),
    api_key=os.environ.get("AZURE_OPENAI_BASE_KEY")
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
Output two SEPARATE judgments:
1. "review_priority": low / medium / high.
2. "contact_method": one of "call_needed", "email_sufficient", \
"no_action_needed", "refer_total_loss_adjuster", "refer_attorney_desk".

Corrections to apply:
- Non-drivable vehicle alone is NOT a red flag — most are towed from the \
scene already; treat it as neutral information.
- Tow coordination for an in-network shop is a separate team's job, never \
an adjuster call reason. Out-of-network TOW cost is customer-reimbursed \
via email — but out-of-network REPAIR cost is never paid upfront by the \
customer; that would be false to tell them.
- Rental eligibility: liability claims mean a third party is being paid \
out and DOES qualify for a rental. A first-party insured who is \
attorney-represented but lacks collision/comprehensive coverage likely \
has NO rental available.
- UMPD (uninsured/underinsured motorist property damage) coverage \
belongs ONLY to the first-party insured — it is never something a THIRD \
PARTY "carries" or lacks, and a third party is never the at-fault \
insured's UMPD counterpart. Never frame UMPD coverage status as \
belonging to or being asked about by a third-party claimant — that's a \
category error. UMPD rental coverage specifically sometimes applies and \
is state-dependent — phrase it as "may be covered depending on your \
state; we'll confirm" rather than a flat yes or no.
- If a customer doesn't carry rental coverage at all, still let them know \
the company offers a discounted rental rate as an option, even without \
that coverage.
- Always include a "party_type" field: "1st_party" if this concerns the \
insured, "3rd_party" if this concerns a third-party claimant, or "mixed" \
if the message thread contains BOTH 1st-party and 3rd-party senders — \
in that case, do not average or generalize across them; note in \
"reasoning" that this claim has multiple party types and any \
customer-facing action should be handled per-sender, not as one blended \
response.
- NEVER combine or imply shared financial details (deductible, coverage \
limit) across different sender types on the same claim. Deductible and \
coverage-limit information belongs to the 1st-party insured ONLY and \
must never be attributed to or implied for a 3rd-party claimant.
- Attorney representation on auto claims is uncommon overall and usually \
tied to an injury, not pure property damage. IMPORTANT: only classify as \
refer_attorney_desk if the claim's reassignment_type field explicitly \
says "attorney" — do NOT infer attorney representation just because a \
message mentions an attorney in passing; message content can be noisy or \
inconsistent with the claim's actual official status, and the status \
field is the source of truth.
- Similarly, only use refer_total_loss_adjuster if is_total_loss is \
explicitly true in the status fields — not from message content alone.
- Be conservative with call_needed — it should be the exception, not the \
default. Most task-queue items are routine (a coverage question, a \
supplement update, a document follow-up) and should be email_sufficient. \
Reserve call_needed specifically for: explicit "please call me" requests, \
clear frustration/escalation language, or a situation genuinely too \
complex to resolve in writing. If you're unsure, prefer email_sufficient.
- A shop submitting a supplement means the customer is already using that \
shop — never suggest shop-routing for these.
- Missing/unclear info at intake should default toward a preliminary \
retrieval email, not automatic high priority or a forced call.
- Tow-coverage questions ("is my tow covered") are answered directly \
(email_sufficient): covered under 1st-party coverage or as a claimant, \
paid directly in-network, reimbursed out-of-network.
- Total-loss and attorney-represented claims should NOT default to \
call_needed from the claim handler — route these to \
refer_total_loss_adjuster / refer_attorney_desk instead.
"""

system_prompt = f"""You are classifying an auto insurance claim for task-queue \
prioritization. You'll get the claim's status fields and, if any exist, \
its inbound messages.
{SHARED_GUIDANCE}
Return ONLY a JSON object, no preamble, no markdown fences:
{{
  "review_priority": "<low, medium, or high>",
  "contact_method": "<call_needed, email_sufficient, no_action_needed, \
refer_total_loss_adjuster, or refer_attorney_desk>",
  "party_type": "<1st_party, 3rd_party, or mixed — which party this \
concerns; use mixed if the thread has multiple sender types>",
  "reasoning": "<one short sentence>"
}}
"""


def classify_claim(claim_id: str, rules_row: pd.Series, messages: pd.DataFrame, system_prompt) -> dict:
    lines = [f"CLAIM {claim_id} STATUS:"]
    for field in ["claim_type", "reassignment_type", "is_umpd_claim", "vehicle_drivable",
                  "inspection_method", "has_rental", "is_total_loss", "potential_issue_flag"]:
        if field in rules_row and pd.notna(rules_row[field]) and rules_row[field] != "":
            lines.append(f"  - {field}: {rules_row[field]}")
    if rules_row.get("is_total_loss"):
        lines.append(f"  - total_loss_adjuster_contact: {fake_phone()}")
    if rules_row.get("reassignment_type") == "attorney":
        lines.append(f"  - attorney_desk_contact: {fake_phone()}")

    # Surface mixed-party threads explicitly so the model isn't left to
    # infer this from scanning sender_type values on its own.
    if len(messages):
        distinct_senders = messages["sender_type"].nunique()
        if distinct_senders > 1:
            lines.append(
                f"\nNOTE: this thread contains {distinct_senders} distinct "
                f"sender types ({', '.join(sorted(messages['sender_type'].unique()))}). "
                f"Treat party_type as 'mixed' and do not blend financial "
                f"details (deductible/coverage limit) across sender types."
            )
        lines.append("\nMESSAGES:")
        for _, m in messages.iterrows():
            lines.append(f"  - [{m['sender_type']}]: {m['message_body']}")
    else:
        lines.append("\n(No messages yet — this is a freshly filed claim.)")

    context = "\n".join(lines)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ],
        temperature=0,
        max_tokens=400
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"review_priority": "medium", "contact_method": "email_sufficient",
                "party_type": "unknown", "reasoning": "PARSE_ERROR — defaulted safely"}


## step 2's logic

def build_base_task_queue(comms: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
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

    base_queue = build_base_task_queue(comms, rules)
    print(f"Base task queue (free, full dataset): {len(base_queue)} tasks "
          f"across {base_queue['claim_id'].nunique()} claims\n")

    unique_claims = base_queue["claim_id"].unique().tolist()
    sample_claims = pd.Series(unique_claims).sample(
        min(N_CLAIMS_TO_CLASSIFY, len(unique_claims)), random_state=42
    ).tolist()

    print(f"Running LLM classification on a sample of {len(sample_claims)} claims "
          f"(of {len(unique_claims)} total in queue)...\n")

    classifications = {}
    for i, claim_id in enumerate(sample_claims):
        print(f"  Classifying {i + 1}/{len(sample_claims)}: {claim_id}...")
        claim_msgs = comms[comms["claim_id"] == claim_id]
        rules_row = rules_lookup.loc[claim_id]
        result = classify_claim(claim_id, rules_row, claim_msgs, system_prompt)

        ## TL adjusters and repped adjusters handle seperate portions of claim
        
        if rules_row.get("is_total_loss"):
            result["contact_method"] = "refer_total_loss_adjuster"
        elif rules_row.get("reassignment_type") == "attorney":
            result["contact_method"] = "refer_attorney_desk"

        classifications[claim_id] = result

    ## merge LLM classification 
    
    def resolve_row(row):
        cid = row["claim_id"]
        if cid in classifications:
            c = classifications[cid]
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

    # referred-out claims (total loss / attorney) don't belong in the
    # claim handler's escalation queue
    task_queue["routed_out_of_handler_queue"] = task_queue["contact_method"].isin(
        ["refer_total_loss_adjuster", "refer_attorney_desk"]
    )

    print(f"\nLLM-reviewed tasks: {(task_queue['classification_source'] == 'llm_reviewed').sum()}")
    print(f"Heuristic-only tasks: {(task_queue['classification_source'] == 'heuristic').sum()}")
    print(f"Routed out of handler queue (TL/attorney referral): "
          f"{task_queue['routed_out_of_handler_queue'].sum()}")

    print("\nContact method breakdown (LLM-reviewed subset only):")
    print(task_queue[task_queue["classification_source"] == "llm_reviewed"]["contact_method"].value_counts())

    mixed_party_claims = (task_queue[task_queue["classification_source"] == "llm_reviewed"]
                           ["party_type"] == "mixed").sum()
    if mixed_party_claims:
        print(f"\nNOTE: {mixed_party_claims} llm-reviewed task(s) flagged as 'mixed' "
              f"party_type — these claims have multiple sender types on one thread "
              f"and should not get one blended customer-facing message.")

    task_queue.to_csv("merged task queue.csv", index=False)
    print("\nSaved to merged task queue.csv")

    return task_queue


if __name__ == "__main__":
    task_queue = main()
