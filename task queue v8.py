import json
import pandas as pd
from openai import OpenAI

COMMS_PATH = "synthetic communications v7.csv"
RULES_PATH = "synthetic claim business rules v7.csv"
OUTPUT_PATH = "task_queue_v8.csv"
N_TASKS_TO_CLASSIFY = 75  # Sample size
MODEL = "gpt-4.1-mini"

## API key, can be changed out.


client = OpenAI(
    base_url=os.environ.get("AZURE_OPENAI_BASE_URL"),
    api_key=os.environ.get("AZURE_OPENAI_BASE_KEY")
)

NO_TASK_CATEGORIES = {
    "closed_loop_confirmation",
    "document_dropoff",
    "photo_submission",
}

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

SYSTEM_PROMPT = f"""You are an auto insurance workflow triage assistant reviewing ONE \
inbound message for an adjuster's task queue.
{SHARED_GUIDANCE}
Return ONLY valid JSON, no preamble, no markdown fences:

{{
  "category": "<snake_case>",
  "review_priority": "<low|medium|high>",
  "contact_method": "<call_needed|email_sufficient|no_action_needed|refer_total_loss_adjuster|refer_attorney_desk>",
  "party_type": "<1st_party|3rd_party>",
  "next_action": "<short_snake_case_action>",
  "confidence": 0.00,
  "reasoning": "<one sentence>"
}}
"""

NEXT_ACTIONS = {
    "urgent_callback_request": "call_customer",
    "supplement_signoff_needed": "review_supplement",
    "rental_coverage_confirmation": "verify_rental",
    "umpd_coverage_question": "verify_umpd_eligibility",
    "attorney_representation_notice": "reassign_claim",
    "claimant_announces_representation": "reassign_claim",
    "tpc_liability_inquiry": "review_liability",
    "deductible_inquiry": "send_deductible_info",
}


def classify_message(message_row, rules_row):
    context = f"""
MESSAGE:
{message_row['message_body']}

SENDER_TYPE: {message_row['sender_type']}
CHANNEL: {message_row['channel']}

CLAIM_CONTEXT:
claim_type={rules_row.get('claim_type')}
reassignment_type={rules_row.get('reassignment_type')}
is_total_loss={rules_row.get('is_total_loss')}
is_umpd_claim={rules_row.get('is_umpd_claim')}
vehicle_drivable={rules_row.get('vehicle_drivable')}
has_rental={rules_row.get('has_rental')}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ],
        temperature=0,
        max_tokens=300
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith('```'):
        raw = raw.strip('`')
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except Exception:
        result = {
            'category': message_row.get('true_category', 'unknown'),
            'review_priority': 'medium',
            'contact_method': 'email_sufficient',
            'party_type': '1st_party',
            'next_action': 'manual_review',
            'confidence': 0.50,
            'reasoning': 'parse_error'
        }

    ## TL adjusters and repped adjusters handle seperate portions of claim
    
    if rules_row.get('is_total_loss'):
        route_to = 'total_loss_adjuster'
        result['contact_method'] = 'refer_total_loss_adjuster'
    elif str(rules_row.get('reassignment_type', '')).lower() == 'attorney':
        route_to = 'attorney_desk'
        result['contact_method'] = 'refer_attorney_desk'
    else:
        route_to = 'handling_adjuster'

    result['route_to'] = route_to

    if not result.get('next_action'):
        result['next_action'] = NEXT_ACTIONS.get(
            result.get('category', ''),
            'manual_review'
        )

    return result


def build_task_queue(comms):
    tasks = []

    for _, msg in comms.iterrows():
        category = msg['true_category']

        needs_task = (
            msg['requires_fallback_task']
            or category == 'urgent_callback_request'
            or (
                not msg['auto_respond_eligible']
                and not msg['is_referral']
                and category not in NO_TASK_CATEGORIES
            )
        )

        if not needs_task:
            continue

        tasks.append(msg)

    return pd.DataFrame(tasks)


def main():
    comms = pd.read_csv(COMMS_PATH, encoding="utf-8", encoding_errors="replace")
    rules = pd.read_csv(RULES_PATH, encoding="utf-8", encoding_errors="replace")
    rules_lookup = rules.set_index('claim_id')

    queue = build_task_queue(comms)
    print(f"Base task queue (free, full dataset): {len(queue)} tasks qualify for review\n")

    sample_queue = queue.sample(
        min(N_TASKS_TO_CLASSIFY, len(queue)), random_state=42
    )
    print(f"Running LLM classification on a sample of {len(sample_queue)} tasks "
          f"(of {len(queue)} total qualifying)...\n")

    outputs = []

    for i, (_, row) in enumerate(sample_queue.iterrows()):
        print(f"  Classifying {i + 1}/{len(sample_queue)}: {row['message_id']} "
              f"({row['claim_id']})...")
        rules_row = rules_lookup.loc[row['claim_id']]
        result = classify_message(row, rules_row)

        combined = row.to_dict()
        combined.update(result)
        outputs.append(combined)

    result_df = pd.DataFrame(outputs)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(result_df)} tasks to {OUTPUT_PATH}")
    print("\nContact method breakdown:")
    print(result_df["contact_method"].value_counts())

    return result_df


if __name__ == '__main__':
    main()
