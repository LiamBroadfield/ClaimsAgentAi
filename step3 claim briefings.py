"""
Step 3 (v2): claim briefing for adjusters — message-thread path AND
FNOL-only path (no messages needed), with corrected business logic:

- "review_priority" (low/medium/high) is now tracked SEPARATELY from
  "contact_method" — mirrors the step 1 lesson that needing review isn't
  the same as being urgent.
- Total-loss claims are referred to a separate total-loss adjuster, not
  the claim handler — the claim handler is NOT flagged to call; instead a
  TL adjuster contact + a next-steps email are provided.
- Attorney-represented claims get similar referral-plus-email treatment,
  not necessarily a call from the claim owner.
- Tow coordination for an in-network shop is a separate team's job, not
  the adjuster's — not a call trigger. An out-of-network tow means the
  customer is emailed to submit that cost for reimbursement.
- A non-drivable vehicle is NOT by itself a red flag — only if the
  vehicle's location is undisclosed (most are towed from the scene
  already). NOTE: this dataset doesn't yet track a "location disclosed"
  field, so that specific check can't be fully applied — flagged in the
  as a known gap rather than faked.
- Coverage limits are disclosed to the customer proactively, always —
  not just when there's a potential issue.

USAGE: same folder as synthetic_communications_v5.csv and
       synthetic_claim_business_rules_v5.csv. Run the whole file.
"""

import json
import random
import pandas as pd
from openai import OpenAI

COMMS_PATH = "synthetic communications v7.csv"
RULES_PATH = "synthetic claim business rules v7.csv"
N_CLAIMS_WITH_MESSAGES = 5
N_CLAIMS_FNOL_ONLY = 5
MODEL = "gpt-4.1-mini"

client = OpenAI(
    base_url=os.environ.get("AZURE_OPENAI_BASE_URL"),
    api_key=os.environ.get("AZURE_OPENAI_BASE_KEY"))
random.seed(21)


def fake_phone():
    return f"800-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


SHARED_GUIDANCE = """
Output two SEPARATE judgments — do not conflate them:

1. "review_priority": low / medium / high — does an adjuster need to look \
at this claim soon? A routine coverage question can be medium priority \
(someone should answer it) without being high. Reserve "high" for real \
time pressure: frustration/escalation, explicit callback requests, or an \
undisclosed vehicle location.

2. "contact_method": one of:
   - "call_needed" — a live call from the assigned claim handler is \
warranted (frustration, explicit callback request, or a complex situation \
needing real conversation).
   - "email_sufficient" — a clear written answer fully addresses this.
   - "no_action_needed" — nothing outstanding.
   - "refer_total_loss_adjuster" — this is a total-loss claim; do NOT \
recommend the claim handler call. A separate total-loss adjuster owns \
this instead. Provide their contact and a next-steps email.
   - "refer_attorney_desk" — this claim is attorney-represented; do NOT \
default to a call from the claim owner. Provide the appropriate contact \
and a next-steps email instead.

Important corrections to standard assumptions:
- A non-drivable vehicle is NOT automatically a red flag or reason to \
call — most vehicles are already towed from the accident scene. Only \
treat it as needing attention if the vehicle's current location is \
unknown/undisclosed (this dataset doesn't track that field yet, so treat \
"vehicle_drivable: False" alone as neutral information, not a flag).
- Tow coordination for an IN-NETWORK shop is handled by a separate \
logistics team, not the adjuster — never list this as something the \
adjuster needs to call about. If the customer wants an OUT-OF-NETWORK \
tow, the right action is an email telling them to submit that TOW cost \
for reimbursement — not a call.
- CRITICAL: out-of-network shops do NOT require the customer to pay for \
repairs upfront. Never say or imply this to a customer — it's false and \
a legitimate reason for them to be frustrated if told otherwise. The only \
thing that differs for an out-of-network shop is the INSPECTION METHOD \
used to write the estimate: virtual (photos/video — the best option when \
the customer wants an out-of-network shop), CCC1 (if that specific shop's \
software supports it — CCC1 being auto-assigned implies the customer \
already chose their own shop), or rarely a physical field adjuster. \
Repair cost itself is paid the same way regardless of network status; \
only the out-of-network TOW cost (not the repair) works via customer \
reimbursement.
- Default inspection routing at FNOL is an in-network shop near the \
participant — UNLESS this is a THIRD-PARTY claim, in which case do NOT \
auto-route to any shop yet: liability must first be established via \
contact and evidence (police report, photos/video) before an inspection \
method is assigned.
- Rental eligibility: on a "liability" claim_type claim, a third party \
only qualifies for a rental ONCE liability has been determined in their \
favor — never proactively before that. If liability is not yet \
determined, nothing (rental, coverage limit, payment timing) should be \
offered to the third party; they should only be told liability is being \
determined and to send supporting evidence. Conversely, a first-party \
insured who is attorney-represented but whose policy has no \
collision/comprehensive coverage most likely will NOT have rental \
available to them.
- coverage_limit is ADJUSTER-FACING CONTEXT ONLY — it informs \
review_priority/reasoning/key_flags, but must NEVER appear in \
customer_notification, for any party, under any circumstance. Only treat \
it as relevant at all if estimated_repair_or_payout is approaching or \
exceeding it; otherwise ignore coverage_limit entirely. When it is \
relevant, note the TOTAL PAYMENTS made on the claim so far (including any \
supplements) in your own reasoning/key_flags — use \
estimated_repair_or_payout as the best available proxy for that — but \
this stays internal, never customer-facing.
- Attorney representation on auto claims is actually uncommon overall, \
and when it happens it's typically tied to an injury component rather \
than pure property damage — keep this in mind when explaining an \
attorney-repped claim's context, though always treat this specific \
claim's stated fields as ground truth.
- A repair shop submitting a supplement means the customer is ALREADY \
using that shop for repairs — it's a confirmation of an existing repair \
relationship, not a new shop being chosen. Never suggest shop-routing \
action for a supplement-related message. A supplement is NEVER valid on a \
total-loss claim — a total-loss vehicle isn't being repaired, so treat any \
"supplement" mention alongside is_total_loss=true as a data inconsistency, \
not a real repair event; flag it in reasoning rather than treating it at \
face value. An IN-NETWORK shop's supplement is routine and essentially \
pre-approved under the network agreement — this is email_sufficient or \
even no_action_needed, not something requiring extra adjuster review. An \
OUT-OF-NETWORK shop's supplement is the one that may actually need \
adjuster attention, since there's no standing network agreement covering it.
- If key information is missing or unclear (e.g. incomplete details at \
intake, conflicting info), don't automatically treat this as high \
priority or requiring a call. A preliminary email trying to retrieve the \
missing information is often the better first move — a call can still be \
offered as an option, but shouldn't be the forced default.
- Questions about whether a TOW is covered (sometimes phrased as a \
"roadside assistance" question) can be answered directly, not referred \
elsewhere: tows are covered as long as the person has 1st-party coverage, \
or they're a claimant — paid directly if an in-network shop is used, \
reimbursed if a non-network shop is used. This is a routine \
email_sufficient answer, not something requiring referral.
- For 1st-party insureds ONLY: proactively disclose the deductible in \
every notification, regardless of whether there's a coverage issue flag. \
Never disclose coverage_limit to the customer — see the adjuster-facing \
rule above.
- For third-party claimants (TPC): NEVER disclose deductible or coverage \
details under any circumstance, even if other rules in this prompt \
mention disclosure. TPC messages should only be told liability needs \
to be determined and to send supporting evidence. This rule overrides the \
disclosure rule above whenever sender_type is TPC.
"""

THREAD_SYSTEM_PROMPT = f"""You are helping an auto insurance adjuster get up to speed on a \
claim they're about to work. You'll be given every inbound message on the \
claim so far, in order, plus the claim's current status fields (including, \
where relevant, contact info for a total-loss adjuster or attorney desk).
{SHARED_GUIDANCE}
Return ONLY a JSON object, no preamble, no markdown fences:

{{
  "summary": "<2-4 sentence plain-language summary of current state>",
  "review_priority": "<low, medium, or high>",
  "contact_method": "<call_needed, email_sufficient, no_action_needed, \
refer_total_loss_adjuster, or refer_attorney_desk>",
  "call_or_referral_focus": "<what to address on a call, OR what the \
next-steps email should cover if this is a referral case. Empty string \
if no_action_needed.>",
  "customer_notification": "<proactive customer-facing draft — for \
  1ST-PARTY INSUREDS ONLY, must always include deductible (never \
  coverage_limit — that is adjuster-only, see rules above), plus \
  rental/UMPD/inspection info if relevant. For THIRD-PARTY CLAIMANTS, \
  NEVER include deductible or coverage limit — only that liability needs \
  to be determined and to send supporting evidence, UNLESS liability has \
  already been determined in their favor per the claim status fields. \
  Empty string only if truly nothing to add.>",
  "key_flags": ["<short list for at-a-glance context>"]
}}
"""

FNOL_SYSTEM_PROMPT = f"""You are helping an auto insurance adjuster prepare for a claim \
that was JUST FILED — no messages or contact has happened yet. Many \
insureds expect the company to reach out first. You'll be given the \
claim's status fields only (including, where relevant, contact info for a \
total-loss adjuster or attorney desk).
{SHARED_GUIDANCE}
Return ONLY a JSON object, no preamble, no markdown fences:

{{
  "summary": "<2-3 sentence plain-language summary based on status fields \
alone>",
  "review_priority": "<low, medium, or high>",
  "contact_method": "<call_needed, email_sufficient, no_action_needed, \
refer_total_loss_adjuster, or refer_attorney_desk — 'email_sufficient' \
should be your default for a routine new claim with no red flags; a fresh \
claim status update is generally enough at this stage.>",
  "call_or_referral_focus": "<what to prepare for on a call, or what the \
next-steps/referral email should cover. Empty string if not applicable.>",
  "customer_notification": "<draft of what to proactively send the \
customer now — for 1ST-PARTY INSUREDS ONLY, ALWAYS state their \
deductible (never coverage_limit — that's adjuster-only info), plus \
rental options/timeline, UMPD status, and which inspection method \
they're routed to and what it means for them. Only mention coverage \
limit at all if estimated_repair_or_payout is approaching or exceeding \
it, and even then, phrase it as something the adjuster is tracking — \
never state the dollar figure to the customer directly. For \
THIRD-PARTY CLAIMANTS, NEVER mention deductible or coverage limits — \
only that liability needs to be determined and to send supporting \
evidence, unless liability has already been determined in their favor \
per the claim status fields. Should read like a reassuring 'here's \
what happens next' message.>",
  "key_flags": ["<short list for at-a-glance context>"]
}}
"""


def build_thread_context(claim_id: str, messages: pd.DataFrame, rules_row: pd.Series) -> str:
    lines = [f"CLAIM {claim_id} — STATUS FIELDS:"]
    lines += _status_field_lines(rules_row)
    lines.append(
        "\n(Note: insured names in this test data are randomly generated "
        "per message and are not reliable identifiers — treat all "
        "'insured' messages below as the same person unless content "
        "clearly indicates otherwise.)"
    )

    # Only "tpc" and "attorney" (when representing a third party) are a
    # genuinely different PARTY from the insured. "agent" and "lienholder"
    # are on the insured's side (working on their behalf or holding a
    # financial interest in their vehicle) — never count them as a
    # separate party for mixed-thread/disclosure purposes.
    OTHER_PARTY_SENDER_TYPES = {"tpc", "attorney"}
    distinct_senders = set(messages["sender_type"].unique())
    other_party_present = bool(distinct_senders & OTHER_PARTY_SENDER_TYPES)
    if other_party_present:
        lines.append(
            f"\nNOTE: this thread includes at least one sender representing "
            f"a DIFFERENT party than the insured "
            f"({', '.join(sorted(distinct_senders & OTHER_PARTY_SENDER_TYPES))}). "
            f"Do NOT write one blended customer_notification covering both "
            f"sides — state that this claim involves multiple parties and "
            f"that separate per-party notifications are needed, with NO "
            f"deductible or coverage_limit mentioned in that case. Senders "
            f"like 'agent' or 'lienholder' are on the INSURED's side, not a "
            f"separate party, and don't trigger this note on their own."
        )

    lines.append("\nMESSAGES (in order received):")
    for _, msg in messages.iterrows():
        lines.append(f"  - [{msg['sender_type']} via {msg['channel']}]: {msg['message_body']}")
    return "\n".join(lines)


def build_fnol_context(claim_id: str, rules_row: pd.Series) -> str:
    lines = [f"CLAIM {claim_id} — STATUS FIELDS (no messages yet):"]
    lines += _status_field_lines(rules_row)
    return "\n".join(lines)


def _status_field_lines(rules_row: pd.Series) -> list:
    status_fields = [
        "claim_type", "deductible", "coverage_limit", "potential_issue_flag",
        "reassignment_type", "reassignment_sla", "is_umpd_claim",
        "insured_has_umpd_coverage", "umpd_review_needed", "vehicle_drivable",
        "inspection_method", "estimate_written_days", "method_assignment_days",
        "has_rental", "rental_coverage_type", "advance_pay_days",
        "is_total_loss", "rental_end_rule",
    ]
    lines = []
    for field in status_fields:
        if field in rules_row and pd.notna(rules_row[field]) and rules_row[field] != "":
            lines.append(f"  - {field}: {rules_row[field]}")

    if rules_row.get("is_total_loss"):
        lines.append(f"  - total_loss_adjuster_contact: {fake_phone()} (separate from claim handler)")
    if rules_row.get("reassignment_type") == "attorney":
        lines.append(f"  - attorney_desk_contact: {fake_phone()}")

    return lines


def call_model(system_prompt: str, user_context: str, _retry: bool = True) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_context}
        ],
        temperature=0,
        max_tokens=600
    )
    raw_text = response.choices[0].message.content.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Likely truncated (hit max_tokens) or an unexpected preamble.
        # Retry once with a stricter reminder before giving up.
        if _retry:
            stricter_context = (
                user_context
                + "\n\nIMPORTANT: your previous response was not valid JSON "
                "(possibly cut off). Keep every field concise — 1-2 "
                "sentences max — and return ONLY the JSON object, nothing "
                "else, no matter how complex the claim is."
            )
            return call_model(system_prompt, stricter_context, _retry=False)
        return {
            "summary": f"PARSE_ERROR after retry: {raw_text[:300]}",
            "review_priority": "unknown", "contact_method": "unknown",
            "call_or_referral_focus": "", "customer_notification": "", "key_flags": [],
        }


def main():
    comms = pd.read_csv(COMMS_PATH, encoding="utf-8", encoding_errors="replace")
    rules = pd.read_csv(RULES_PATH, encoding="utf-8", encoding_errors="replace")
    rules_lookup = rules.set_index("claim_id")

    msg_counts = comms.groupby("claim_id").size()
    claims_with_messages = msg_counts[msg_counts >= 3].index.tolist()
    all_claims = rules["claim_id"].tolist()

    sample_with_msgs = pd.Series(claims_with_messages).sample(
        min(N_CLAIMS_WITH_MESSAGES, len(claims_with_messages)), random_state=7
    ).tolist()
    sample_fnol_only = pd.Series(all_claims).sample(
        min(N_CLAIMS_FNOL_ONLY, len(all_claims)), random_state=13
    ).tolist()

    print(f"Claims with 3+ messages available: {len(claims_with_messages)}\n")

    results = []

    for i, claim_id in enumerate(sample_with_msgs):
        print(f"[Thread path] Summarizing {i + 1}/{len(sample_with_msgs)}: {claim_id}...")
        claim_messages = comms[comms["claim_id"] == claim_id]
        rules_row = rules_lookup.loc[claim_id]
        context = build_thread_context(claim_id, claim_messages, rules_row)
        summary = call_model(THREAD_SYSTEM_PROMPT, context)
        summary.update({"claim_id": claim_id, "path": "message_thread",
                        "num_messages": len(claim_messages)})
        results.append(summary)

    for i, claim_id in enumerate(sample_fnol_only):
        print(f"[FNOL-only path] Summarizing {i + 1}/{len(sample_fnol_only)}: {claim_id}...")
        rules_row = rules_lookup.loc[claim_id]
        context = build_fnol_context(claim_id, rules_row)
        summary = call_model(FNOL_SYSTEM_PROMPT, context)
        summary.update({"claim_id": claim_id, "path": "fnol_only", "num_messages": 0})
        results.append(summary)

    print("\n" + "=" * 80)
    for r in results:
        print(f"\nCLAIM {r['claim_id']}  [{r['path']}, {r['num_messages']} messages]")
        print(f"Review priority: {r.get('review_priority', 'unknown').upper()}"
              f"   |   Contact method: {r.get('contact_method', 'unknown').upper()}")
        print(f"Summary: {r.get('summary', '')}")
        if r.get("call_or_referral_focus"):
            print(f"Call/referral focus: {r['call_or_referral_focus']}")
        if r.get("customer_notification"):
            print(f"Customer notification draft: {r['customer_notification']}")
        if r.get("key_flags"):
            print(f"Flags: {', '.join(r['key_flags'])}")
        print("-" * 80)

    out_df = pd.DataFrame(results)
    out_df.to_csv("step3_claim_briefings.csv", index=False)
    print("\nSaved to step3_claim_briefings.csv")

    return out_df


if __name__ == "__main__":
    briefings = main()
