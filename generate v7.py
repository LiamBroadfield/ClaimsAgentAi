"""
v4 synthetic dataset generator. Adds on top of v3:

- realistic 800-XXX-XXXX placeholder phone numbers wherever a number is needed
- corrected UMPD filing logic:
    * has UMPD coverage + at-fault party unknown/uninsured -> review reply, 1-3 biz days
    * has UMPD coverage + at-fault party known/insured -> no special reply
    * no UMPD coverage + claimant did NOT cause property damage -> claim not filed, notify
    * no UMPD coverage + claimant DID cause property damage -> claim is filed anyway,
      but claimant is told they don't have that coverage
    * UMPD confusion most commonly surfaces as a rental-related question, not a filing one
- five inspection/repair methods with realistic weighting and rules:
    1. network shop (most common, "sell" it — company tows there on its own dime,
       estimate written within 1 business day, repairs start immediately, least
       customer effort)
    2. virtual/photo estimate (2nd most common, fastest, lowest estimate since
       visual-only, customers often dislike it; if a non-network shop is named at
       any point, supplement instructions auto-send as a placeholder in the
       post-estimate email)
    3. field inspection (circumstantial/rare, up to 7 days to assign, only when
       other options don't work and vehicle is at a non-network shop)
    4. CCC1 (circumstantial, ~7 days, easiest when customer already has a shop
       of choice)
    5. bill review (extremely rare — customer already paid out of pocket)

v5 adds on top of v4:
- ~20-30% of claimant messages are urgent/disgruntled callback requests
- a "call" channel (customers who called in), which carries priority
- fallback-to-task logic for odd one-off requests auto-response can't handle
- an AI-use disclosure appended to every auto-generated/AI-drafted reply
- referral-needed messages that just need routing to another number/department
- a total-loss automated note logging that the customer got the adjuster's direct number

All data fabricated for prototyping only.
"""

import csv
import random

# Dedicated RNGs so cosmetic/numeric fields (deductible, repair cost,
# follow-up scheduling) can be added to or tweaked freely without ever
# perturbing the shared `random` stream that drives
# claim_type/messages/urgency generation elsewhere in this file. This is
# the permanent fix for urgency-match drifting every time an unrelated
# field gets added — those fields no longer share a stream with anything
# that affects classification-relevant output.
followup_rng = random.Random(999)
aux_rng = random.Random(777)
from collections import Counter

random.seed(31)

N_CLAIMS = 900
N_COMMUNICATIONS = 8000

AI_DISCLOSURE = (
    " This message was generated with the help of AI and reviewed for "
    "accuracy before sending."
)


def fake_phone():
    return f"800-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


FIRST_NAMES = ["Jordan", "Maria", "David", "Priya", "Sam", "Alex", "Taylor",
               "Chen", "Fatima", "Marcus", "Emily", "Noah", "Grace", "Owen",
               "Layla", "Diego", "Hana", "Victor"]
LAST_NAMES = ["Reyes", "Kim", "Johnson", "Patel", "Nguyen", "Brooks", "Silva",
              "Carter", "Okafor", "Lindgren", "Ramirez", "Bennett", "Wallace",
              "Ferreira", "Yamamoto", "Cohen"]
SHOPS = ["Riverside Auto Body", "Precision Collision Center", "Maple Street Auto",
         "Downtown Body Works", "Evergreen Auto Repair", "Summit Collision",
         "Coastal Auto Care", "Ironline Body Shop"]
NON_NETWORK_SHOPS = ["Joe's Discount Auto", "Family Owned Collision",
                      "QuickFix Body Shop", "Neighborhood Auto Works"]
LAW_FIRMS = ["Whitfield & Cross", "Marsh Injury Law", "Delgado Legal Group",
             "Harmon & Reyes LLP", "The Kessler Firm", "Boone Law Partners"]
LIENHOLDERS = ["First National Auto Finance", "Summit Credit Union",
               "Pinnacle Bank Auto Loans", "Coastal Federal Credit Union"]
AGENT_AGENCIES = ["Northshore Insurance Agency", "Baywater Agents Group",
                   "Cascade Independent Agents", "Union Square Insurance"]

CLAIM_TYPES = ["collision", "comprehensive", "liability", "uninsured_motorist"]

CHANNEL_WEIGHTS_DEFAULT = {"text": 0.35, "email": 0.35, "app": 0.10, "call": 0.20}
CHANNEL_WEIGHTS_AGENT = {"text": 0.08, "email": 0.24, "app": 0.55, "call": 0.13}

SENDER_TYPE_WEIGHTS = {
    "insured": 0.62,
    "shop": 0.18,
    "agent": 0.09,
    "tpc": 0.07,
    "attorney": 0.03,
    "lienholder": 0.01,
}

INSPECTION_METHOD_WEIGHTS = {
    "network_shop": 0.55,
    "virtual_photo": 0.30,
    "ccc1": 0.08,
    "field": 0.05,
    "bill_review": 0.02,
}

# template, urgency, category, auto_respond_eligible, attorney_flag, rental_related, weight
CLAIMANT_TEMPLATES = [
    ("Hi, this is {name}, claim #{claim_id}. My repair shop said the estimate "
     "will take about {days} days. Still haven't heard back about my rental "
     "car coverage — can someone confirm that's approved?",
     "medium", "rental_coverage_confirmation", False, False, True, 4),
    ("Following up on #{claim_id}. Attached photos of the damage you "
     "requested. Let me know if you need anything else.",
     "low", "document_dropoff", False, False, False, 4),
    ("Hi, {name} here, #{claim_id}. Letter says you need a police report but "
     "this was a parking lot fender bender, no police involved. What do I "
     "send instead?",
     "medium", "documentation_clarification", False, False, False, 4),
    ("Hi there, #{claim_id}. Settlement check arrived, everything looks "
     "correct, no further action needed.",
     "low", "closed_loop_confirmation", False, False, False, 4),
    ("What's my deductible on this claim (#{claim_id})? Want to know what "
     "I'm on the hook for before I approve the repair.",
     "medium", "deductible_inquiry", False, False, False, 4),
    ("Hi, this is {name} on #{claim_id} — I retained {firm} for this claim, "
     "they'll be in touch going forward.",
     "high", "claimant_announces_representation", False, True, False, 3),
    ("Hi, #{claim_id} — how do I actually get a rental car? Do I book it "
     "myself or does someone set that up?",
     "medium", "rental_how_to_reserve", True, False, True, 4),
    ("This is {name}, #{claim_id}. My rental keeps getting extended, how "
     "much longer will it actually be covered? Don't want a surprise bill.",
     "medium", "rental_duration_inquiry", False, False, True, 4),
    ("Hi, #{claim_id}, quick question — is my rental direct billed or do I "
     "pay and get reimbursed?",
     "medium", "rental_payment_method_inquiry", False, False, True, 4),
    ("Hi, this is {name}, #{claim_id}. I already took my car to {nonnet_shop}, "
     "is that a problem? Are they in your network?",
     "medium", "non_network_shop_notice", False, False, False, 4),
    ("This is {name}, claim #{claim_id}. What shops are in your network near "
     "me? Want to pick one that won't cause delays.",
     "low", "in_network_shop_inquiry", False, False, False, 4),
    ("Hi, #{claim_id}, my rental had a deposit charge on my card, I thought "
     "that was covered? Also got charged for the extra insurance I added at "
     "the counter.",
     "medium", "rental_deposit_addon_dispute", False, False, True, 4),
    ("This is {name}, #{claim_id}. My car won't drive at all after the "
     "accident, what happens now?",
     "high", "vehicle_non_drivable_notice", False, False, False, 4),
    ("Hi, #{claim_id} — someone hit me and I don't think they had insurance. "
     "If I need a rental, does my policy actually cover that?",
     "medium", "umpd_coverage_question", False, False, True, 4),
    ("This is {name}, #{claim_id}. Can I just send photos instead of taking "
     "it somewhere? Trying to avoid the hassle.",
     "low", "virtual_estimate_request", False, False, False, 4),
    ("Hi, #{claim_id}, can I use my own shop, {nonnet_shop}, instead of one "
     "of yours? They quoted me already.",
     "medium", "own_shop_preference", False, False, False, 4),
    ("This is {name} on #{claim_id} — it's been almost a week, when will "
     "the field inspector actually come out?",
     "medium", "field_inspection_status", False, False, False, 3),
    ("Hi, #{claim_id}, I already paid to fix it myself before filing, can "
     "you just review the bill and reimburse me?",
     "low", "bill_review_request", False, False, False, 2),
    # --- fallback / task-generating: odd one-off requests auto-response can't handle ---
    ("Hi, #{claim_id}, this is {name}. Random question — I'm moving next month, "
     "does that change anything about my claim or do I need to update something?",
     "low", "fallback_odd_request_address_change", False, False, False, 2),
    ("This is {name} on #{claim_id}. My ex is also on this policy and I don't "
     "want them contacted about this claim at all, is that something you can "
     "note somewhere?",
     "medium", "fallback_odd_request_contact_restriction", False, False, False, 2),
    ("Hi, #{claim_id}, weird situation — the other driver's info I gave you "
     "might be wrong, I think I mixed up two people at the scene. Not sure how "
     "to fix that.",
     "medium", "fallback_odd_request_incorrect_info", False, False, False, 2),
    # --- referral-needed: just needs routing to another number/department ---
    ("Hi, #{claim_id}, I need to submit a medical bill for the injury from this "
     "accident, who do I send that to?",
     "medium", "referral_medical_bills", False, False, False, 3),
    ("This is {name}, #{claim_id}. My windshield cracked, is that handled "
     "under this claim or somewhere else?",
     "low", "referral_glass_claims", False, False, False, 3),
    ("Hi, #{claim_id}, I'm stuck on the side of the road right now, is this "
     "the right number for roadside assistance?",
     "high", "referral_roadside_assistance", False, False, False, 3),
    ("This is {name} — I have a question about my monthly premium, not sure "
     "if this is the right place to ask about claim #{claim_id}.",
     "low", "referral_billing_premium", False, False, False, 3),
    # --- urgent/disgruntled callback requests (weighted heavily — target ~25% of claimant volume) ---
    ("This is {name} again on #{claim_id}. It's been almost two weeks, no "
     "update, medical bills piling up, no callback. I need my adjuster to call "
     "me today, this is urgent.",
     "high", "urgent_callback_request", False, False, False, 9),
    ("I am extremely frustrated with claim #{claim_id}. I have called and "
     "emailed multiple times with no response. Please have the adjuster who "
     "owns this claim call me back immediately.",
     "high", "urgent_callback_request", False, False, False, 9),
    ("This is {name} on #{claim_id} — I need a call back TODAY. This has "
     "dragged on too long and I'm losing patience with this whole process.",
     "high", "urgent_callback_request", False, False, False, 9),
    ("Hi, #{claim_id}, this is {name}, I've left messages with no response. "
     "Can you please have my claim owner personally call me? I don't want "
     "another generic email.",
     "high", "urgent_callback_request", False, False, False, 8),
]

# Referral numbers by category, used for referral-needed messages
REFERRAL_NUMBERS = {
    "referral_medical_bills": fake_phone,
    "referral_glass_claims": fake_phone,
    "referral_roadside_assistance": fake_phone,
    "referral_billing_premium": fake_phone,
}

# non-claimant senders keep a simple uniform-random 6-tuple format
TEMPLATES = {
    "shop": [
        ("Hi, this is {shop} on claim #{claim_id}. Submitting a supplement for "
         "additional parts found during teardown, total add is ${amt}. How long "
         "is supplement turnaround right now?",
         "low", "supplement_turnaround_question", True, False, False),
        ("{shop} here, #{claim_id}. Vehicle is ready for pickup, just need final "
         "sign-off on the supplement so we can close out billing.",
         "medium", "supplement_signoff_needed", False, False, False),
        ("This is {shop}, claim #{claim_id}. Parts are backordered, repair will "
         "run {days} extra days. Wanted to flag in case rental coverage needs "
         "extending.",
         "medium", "repair_delay_rental_impact", False, False, True),
        ("{shop} — supplement #2 submitted on #{claim_id}, ${amt} for hidden "
         "frame damage. Standard turnaround question, no rush.",
         "low", "supplement_turnaround_question", True, False, False),
    ],
    "attorney": [
        ("This letter is to notify you that {firm} now represents the claimant "
         "on claim #{claim_id} for all matters related to this loss. Direct all "
         "future correspondence to our office.",
         "high", "attorney_representation_notice", False, True, False),
        ("Following up on #{claim_id}, our client's medical treatment is "
         "ongoing. Requesting a status update on liability determination and "
         "reserve information.",
         "high", "attorney_status_liability_request", False, True, False),
    ],
    "agent": [
        ("Hi, this is the agent of record for #{claim_id}, {agency}. Client "
         "called asking if their comp/collision coverage applies here — can you "
         "confirm coverage type applied?",
         "medium", "agent_coverage_confirmation", False, False, False),
        ("{agency} — checking status on #{claim_id} for my client, they asked "
         "me directly since they haven't heard from the adjuster.",
         "medium", "agent_status_check", False, False, False),
        ("Quick one from {agency}: does policy on #{claim_id} include rental "
         "reimbursement? Client is asking before they commit to a rental.",
         "low", "agent_rental_coverage_question", False, False, True),
    ],
    "lienholder": [
        ("This is {lienholder} regarding claim #{claim_id}. Please confirm "
         "our loss payee interest is reflected on any issued settlement check.",
         "medium", "lienholder_loss_payee_confirmation", False, False, False),
        ("{lienholder} — requesting payoff verification status for total loss "
         "claim #{claim_id} before we can release title.",
         "high", "lienholder_payoff_verification", False, False, False),
    ],
    "tpc": [
        ("I was the other driver in the accident with your insured on "
         "#{claim_id}. Wanted to ask about liability determination and how to "
         "get my vehicle damage covered.",
         "high", "tpc_liability_inquiry", False, False, False),
        ("This is regarding #{claim_id}, I'm the third party. My mechanic gave "
         "an estimate, how do I submit it?",
         "medium", "tpc_property_damage_submission", False, False, False),
        ("Follow-up on #{claim_id} — still waiting to hear if your insured was "
         "found at fault so I can move forward with my repair.",
         "medium", "tpc_liability_status_check", False, False, False),
    ],
}


def tpc_response():
    return (
        "Thank you for reaching out. Liability for this loss has not yet been "
        "determined and will require contact with our insured before we can "
        "proceed. In the meantime, please send any supporting evidence you have "
        "— such as a police report, photos, or video of the incident/damage — "
        "so it can be added to the file."
    )


def umpd_review_reply():
    return (
        "Your claim involves uninsured/underinsured motorist property damage "
        "coverage, which requires a coverage review before it can proceed. "
        "This review typically takes 1-3 business days; you'll be contacted "
        "once it's complete." + AI_DISCLOSURE
    )


def umpd_no_coverage_not_filed_reply():
    return (
        f"Based on your policy, you do not currently carry uninsured/"
        f"underinsured motorist property damage coverage, so this loss "
        f"would not be eligible under that coverage and a claim will not be "
        f"filed. If you have questions about your coverage options, please "
        f"contact your agency at {fake_phone()}." + AI_DISCLOSURE
    )


def umpd_no_coverage_but_filed_reply():
    return (
        "This claim is being filed because it involves property damage you "
        "caused. For your awareness, your policy does not include "
        "uninsured/underinsured motorist property damage coverage, so that "
        "specific coverage does not apply here." + AI_DISCLOSURE
    )


def injury_reassignment_reply():
    return (
        "Because this claim involves a third-party injury, it is being "
        "immediately reassigned to an adjuster with specialized injury-claim "
        "experience. You will be contacted shortly." + AI_DISCLOSURE
    )


def attorney_reassignment_reply():
    return (
        "Because this claim is now attorney-represented, it is being "
        "reassigned to an adjuster who handles represented claims. "
        "Reassignment typically completes within 5-7 business days." + AI_DISCLOSURE
    )


def weighted_choice(weights: dict):
    items, probs = zip(*weights.items())
    return random.choices(items, weights=probs, k=1)[0]


def weighted_choice_rng(rng: random.Random, weights: dict):
    """Same as weighted_choice, but draws from a specific Random instance
    instead of the shared module-level stream."""
    items, probs = zip(*weights.items())
    return rng.choices(items, weights=probs, k=1)[0]


def make_claim_ids(n):
    return [f"CLM{100000 + i}" for i in range(1, n + 1)]


def pick_claimant_template():
    weights = [t[6] for t in CLAIMANT_TEMPLATES]
    return random.choices(CLAIMANT_TEMPLATES, weights=weights, k=1)[0][:6]


def generate_communications(claim_ids, attorney_claim_ids, claim_type_lookup, total_loss_claim_ids, umpd_claim_ids):
    attorney_claim_ids = list(attorney_claim_ids)
    total_loss_claim_ids = set(total_loss_claim_ids)
    non_tl_claim_ids = [c for c in claim_ids if c not in total_loss_claim_ids]
    umpd_claim_ids = set(umpd_claim_ids)
    non_umpd_claim_ids = [c for c in claim_ids if c not in umpd_claim_ids]
    rows = []
    for i in range(1, N_COMMUNICATIONS + 1):
        claim_id = random.choice(claim_ids)
        claim_type = claim_type_lookup[claim_id]
        sender_type = weighted_choice(SENDER_TYPE_WEIGHTS)

        # A repair shop has no reason to be messaging about a total-loss
        # claim — nothing is being repaired there. Redraw the claim_id
        # (from the non-total-loss pool) rather than let a shop message
        # (including a supplement) land on one.
        if sender_type == "shop" and claim_id in total_loss_claim_ids and non_tl_claim_ids:
            claim_id = random.choice(non_tl_claim_ids)

        # A genuine third-party claimant message doesn't fit a UMPD claim
        # yet — UMPD means the at-fault party is uninsured/unknown, so
        # there's no legitimate "third party awaiting liability
        # determination" correspondence to generate here. Redraw rather
        # than produce that contradiction. (Detailed UMPD/TPC interaction
        # logic is planned as part of the broader loss-type work later.)
        if sender_type == "tpc" and claim_id in umpd_claim_ids and non_umpd_claim_ids:
            claim_id = random.choice(non_umpd_claim_ids)
        channel_weights = CHANNEL_WEIGHTS_AGENT if sender_type == "agent" else CHANNEL_WEIGHTS_DEFAULT
        channel = weighted_choice(channel_weights)

        # An actual attorney message should only ever land on a claim the
        # rules table agrees is attorney-repped — otherwise the message
        # data and the "official" claim status disagree, which is exactly
        # the inconsistency being fixed here.
        if sender_type == "attorney":
            if attorney_claim_ids:
                claim_id = random.choice(attorney_claim_ids)
            else:
                sender_type = "insured"  # no attorney claims exist in this run — fall back

        # claim_id may have been redrawn by any of the guards above —
        # refresh claim_type here, once, right before it's used, so it's
        # guaranteed to match whichever claim_id actually ends up on this
        # row no matter which (if any) redraw path ran.
        claim_type = claim_type_lookup[claim_id]

        if sender_type == "insured":
            template, urgency, category, auto_respond, attorney_flag, rental_related = pick_claimant_template()
            # Same consistency rule for a claimant announcing representation —
            # only allow it on a claim actually flagged attorney-repped.
            tries = 0
            while category == "claimant_announces_representation" and claim_id not in attorney_claim_ids and tries < 10:
                template, urgency, category, auto_respond, attorney_flag, rental_related = pick_claimant_template()
                tries += 1
            if category == "claimant_announces_representation" and claim_id not in attorney_claim_ids:
                attorney_flag = False  # safety net if the loop somehow didn't resolve
        else:
            template, urgency, category, auto_respond, attorney_flag, rental_related = random.choice(TEMPLATES[sender_type])

        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        body = template.format(
            name=name, claim_id=claim_id, shop=random.choice(SHOPS),
            nonnet_shop=random.choice(NON_NETWORK_SHOPS),
            firm=random.choice(LAW_FIRMS), lienholder=random.choice(LIENHOLDERS),
            agency=random.choice(AGENT_AGENCIES),
            days=random.choice([3, 5, 7, 10, 12, 14, 18, 21]),
            amt=random.choice([420, 615, 980, 1250, 1875, 2300]),
        )

        # calls are transcribed as notes rather than raw text, so keep full body;
        # texts stay clipped like earlier versions
        if channel == "text" and len(body) > 140:
            body = body.split(". ")[0] + ("." if not body.endswith(".") else "")

        priority_boost = channel == "call"  # customers who called in get priority

        # fallback-to-task: odd one-off requests the auto-responder can't handle
        requires_fallback_task = category.startswith("fallback_odd_request")
        fallback_task_summary = (
            f"Review needed — claimant raised a non-standard request on "
            f"{claim_id} ({category.replace('fallback_odd_request_', '').replace('_', ' ')}); "
            f"not eligible for auto-response, needs adjuster review."
            if requires_fallback_task else ""
        )

        # referral-needed: route to another number/department, with an AI disclosure
        is_referral = category.startswith("referral_")
        if is_referral:
            referral_number = REFERRAL_NUMBERS[category]()
            referral_reply = (
                f"Thanks for reaching out. This is handled by a different team — "
                f"please contact {referral_number} and they can assist you "
                f"directly." + AI_DISCLOSURE
            )
        else:
            referral_number = ""
            referral_reply = ""

        approved_response = (tpc_response() + AI_DISCLOSURE) if sender_type == "tpc" else ""

        rows.append({
            "message_id": f"MSG{i:05d}",
            "claim_id": claim_id,
            "claim_type": claim_type,
            "sender_type": sender_type,
            "channel": channel,
            "priority_boost_called_in": priority_boost,
            "sender_name": name if sender_type == "insured" else "",
            "message_body": body,
            "true_urgency": urgency,
            "true_category": category,
            "auto_respond_eligible": auto_respond,
            "attorney_flag": attorney_flag,
            "rental_related": rental_related,
            "requires_fallback_task": requires_fallback_task,
            "fallback_task_summary": fallback_task_summary,
            "is_referral": is_referral,
            "referral_number": referral_number,
            "referral_auto_reply": referral_reply,
            "tpc_restricted_response": approved_response,
        })
    return rows


def generate_claim_business_rules(claim_ids):
    rows = []
    for claim_id in claim_ids:
        claim_type = random.choice(CLAIM_TYPES)
        deductible = weighted_choice_rng(aux_rng, {
            250: 0.15,
            500: 0.45,
            1000: 0.20,
            1500: 0.12,
            2500: 0.08,
        })
        coverage_limit = random.choice([25000, 50000, 100000, 250000])

        # moved up from later in the loop — repair cost realism below
        # needs to know drivability before it can be computed
        vehicle_drivable = random.random() < 0.75
        tow_required = not vehicle_drivable

        # Coverage-limit issues (approaching/exceeding) should be rare in
        # practice — most claims settle comfortably under the limit. Only
        # ~7% of claims deliberately land in that danger zone; the rest are
        # drawn from a lower, more realistic range.
        if claim_type == "liability" and random.random() < 0.07:
            est_repair_or_payout = round(random.uniform(coverage_limit * 0.85, coverage_limit * 1.05), 2)
        elif vehicle_drivable:
            # A driveable vehicle implies cosmetic/moderate damage only —
            # it doesn't make sense for a driveable car to carry a $10k+
            # repair estimate. Keep this tier tightly capped.
            est_repair_or_payout = round(aux_rng.uniform(800, 6000), 2)
        else:
            # Non-drivable implies real structural/mechanical damage, so
            # the wider realistic range applies: most still land $1k-8k,
            # a smaller share run $8k-15k, and only a rare tail (~5%)
            # reaches $15k-30k — genuinely expensive vehicles, not the norm.
            _r = aux_rng.random()
            if _r < 0.65:
                est_repair_or_payout = round(aux_rng.uniform(1000, 8000), 2)
            elif _r < 0.90:
                est_repair_or_payout = round(aux_rng.uniform(8000, 15000), 2)
            else:
                est_repair_or_payout = round(aux_rng.uniform(15000, 30000), 2)
        deductible_met = est_repair_or_payout >= deductible
        approaching_limit = est_repair_or_payout >= coverage_limit * 0.85
        exceeds_limit = est_repair_or_payout > coverage_limit
        if exceeds_limit:
            issue_flag = "exceeds_coverage_limit"
        elif approaching_limit:
            issue_flag = "approaching_coverage_limit"
        elif not deductible_met:
            issue_flag = "below_deductible"
        else:
            issue_flag = "none"

        has_3rd_party_injury = random.random() < 0.06
        # attorneys primarily represent for injuries in reality — so attorney
        # representation is much more likely when there's an injury involved,
        # and rare otherwise
        is_attorney_repped = (
            random.random() < 0.20 if has_3rd_party_injury else random.random() < 0.02
        )
        if has_3rd_party_injury:
            reassignment_type = "injury"
            reassignment_reply = injury_reassignment_reply()
            reassignment_sla = "immediate"
        elif is_attorney_repped:
            reassignment_type = "attorney"
            reassignment_reply = attorney_reassignment_reply()
            reassignment_sla = "5-7 business days"
        else:
            reassignment_type = "none"
            reassignment_reply = ""
            reassignment_sla = ""

        is_umpd_claim = claim_type == "uninsured_motorist" and random.random() < 0.5
        insured_has_umpd_coverage = ""
        at_fault_party_status = ""
        claimant_caused_property_damage = ""
        umpd_claim_filed = ""
        umpd_auto_reply = ""
        umpd_review_needed = False

        if is_umpd_claim:
            insured_has_umpd_coverage = random.random() < 0.75
            if insured_has_umpd_coverage:
                at_fault_party_status = random.choice(["unknown", "uninsured", "known_insured"])
                if at_fault_party_status in ("unknown", "uninsured"):
                    umpd_review_needed = True
                    umpd_auto_reply = umpd_review_reply()
                umpd_claim_filed = True
            else:
                claimant_caused_property_damage = random.random() < 0.2
                if claimant_caused_property_damage:
                    umpd_claim_filed = True
                    umpd_auto_reply = umpd_no_coverage_but_filed_reply()
                else:
                    umpd_claim_filed = False
                    umpd_auto_reply = umpd_no_coverage_not_filed_reply()

        # (vehicle_drivable / tow_required now set earlier in this loop,
        # before the repair-cost block that depends on it)

        # Subrogation potential: only meaningful on collision claims where
        # the insured used their own coverage after being hit by an
        # identifiable at-fault third party. Total-loss/liability/UM claims
        # follow different tracks and aren't scored here.
        subrogation_potential = (
            claim_type == "collision" and followup_rng.random() < 0.25
        )

        # Police report: more likely on liability/UM claims (multi-party,
        # fault disputes) than routine single-vehicle collision/comprehensive
        police_report_filed = followup_rng.random() < (
            0.65 if claim_type in ("liability", "uninsured_motorist") else 0.45
        )

        inspection_method = weighted_choice(INSPECTION_METHOD_WEIGHTS)
        non_network_shop_indicated = random.random() < 0.25

        if inspection_method == "network_shop":
            estimate_written_days = 1
            repairs_start = "immediately after estimate"
            tow_included_and_paid = tow_required
            customer_effort_level = "low"
            lowest_estimate = False
            supplement_auto_send = False
            method_assignment_days = ""
        elif inspection_method == "virtual_photo":
            estimate_written_days = 0
            repairs_start = "after shop confirms supplement, if any"
            tow_included_and_paid = False
            customer_effort_level = "medium"
            lowest_estimate = True
            supplement_auto_send = non_network_shop_indicated
            method_assignment_days = ""
        elif inspection_method == "field":
            estimate_written_days = ""
            repairs_start = "after field inspection completes"
            tow_included_and_paid = False
            customer_effort_level = "low"
            lowest_estimate = False
            supplement_auto_send = False
            method_assignment_days = 7
        elif inspection_method == "ccc1":
            estimate_written_days = ""
            repairs_start = "after CCC1 estimate finalized"
            tow_included_and_paid = False
            customer_effort_level = "low"
            lowest_estimate = False
            supplement_auto_send = False
            method_assignment_days = 7
        else:
            estimate_written_days = ""
            repairs_start = "already completed by customer"
            tow_included_and_paid = False
            customer_effort_level = "high (already paid out of pocket)"
            lowest_estimate = False
            supplement_auto_send = False
            method_assignment_days = ""

        has_rental = random.random() < 0.55

        # Total loss only ever applies to non-drivable claims with a
        # repair estimate high enough to plausibly exceed vehicle value —
        # never to a driveable vehicle, and never below the deductible.
        # NOTE: liability-claim total loss (a third party's vehicle being
        # totaled) isn't modeled yet — planned as part of the broader
        # loss-type work later.
        if not vehicle_drivable and est_repair_or_payout >= 15000:
            is_total_loss = random.random() < 0.55
        elif not vehicle_drivable and est_repair_or_payout >= 8000:
            is_total_loss = random.random() < 0.20
        else:
            is_total_loss = False

        if is_total_loss:
            adjuster_direct_number = fake_phone()
            total_loss_adjuster_number_note = (
                f"Automated note: customer was provided the assigned "
                f"adjuster's direct number ({adjuster_direct_number}) for "
                f"this total loss claim." + AI_DISCLOSURE
            )
        else:
            adjuster_direct_number = ""
            total_loss_adjuster_number_note = ""

        if has_rental:
            party_type = "1st_party" if random.random() < 0.7 else "3rd_party"
            # Reimbursement is the atypical alternative for a 1st-party
            # insured, not a coin flip — direct bill/flat payout is the
            # norm, reimbursement only comes up occasionally when it's
            # offered as an option (and only if they carry the coverage).
            direct_bill_per_day = party_type == "1st_party" and random.random() < 0.85

            if direct_bill_per_day:
                reservation_flow = "direct_bill_proceeds"
                advance_pay_days = ""
                advance_pay_amount = ""
            else:
                reservation_flow = "reimbursement_then_advance_pay_option"
                advance_pay_days = 15 if not vehicle_drivable else 8
                advance_pay_amount = round(advance_pay_days * 30, 2)

            rental_end_rule = (
                "ends 2 days after ACV presentation" if is_total_loss
                else "ends at repair completion per standard terms"
            )
            deposit_excluded = True
            addon_fees_excluded = True
        else:
            party_type = ""
            direct_bill_per_day = ""
            reservation_flow = ""
            advance_pay_days = ""
            advance_pay_amount = ""
            rental_end_rule = ""
            deposit_excluded = ""
            addon_fees_excluded = ""

        # --- follow-up scheduling fields ---
        # fraud logic itself is deferred/built later — this is just a bare
        # flag so the follow-up engine has something to key off of now.
        escalation_needed = followup_rng.random() < 0.03
        fraud_pending_assignment = followup_rng.random() < 0.02

        repair_status = weighted_choice_rng(followup_rng, {
            "not_yet_scheduled": 0.30,
            "scheduled_future": 0.25,
            "in_progress": 0.25,
            "done_awaiting_shop_payment": 0.10,
            "shop_paid": 0.10,
        })

        # vehicle_drivable was only ever set once, at loss time — it never
        # reflected repair progress, so a claim could describe a vehicle
        # as "not driveable" while also "ready for pickup" with repairs
        # done. Once repairs are complete, the vehicle is driveable again
        # (this never applies to total-loss claims, which aren't repaired
        # at all).
        if repair_status in ("done_awaiting_shop_payment", "shop_paid") and not is_total_loss:
            vehicle_drivable = True

        if repair_status == "scheduled_future":
            # weighted toward the 1-2 week range (parts backorder is common)
            repair_appointment_business_days_out = followup_rng.choices(
                [3, 5, 7, 8, 10, 12, 15],
                weights=[10, 20, 20, 20, 15, 10, 5],
                k=1,
            )[0]
        else:
            repair_appointment_business_days_out = ""

        # more likely once repairs are underway or further along
        if repair_status in ("in_progress", "done_awaiting_shop_payment", "shop_paid"):
            initial_estimate_payment_issued = followup_rng.random() < 0.40
        else:
            initial_estimate_payment_issued = followup_rng.random() < 0.10

        # weighted toward recently touched; small tail for stale claims so
        # the no-response/closure rule has something real to catch
        business_days_since_last_contact = followup_rng.choices(
            list(range(0, 31)),
            weights=[10] * 10 + [4] * 10 + [1] * 10 + [2],
            k=1,
        )[0]

        rows.append({
            "claim_id": claim_id,
            "claim_type": claim_type,
            "deductible": deductible,
            "coverage_limit": coverage_limit,
            "estimated_repair_or_payout": est_repair_or_payout,
            "deductible_met": deductible_met,
            "potential_issue_flag": issue_flag,

            "is_attorney_repped": is_attorney_repped,
            "has_3rd_party_injury": has_3rd_party_injury,
            "reassignment_type": reassignment_type,
            "reassignment_sla": reassignment_sla,
            "reassignment_auto_reply": reassignment_reply,

            "is_umpd_claim": is_umpd_claim,
            "insured_has_umpd_coverage": insured_has_umpd_coverage,
            "at_fault_party_status": at_fault_party_status,
            "claimant_caused_property_damage": claimant_caused_property_damage,
            "umpd_claim_filed": umpd_claim_filed,
            "umpd_review_needed": umpd_review_needed,
            "umpd_auto_reply": umpd_auto_reply,

            "vehicle_drivable": vehicle_drivable,
            "tow_required": tow_required,
            "police_report_filed": police_report_filed,
            "subrogation_potential": subrogation_potential,
            "inspection_method": inspection_method,
            "non_network_shop_indicated": non_network_shop_indicated,
            "estimate_written_days": estimate_written_days,
            "repairs_start": repairs_start,
            "tow_included_and_paid_by_carrier": tow_included_and_paid,
            "customer_effort_level": customer_effort_level,
            "lowest_estimate_method": lowest_estimate,
            "supplement_instructions_auto_send": supplement_auto_send,
            "method_assignment_days": method_assignment_days,

            "has_rental": has_rental,
            "is_total_loss": is_total_loss,
            "party_type": party_type,
            "direct_bill_per_day": direct_bill_per_day,
            "reservation_flow": reservation_flow,
            "advance_pay_days": advance_pay_days,
            "advance_pay_daily_rate": 30 if has_rental else "",
            "advance_pay_amount": advance_pay_amount,
            "rental_deposit_excluded": deposit_excluded,
            "rental_addon_fees_excluded": addon_fees_excluded,
            "rental_end_rule": rental_end_rule,
            "adjuster_direct_number": adjuster_direct_number,
            "total_loss_adjuster_number_note": total_loss_adjuster_number_note,

            "escalation_needed": escalation_needed,
            "fraud_pending_assignment": fraud_pending_assignment,
            "repair_status": repair_status,
            "repair_appointment_business_days_out": repair_appointment_business_days_out,
            "initial_estimate_payment_issued": initial_estimate_payment_issued,
            "business_days_since_last_contact": business_days_since_last_contact,
        })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    claim_ids = make_claim_ids(N_CLAIMS)

    # Rules must be generated FIRST now — comms generation needs to know
    # which claims are actually attorney-repped so messages stay consistent
    # with the claim's official status.
    rules = generate_claim_business_rules(claim_ids)
    attorney_claim_ids = [r["claim_id"] for r in rules if r["reassignment_type"] == "attorney"]
    total_loss_claim_ids = [r["claim_id"] for r in rules if r["is_total_loss"]]
    umpd_claim_ids = [r["claim_id"] for r in rules if r["is_umpd_claim"]]
    claim_type_lookup = {r["claim_id"]: r["claim_type"] for r in rules}
    write_csv(rules, "synthetic claim business rules v7.csv")

    comms = generate_communications(claim_ids, attorney_claim_ids, claim_type_lookup, total_loss_claim_ids, umpd_claim_ids)
    write_csv(comms, "synthetic communications v7.csv")

    insured_msgs = [r for r in comms if r["sender_type"] == "insured"]
    print("\nSender type mix:", Counter(r["sender_type"] for r in comms))
    print("Channel mix:", Counter(r["channel"] for r in comms))
    print("Calls (priority boost):", sum(1 for r in comms if r["priority_boost_called_in"]))

    urgent_callback_ct = sum(1 for r in insured_msgs if r["true_category"] == "urgent_callback_request")
    print(f"\nInsured messages: {len(insured_msgs)}")
    print(f"Urgent callback requests: {urgent_callback_ct} "
          f"({urgent_callback_ct/len(insured_msgs)*100:.1f}% of insured messages)")

    print("\nFallback-to-task messages:", sum(1 for r in comms if r["requires_fallback_task"]))
    print("Referral-needed messages:", sum(1 for r in comms if r["is_referral"]))
    print("Referral category mix:",
          Counter(r["true_category"] for r in comms if r["is_referral"]))

    print("\nInspection method mix:", Counter(r["inspection_method"] for r in rules))
    print("Supplement auto-send (virtual + non-network indicated):",
          sum(1 for r in rules if r["supplement_instructions_auto_send"]))
    print("\nUMPD claims:", sum(1 for r in rules if r["is_umpd_claim"]))
    print("  review needed:", sum(1 for r in rules if r["umpd_review_needed"]))
    print("  filed anyway (caused property damage, no coverage):",
          sum(1 for r in rules if r["umpd_claim_filed"] is True and r["insured_has_umpd_coverage"] is False))
    print("  not filed (no coverage, no property damage caused):",
          sum(1 for r in rules if r["umpd_claim_filed"] is False))
    print("\nTotal loss claims with adjuster-number note:",
          sum(1 for r in rules if r["total_loss_adjuster_number_note"]))

    # --- consistency check: this is the whole point of v6 ---
    attorney_msg_claims = {r["claim_id"] for r in comms if r["attorney_flag"]}
    rules_attorney_claims = set(attorney_claim_ids)
    print(f"\n[Consistency check] Claims with reassignment_type=='attorney' in rules: "
          f"{len(rules_attorney_claims)}")
    print(f"[Consistency check] Claims with an attorney-flagged message: "
          f"{len(attorney_msg_claims)}")
    print(f"[Consistency check] Mismatch (message says attorney, rules doesn't): "
          f"{len(attorney_msg_claims - rules_attorney_claims)}")

    print("\nPer-category message counts (all categories, sorted, lowest first):")
    for cat, ct in sorted(Counter(r["true_category"] for r in comms).items(), key=lambda x: x[1]):
        print(f"  {cat}: {ct}")

    print("\nSample placeholder phone number:", fake_phone())


if __name__ == "__main__":
    main()
