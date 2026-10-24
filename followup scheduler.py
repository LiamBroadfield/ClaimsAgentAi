"""
Follow-up scheduling engine.

Given a claim's current status fields, computes when its next follow-up
should happen (in business days from a reference date), or whether it
needs no follow-up at all — either because something already closed the
loop (payment issued, shop paid) or because it's gone quiet long enough
to recommend closing it out.

Rules (in priority order — first match wins):
  1. Suppression: initial_estimate_payment_issued, or repair_status ==
     "shop_paid" -> no follow-up needed, loop is already closed.
  2. Stale claim: business_days_since_last_contact >=
     NO_RESPONSE_CLOSURE_DAYS -> recommend closing the claim out, customer
     can be told it's reopenable when they're ready to pursue it.
  3. escalation_needed or fraud_pending_assignment -> next business day.
  4. repair_status == "done_awaiting_shop_payment" -> 2 business days.
  5. repair_status == "scheduled_future" -> fixed 1-or-2-week follow-up,
     chosen so it always lands before the actual appointment (see
     REPAIR_APPOINTMENT_BUCKETS below). Falls back to the standard 3-day
     rule if the appointment itself is too close for a week-out check-in
     to make sense.
  6. repair_status == "in_progress" (vehicle being worked on) -> 5
     business days (about a week).
  7. Everything else (not yet scheduled, awaiting contact, general
     follow-up, waiting on another party) -> 3 business days.

USAGE: same folder as synthetic_claim_business_rules_v7.csv. Run the
whole file, or import build_followup_queue() elsewhere.
"""

from datetime import date, timedelta
import pandas as pd

RULES_PATH = "synthetic claim business rules v7.csv"
OUTPUT_PATH = "followup_queue.csv"

NO_RESPONSE_CLOSURE_DAYS = 20  # ~4 business weeks with no movement

# (min_business_days_out_exclusive, follow_up_business_days) — checked in
# order; the first bucket the appointment clears wins. This guarantees the
# follow-up always lands before the appointment itself.
REPAIR_APPOINTMENT_BUCKETS = [
    (10, 10),  # appointment >10 business days out -> follow up in 2 weeks
    (5, 5),    # appointment >5 business days out  -> follow up in 1 week
]


def add_business_days(start_date: date, n: int) -> date:
    """Add n business days (Mon-Fri) to start_date. Does not account for
    holidays — worth layering in later if this needs to be exact."""
    d = start_date
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def compute_followup(rules_row: pd.Series, reference_date: date = None) -> dict:
    """Return the follow-up decision for one claim."""
    if reference_date is None:
        reference_date = date.today()

    # --- 0. total-loss claims are a separate salvage track, handled by the
    # total-loss adjuster — not scored by this engine at all ---
    if rules_row.get("is_total_loss"):
        return {
            "action": "handled_by_total_loss_track",
            "follow_up_business_days": None,
            "follow_up_date": None,
            "reason": "Total-loss claim — follows the salvage track, "
                      "handled by the total-loss adjuster, not standard "
                      "follow-up scheduling.",
        }

    # --- 1. suppression: loop already closed — but only if there's no
    # open subrogation potential. Subrogation applies whenever the insured
    # isn't at fault/liable, and stays open regardless of which payment
    # path (non-network estimate vs. in-network shop) closed the repair
    # side of the claim. ---
    subrogation_open = bool(rules_row.get("subrogation_potential")) and pd.notna(
        rules_row.get("subrogation_potential")
    )

    if not subrogation_open:
        if rules_row.get("initial_estimate_payment_issued"):
            return {
                "action": "no_followup_needed",
                "follow_up_business_days": None,
                "follow_up_date": None,
                "reason": "Initial estimate payment already issued, no "
                          "subrogation potential.",
            }
        if rules_row.get("repair_status") == "shop_paid":
            return {
                "action": "no_followup_needed",
                "follow_up_business_days": None,
                "follow_up_date": None,
                "reason": "Network shop has already been paid, no "
                          "subrogation potential.",
            }

    # --- 2. stale claim: recommend closure instead of another follow-up ---
    days_since_contact = rules_row.get("business_days_since_last_contact", 0)
    if pd.notna(days_since_contact) and days_since_contact >= NO_RESPONSE_CLOSURE_DAYS:
        return {
            "action": "recommend_closure",
            "follow_up_business_days": None,
            "follow_up_date": None,
            "reason": (
                f"No response in {int(days_since_contact)} business days — "
                f"recommend closing the claim; customer can be informed it's "
                f"reopenable when they're ready to pursue it."
            ),
        }

    # --- 3. escalation / fraud: next business day ---
    if rules_row.get("escalation_needed") or rules_row.get("fraud_pending_assignment"):
        days = 1
        reason = (
            "Escalation needed — flagged for manager review."
            if rules_row.get("escalation_needed")
            else "Fraud flag pending assignment."
        )
        return {
            "action": "schedule_followup",
            "follow_up_business_days": days,
            "follow_up_date": add_business_days(reference_date, days),
            "reason": reason,
        }

    repair_status = rules_row.get("repair_status")

    # --- 4. shop awaiting direct deposit, repairs already done ---
    if repair_status == "done_awaiting_shop_payment":
        days = 2
        return {
            "action": "schedule_followup",
            "follow_up_business_days": days,
            "follow_up_date": add_business_days(reference_date, days),
            "reason": "Repairs complete, shop awaiting direct deposit payment.",
        }

    # --- 5. repair appointment scheduled — fixed 1-or-2-week check-in ---
    if repair_status == "scheduled_future":
        appt_days_out = rules_row.get("repair_appointment_business_days_out")
        if pd.notna(appt_days_out) and appt_days_out != "":
            appt_days_out = int(appt_days_out)
            for threshold, followup_days in REPAIR_APPOINTMENT_BUCKETS:
                if appt_days_out > threshold:
                    return {
                        "action": "schedule_followup",
                        "follow_up_business_days": followup_days,
                        "follow_up_date": add_business_days(reference_date, followup_days),
                        "reason": (
                            f"Repair appointment ~{appt_days_out} business days out — "
                            f"checking back in {followup_days} business days, ahead of it."
                        ),
                    }
        # appointment too close (or missing) for a week-out check-in —
        # fall through to the standard 3-day rule below

    # --- 6. vehicle being worked on ---
    if repair_status == "in_progress":
        days = 5
        return {
            "action": "schedule_followup",
            "follow_up_business_days": days,
            "follow_up_date": add_business_days(reference_date, days),
            "reason": "Vehicle currently in repair — standard 1-week check-in.",
        }

    # --- 7. default: awaiting contact / general follow-up / other party ---
    days = 3
    return {
        "action": "schedule_followup",
        "follow_up_business_days": days,
        "follow_up_date": add_business_days(reference_date, days),
        "reason": "Standard follow-up — awaiting contact or response from another party.",
    }


def build_followup_queue(rules: pd.DataFrame, reference_date: date = None) -> pd.DataFrame:
    if reference_date is None:
        reference_date = date.today()

    results = []
    for _, row in rules.iterrows():
        decision = compute_followup(row, reference_date)
        results.append({
            "claim_id": row["claim_id"],
            **decision,
        })

    return pd.DataFrame(results)


def main():
    rules = pd.read_csv(RULES_PATH, encoding="utf-8", encoding_errors="replace")
    today = date.today()

    queue = build_followup_queue(rules, reference_date=today)

    print(f"Follow-up decisions computed for {len(queue)} claims "
          f"(reference date: {today})\n")
    print("Action breakdown:")
    print(queue["action"].value_counts())

    scheduled = queue[queue["action"] == "schedule_followup"]
    if len(scheduled):
        print("\nFollow-up business-day breakdown (scheduled only):")
        print(scheduled["follow_up_business_days"].value_counts().sort_index())

    closures = queue[queue["action"] == "recommend_closure"]
    print(f"\nRecommended for closure (no response {NO_RESPONSE_CLOSURE_DAYS}+ "
          f"business days): {len(closures)}")

    queue.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    return queue


if __name__ == "__main__":
    followup_queue = main()
