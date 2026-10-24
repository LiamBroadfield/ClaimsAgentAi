"""
Adjuster queue split — collapses task_queue_v8's 5-way contact_method into
a simple, dichotomous view for the adjuster: "call_needed" or "needs_review".
Everything else (no_action_needed, refer_total_loss_adjuster,
refer_attorney_desk) is routed and handled by the AI assistant directly —
it never lands in the adjuster's queue at all.

IMPORTANT: "AI-handled" does not mean undocumented. Every AI-handled task
still gets a logged action note, same as anything an adjuster does — the
assistant doesn't get to skip the paper trail just because a human didn't
personally act on it.

This is a thin layer on top of task_queue_v8's output — it doesn't
reclassify anything, it just relabels and documents what's already there.

USAGE: run task_queue_v8.py first so task_queue_v8.csv exists, then run
this file (or import build_adjuster_queue elsewhere).
"""

import pandas as pd

TASK_QUEUE_PATH = "task_queue_v8.csv"
OUTPUT_PATH = "adjuster_queue.csv"

# contact_method -> what the adjuster actually sees.
# "ai_handled" means it never reaches the adjuster's queue — the assistant
# has already acted and documented the outcome.
ADJUSTER_BUCKET_MAP = {
    "call_needed": "call_needed",
    "email_sufficient": "needs_review",
    "no_action_needed": "ai_handled",
    "refer_total_loss_adjuster": "ai_handled",
    "refer_attorney_desk": "ai_handled",
}

# Documentation templates for what the AI assistant did on each
# ai-handled contact_method — this is what gets logged to the claim file.
AI_ACTION_NOTES = {
    "no_action_needed": (
        "AI assistant reviewed this message and determined no adjuster "
        "action was required. Logged to claim file automatically."
    ),
    "refer_total_loss_adjuster": (
        "AI assistant routed this message to the total-loss adjuster "
        "queue based on claim status (is_total_loss). Logged to claim "
        "file automatically."
    ),
    "refer_attorney_desk": (
        "AI assistant routed this message to the attorney desk based on "
        "claim status (reassignment_type=attorney). Logged to claim file "
        "automatically."
    ),
}


def build_adjuster_queue(task_df: pd.DataFrame) -> pd.DataFrame:
    df = task_df.copy()

    df["adjuster_bucket"] = df["contact_method"].map(ADJUSTER_BUCKET_MAP)
    unmapped = df["adjuster_bucket"].isna()
    if unmapped.any():
        # anything not in the map falls back to needs_review rather than
        # silently vanishing from the adjuster's view
        df.loc[unmapped, "adjuster_bucket"] = "needs_review"

    def documentation(row):
        if row["adjuster_bucket"] != "ai_handled":
            return ""
        base_note = AI_ACTION_NOTES.get(
            row["contact_method"],
            "AI assistant handled this message automatically.",
        )
        model_reasoning = row.get("reasoning", "")
        if model_reasoning:
            return f"{base_note} Reasoning: {model_reasoning}"
        return base_note

    df["ai_documentation"] = df.apply(documentation, axis=1)

    return df


def main():
    tasks = pd.read_csv(TASK_QUEUE_PATH, encoding="utf-8", encoding_errors="replace")
    queue = build_adjuster_queue(tasks)

    print(f"Adjuster queue built from {len(queue)} tasks\n")
    print("Adjuster-facing bucket breakdown:")
    print(queue["adjuster_bucket"].value_counts())

    adjuster_facing = queue[queue["adjuster_bucket"].isin(["call_needed", "needs_review"])]
    ai_handled = queue[queue["adjuster_bucket"] == "ai_handled"]

    print(f"\nActually in the adjuster's queue: {len(adjuster_facing)}")
    print(f"Handled by AI assistant (documented, not in queue): {len(ai_handled)}")

    if len(ai_handled):
        print("\nAI-handled breakdown by original contact_method:")
        print(ai_handled["contact_method"].value_counts())

    queue.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved full queue (with documentation) to {OUTPUT_PATH}")

    return queue


if __name__ == "__main__":
    adjuster_queue = main()
