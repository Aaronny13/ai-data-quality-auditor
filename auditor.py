#!/usr/bin/env python3
"""
AI Data Quality Auditor
------------------------
A Human-in-the-Loop tool for auditing AI-generated prompt/response pairs.

What it does:
1. Reads a CSV of prompt/response pairs (e.g. AI model outputs to be QA'd).
2. Sends each pair to an LLM (via NVIDIA NIM) for an automated
   factual-accuracy check, scored against a simple atomic rubric.
3. Produces a structured report with a Data Quality Score per row, flags
   items that need human review, and summarizes overall accuracy.

Author: Aaron Sackitey Teye
"""

import csv
import json
import time
import sys
from datetime import datetime
from openai import OpenAI

# ── CONFIG ────────────────────────────────────────────────────────────────
import os

# Reads the key from an environment variable (NVIDIA_API_KEY) if set, so the
# real key never has to live in this file or get committed to GitHub.
# To set it on Windows (PowerShell), before running the script:
#   $env:NVIDIA_API_KEY = "nvapi-yourrealkey"
# Or just paste your key directly into the quotes below for local-only use
# — but if you do that, never push this file to a public repo as-is.
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "PASTE_YOUR_NVIDIA_API_KEY_HERE")
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

INPUT_FILE = "dataset.csv"
OUTPUT_FILE = "audit_report.csv"
SUMMARY_FILE = "audit_summary.txt"

# Rows scoring below this are auto-flagged for human review
REVIEW_THRESHOLD = 70

# Seconds to wait between each API call.
SECONDS_BETWEEN_CALLS = 3

# How many times to retry a single row if we hit a rate limit before
# giving up on that row and moving to the next one.
MAX_RETRIES_ON_RATE_LIMIT = 3

# ── ATOMIC RUBRIC ────────────────────────────────────────────────────────
# Each response is scored 0-100 against these atomic checks. This mirrors
# the kind of rubric structure used in real annotation/eval pipelines:
# break the judgment into discrete, checkable criteria rather than one
# vague "is this good?" question.
RUBRIC_PROMPT = """You are a strict fact-checking auditor. Evaluate the RESPONSE to the
PROMPT below using this atomic rubric. Score each criterion honestly.

Criteria:
1. Factual Accuracy (0-50 pts): Is every factual claim in the response correct?
   Deduct heavily for any false claim, myth, or common misconception stated as fact.
2. Completeness (0-20 pts): Does the response actually answer the question asked?
3. Clarity (0-15 pts): Is the response clearly written and unambiguous?
4. Hedging/Overconfidence (0-15 pts): Does the response avoid stating disputed or
   false claims with unwarranted confidence?

PROMPT: {prompt}

RESPONSE: {response}

Return ONLY raw valid JSON in this exact format. Do not use markdown code fences,
do not add any explanation before or after, output the JSON object only:
{{
  "factual_accuracy": <0-50>,
  "completeness": <0-20>,
  "clarity": <0-15>,
  "hedging_score": <0-15>,
  "total_score": <0-100>,
  "contains_inaccuracy": <true or false>,
  "issue_summary": "<one short sentence describing the main issue, or 'No issues found'>"
}}
"""


def call_llm(prompt_text, response_text):
    """Send one prompt/response pair to the NVIDIA-hosted model for scoring.
    Returns a dict. Retries automatically with backoff if rate-limited."""
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=45.0)
    full_prompt = RUBRIC_PROMPT.format(prompt=prompt_text, response=response_text)

    for attempt in range(1, MAX_RETRIES_ON_RATE_LIMIT + 1):
        try:
            completion = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
                max_tokens=500,
            )
            raw_text = completion.choices[0].message.content.strip()

            # Models sometimes wrap JSON in markdown fences despite instructions —
            # strip those defensively before parsing.
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            return {"error": f"Could not parse model response as JSON: {e}"}
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                wait = 15 * attempt
                print(f"   ⏳ Rate limited, waiting {wait}s before retry "
                      f"({attempt}/{MAX_RETRIES_ON_RATE_LIMIT})...")
                time.sleep(wait)
                continue
            return {"error": f"{type(e).__name__}: {e}"}

    return {"error": "Rate limited repeatedly — gave up after max retries."}


def audit_dataset(input_file, output_file, summary_file):
    if NVIDIA_API_KEY == "PASTE_YOUR_NVIDIA_API_KEY_HERE":
        print("⚠️  No API key set yet. Add your NVIDIA API key to NVIDIA_API_KEY")
        print("   at the top of this script, then run it again.")
        print("   Get a free key at: https://build.nvidia.com\n")
        sys.exit(1)

    rows = []
    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} prompt/response pairs from {input_file}\n")

    results = []
    flagged_count = 0
    error_count = 0
    score_total = 0

    for i, row in enumerate(rows, start=1):
        prompt_text = row["prompt"]
        response_text = row["response"]
        print(f"[{i}/{len(rows)}] Auditing: {prompt_text[:60]}...")

        result = call_llm(prompt_text, response_text)

        if "error" in result:
            error_count += 1
            results.append({
                "prompt": prompt_text,
                "response": response_text,
                "total_score": "ERROR",
                "contains_inaccuracy": "UNKNOWN",
                "issue_summary": result["error"],
                "needs_human_review": "YES",
            })
            print(f"   ⚠️  {result['error']}")
        else:
            score = result.get("total_score", 0)
            score_total += score
            needs_review = score < REVIEW_THRESHOLD
            if needs_review:
                flagged_count += 1

            results.append({
                "prompt": prompt_text,
                "response": response_text,
                "factual_accuracy": result.get("factual_accuracy"),
                "completeness": result.get("completeness"),
                "clarity": result.get("clarity"),
                "hedging_score": result.get("hedging_score"),
                "total_score": score,
                "contains_inaccuracy": result.get("contains_inaccuracy"),
                "issue_summary": result.get("issue_summary"),
                "needs_human_review": "YES" if needs_review else "no",
            })
            flag_marker = "🚩 FLAGGED" if needs_review else "✓ pass"
            print(f"   Score: {score}/100  {flag_marker}")

        time.sleep(SECONDS_BETWEEN_CALLS)  # conservative pacing for the free tier

    # ── Write detailed report ───────────────────────────────────────────
    fieldnames = ["prompt", "response", "factual_accuracy", "completeness",
                  "clarity", "hedging_score", "total_score",
                  "contains_inaccuracy", "issue_summary", "needs_human_review"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Write summary ───────────────────────────────────────────────────
    scored_rows = [r for r in results if isinstance(r["total_score"], (int, float))]
    avg_score = round(score_total / len(scored_rows), 1) if scored_rows else 0

    summary = f"""AI DATA QUALITY AUDIT — SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Total rows audited:        {len(rows)}
Average quality score:     {avg_score}/100
Flagged for human review:  {flagged_count} ({round(flagged_count/len(rows)*100, 1)}%)
Errors during audit:       {error_count}

Review threshold: scores below {REVIEW_THRESHOLD}/100 are auto-flagged.

Next step: a human operator should review the {flagged_count} flagged rows
in {output_file} before this data is approved for downstream use.
"""

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + "=" * 60)
    print(summary)
    print(f"Full report saved to: {output_file}")
    print(f"Summary saved to:     {summary_file}")


if __name__ == "__main__":
    audit_dataset(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)
