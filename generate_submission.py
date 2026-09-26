"""
generate_submission.py -- magicpin AI Challenge
================================================
Runs compose() on all 30 canonical test pairs and writes submission.jsonl.

Features:
  - Rate limiting: 13s delay between calls (stays under 5 req/min free tier)
  - Retry with backoff: retries 429 errors up to 3 times with 65s wait
  - Resume: skips already-generated pairs from existing submission.jsonl

Usage:
  $env:GEMINI_API_KEY = 'AIza...'
  python generate_submission.py
"""
from __future__ import annotations

import json
import time
import pathlib
import sys
from bot import compose, load_context

DATASET_DIR = "dataset/expanded"
TEST_PAIRS_FILE = "dataset/expanded/test_pairs.json"
OUTPUT_FILE = "submission.jsonl"

DELAY_BETWEEN_CALLS = 13   # seconds between calls (keeps under 5 req/min)
MAX_RETRIES = 3
RETRY_WAIT = 65             # seconds to wait on 429


def load_existing(output_path: pathlib.Path) -> set:
    """Return set of test_ids already written to submission.jsonl."""
    done = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    if row.get("test_id"):
                        done.add(row["test_id"])
                except json.JSONDecodeError:
                    pass
    return done


def compose_with_retry(category, merchant, trigger, customer) -> dict:
    """Call compose() with retry on rate-limit (429) errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return compose(category, merchant, trigger, customer)
        except Exception as exc:
            err = str(exc)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < MAX_RETRIES:
                wait = RETRY_WAIT * attempt
                print(f"\n    429 rate limit. Waiting {wait}s (attempt {attempt}/{MAX_RETRIES})...",
                      end=" ", flush=True)
                time.sleep(wait)
                print("retrying...", end=" ", flush=True)
            else:
                raise
    raise RuntimeError("max retries exceeded")


def main():
    pairs_path = pathlib.Path(TEST_PAIRS_FILE)
    if not pairs_path.exists():
        print("ERROR: test_pairs.json not found.")
        print("Run: python dataset/generate_dataset.py --out dataset/expanded")
        sys.exit(1)

    with open(pairs_path, encoding="utf-8") as f:
        test_pairs = json.load(f)["pairs"]

    output_path = pathlib.Path(OUTPUT_FILE)
    already_done = load_existing(output_path)
    remaining = [p for p in test_pairs if p["test_id"] not in already_done]

    model_name = "gemini-3.5-flash-lite"
    print(f"submission.jsonl: {len(already_done)}/30 done, {len(remaining)} remaining.")
    print(f"Model: {model_name} | Delay: {DELAY_BETWEEN_CALLS}s between calls")
    if already_done:
        print(f"Resuming — skipping: {sorted(already_done)}")
    print()

    new_results = []
    failed = []

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, pair in enumerate(remaining):
            test_id = pair["test_id"]
            merchant_id = pair["merchant_id"]
            trigger_id = pair["trigger_id"]
            customer_id = pair.get("customer_id")

            num = len(already_done) + i + 1
            print(f"  [{num:02d}/30] {test_id}: {merchant_id[:28]} x {trigger_id[:26]}...",
                  end=" ", flush=True)

            try:
                t0 = time.time()
                category, merchant, trigger, customer = load_context(
                    DATASET_DIR, merchant_id, trigger_id, customer_id
                )
                result = compose_with_retry(category, merchant, trigger, customer)
                elapsed = time.time() - t0

                row = {
                    "test_id": test_id,
                    "body": result.get("body", ""),
                    "cta": result.get("cta", "none"),
                    "send_as": result.get("send_as", "vera"),
                    "suppression_key": result.get("suppression_key", ""),
                    "rationale": result.get("rationale", ""),
                    "merchant_id": merchant_id,
                    "trigger_id": trigger_id,
                    "customer_id": customer_id,
                    "_elapsed_s": round(elapsed, 2),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                new_results.append(row)
                print(f"OK ({elapsed:.1f}s) | cta={row['cta']} | send_as={row['send_as']}")

            except Exception as exc:
                print(f"FAILED: {str(exc)[:80]}")
                failed.append({"test_id": test_id, "error": str(exc)[:120]})

            # Rate-limit delay (skip after last pair)
            if i < len(remaining) - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    total = len(already_done) + len(new_results)
    print()
    print("=" * 60)
    print(f"Done! {total}/30 total | {len(new_results)} new this run | {len(failed)} failed.")
    print(f"Output: {output_path.resolve()}")

    if failed:
        print(f"\nFailed (re-run to retry):")
        for f in failed:
            print(f"  {f['test_id']}: {f['error']}")

    if new_results:
        s = new_results[0]
        print(f"\nSample ({s['test_id']}):")
        print(f"  Body: {s['body'][:120]}{'...' if len(s['body'])>120 else ''}")
        print(f"  CTA: {s['cta']} | send_as: {s['send_as']}")


if __name__ == "__main__":
    main()
