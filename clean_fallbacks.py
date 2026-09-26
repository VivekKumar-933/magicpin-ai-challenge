import json
from bot import compose, load_context

DATASET_DIR = "dataset/expanded"

with open("submission.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

updated = 0
for r in rows:
    if "JSON parse fallback" in r.get("rationale", "") or r.get("body", "").startswith("{\n"):
        tid = r["test_id"]
        mid = r["merchant_id"]
        trgid = r["trigger_id"]
        cid = r.get("customer_id")
        print(f"Regenerating {tid}: {mid} x {trgid}...")
        cat, merch, trg, cust = load_context(DATASET_DIR, mid, trgid, cid)
        res = compose(cat, merch, trg, cust)
        r["body"] = res.get("body", "")
        r["cta"] = res.get("cta", "none")
        r["send_as"] = res.get("send_as", "vera")
        r["suppression_key"] = res.get("suppression_key", "")
        r["rationale"] = res.get("rationale", "")
        updated += 1
        print(f"  -> OK: {r['body'][:60]}...")

with open("submission.jsonl", "w", encoding="utf-8") as f:
    for r in sorted(rows, key=lambda x: int(x["test_id"][1:])):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nFinished cleanly regenerating {updated} items!")
