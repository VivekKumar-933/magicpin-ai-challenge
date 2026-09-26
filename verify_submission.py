import json
import re

with open("submission.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

fixed_count = 0
for r in lines:
    body = r.get("body", "").strip()
    if body.startswith("{") and "}" in body:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and "body" in parsed:
                r["body"] = parsed["body"]
                if "cta" in parsed and parsed["cta"]:
                    r["cta"] = parsed["cta"]
                if "rationale" in parsed and parsed["rationale"]:
                    r["rationale"] = parsed["rationale"]
                fixed_count += 1
        except Exception:
            pass

# Ensure test_id sorting T01..T30
lines = sorted(lines, key=lambda x: int(x["test_id"][1:]))

with open("submission.jsonl", "w", encoding="utf-8") as f:
    for r in lines:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Total lines: {len(lines)}")
print(f"Fixed nested json entries: {fixed_count}")

# Print all 30 summaries
for r in lines:
    print(f"{r['test_id']}: [{r['send_as']}] (CTA: {r['cta']}) -> {r['body'][:75]}...")
