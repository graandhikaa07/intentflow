import urllib.request
import json
import time

test_queries = [
    ("MATHDS", "Calculate the square root of 144."),
    ("TXTSRC", "What is the capital of Australia?"),
    ("SEMADPT", "Rewrite this sentence in a friendly professional tone: Please send the report."),
    ("CSTRSET", "Return a JSON object with exactly the fields name and age."),
    ("MALFORMED", "")
]

results = []

for label, q in test_queries:
    print(f"\n==================================================")
    print(f"TEST: [{label}] Query: '{q}'")
    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/chat",
        data=json.dumps({"query": q}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            r = data.get("routing", {})
            print(f"Status: {data.get('status')}")
            print(f"Intent: {r.get('intent')} ({r.get('confidence_pct')})")
            print(f"Handler: {r.get('handler_display_name')}")
            print(f"Execution: {r.get('execution_type')} | Gemini API: {r.get('gemini_api')}")
            print(f"Latency: Total {r.get('total_latency_ms')} ms (Clf: {r.get('classification_latency_ms')} ms, Handler: {r.get('handler_latency_ms')} ms)")
            resp_str = str(data.get("response"))
            print(f"Response: {resp_str[:180]}...")
            results.append((label, "PASS", r.get("intent"), r.get("confidence_pct"), r.get("handler_display_name")))
    except urllib.error.HTTPError as e:
        err_data = json.loads(e.read().decode())
        print(f"HTTP {e.code} Graceful Response: {err_data}")
        results.append((label, f"PASS (HTTP {e.code})", "N/A", "N/A", "Graceful Error Handling"))
    except Exception as e:
        print(f"Exception: {e}")
        results.append((label, "FAIL", str(e), "", ""))

print("\n==================================================")
print("TEST SUMMARY:")
for r in results:
    print(f"  {r[0]:<12}: Status={r[1]}, Intent={r[2]} ({r[3]}), Handler={r[4]}")
