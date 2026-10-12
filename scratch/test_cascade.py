import requests
import json

tests = [
    ("calculate the square root of 144", None),
    ("return a json object with fields 'book_title', 'author', and 'publication_year'", None),
    ("write code for finding perfect squares in a given range of numbers", None),
    ("write an email applying for leave for 3 days due to family commitments", None),
]

for q, intent in tests:
    print("=" * 70)
    print(f"QUERY: {q}")
    res = requests.post("http://127.0.0.1:5000/api/chat", json={"query": q, "intent": intent})
    data = res.json()
    r = data.get("routing", {})
    print(f"INTENT: {r.get('intent')} ({r.get('confidence_pct')})")
    print(f"TARGET HANDLER: {r.get('handler_name')}")
    print(f"CASCADE FALLBACK TRIGGERED: {r.get('fallback_triggered')}")
    print(f"ORIGINAL HANDLER: {r.get('original_handler')}")
    print(f"EXECUTION MODE: {r.get('execution_type')}")
    print(f"GEMINI API: {r.get('gemini_api')}")
    print(f"TOTAL LATENCY: {r.get('total_latency_ms')} ms")
    print(f"COST: ${r.get('cost_usd', 0.0):.6f}")
    snippet = str(data.get("response", "")).replace("\n", " ")[:140]
    print(f"RESPONSE PREVIEW: {snippet}...")
