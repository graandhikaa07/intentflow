import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.query_router import QueryRouter

router = QueryRouter()

queries = [
    "What is the capital of Australia?",
    "Who was the first president of the United States?",
    "What is 15 + 27?",
    "Calculate 25 * 48",
    "Return a JSON object with keys 'product_id', 'price_usd', and 'in_stock'",
    "Give me a json of user details",
    "Tell me a joke",
    "Explain how neural networks work",
    "What is photosynthesis?",
    "Rewrite this sentence in a polite tone: send the files now",
    "Write a short poem about the ocean",
]

for q in queries:
    res = router.route_and_execute(q)
    print("=" * 60)
    print(f"Q: {q}")
    print(f"   Classified Intent: {res.get('intent')} ({res.get('confidence')})")
    print(f"   Handler: {res.get('handler')}")
    print(f"   Fallback Triggered: {res.get('fallback_triggered', False)}")
    print(f"   Original Handler: {res.get('original_handler')}")
    print(f"   Fallback Reason: {res.get('fallback_reason')}")
