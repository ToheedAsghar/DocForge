from backend.agents.graph import run_graph
import time

# First query (no cache)
print("First query (cold cache)...")
start = time.time()
result1 = run_graph("What's Manus in detail of more than 500 characters?")
time1 = time.time() - start
print(f"Time: {time1:.2f}s")

print(f"\n\n\n\n\n")
print(f"[INFO]\tResult: {result1.get('synthesized_answer', 'No answer generated')}")
print(f"\n\n\n\n\n")

# Second query (with cache)
print("\nSecond query (warm cache)...")
start = time.time()
result2 = run_graph("What's Manus in detail of more than 500 characters?")
time2 = time.time() - start
print(f"Time: {time2:.2f}s")

print(f"\nSpeedup: {time1/time2:.1f}x faster!")
print(f"From cache: {result2.get('_from_cache', False)}")

print(f"\n\n\n\n\n")
print(f"[INFO]\tResult: {result2.get('synthesized_answer', 'No answer generated')}")
print(f"\n\n\n\n\n")
