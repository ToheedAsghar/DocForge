import time
import asyncio
from backend.agents.graph import run_graph
from backend.logger import get_logger
from backend.spinner import Spinner

logger = get_logger(__name__)

async def main():
    # First query (no cache)
    logger.info("First query (cold cache)...")
    start = time.time()
    async with Spinner("Running first query (cold cache)..."):
        result1 = await run_graph("What's Manus in detail of more than 500 characters?")
    time1 = time.time() - start
    logger.info(f"Time: {time1:.2f}s")

    print(result1.get('synthesized_answer', 'No answer generated'))

    # Second query (with cache)
    logger.info("Second query (warm cache)...")
    start = time.time()
    async with Spinner("Running second query (warm cache)...", style="dots"):
        result2 = await run_graph("What's Manus in detail of more than 500 characters?")
    time2 = time.time() - start
    logger.info(f"Time: {time2:.2f}s")
    logger.info(f"Speedup: {time1/time2:.1f}x faster!")
    logger.info(f"From cache: {result2.get('_from_cache', False)}")

    print(result2.get('synthesized_answer', 'No answer generated'))

asyncio.run(main())
