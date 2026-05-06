"""
Burst Load Testing Script
Designed to stress-test the API endpoint by firing a massive wave of 
concurrent requests and validating execution within strict latency constraints.
"""
import asyncio
import time

import httpx
import pytest

# Constants for load testing
TARGET_URL = "http://127.0.0.1:8000/get_recommendation"
NUM_CONCURRENT_REQUESTS = 1200
MAX_LATENCY_SECONDS = 1.0

@pytest.mark.asyncio
async def test_burst_recommendations():
    """
    Simulates a high-traffic scenario by sending 1,200 simultaneous POST requests.
    Validates both the functional correctness of the response payload and 
    the strict < 1.0s latency requirement for the entire batch.
    """
    # 1. Generate a synthetic dataset of 1,200 valid ContentSubmission payloads
    payloads = [
        {
            "content_id": f"content_{i}",
            "creator_id": i,
            "content_type": "SHORT" if i % 2 == 0 else "LONG",
            "created_timestamp": int(time.time())
        }
        for i in range(NUM_CONCURRENT_REQUESTS)
    ]
    
    # 2. Configure HTTP client limits to prevent connection pool exhaustion
    limits = httpx.Limits(
        max_connections=2000, 
        max_keepalive_connections=2000
    )
    
    # 3. Establish the async HTTP client session
    async with httpx.AsyncClient(limits=limits, timeout=15.0) as client:
        # Start the precise latency timer
        start_time = time.perf_counter()
        
        # Dispatch all 1,200 requests to the event loop concurrently
        tasks = [client.post(TARGET_URL, json=p) for p in payloads]
        
        # Await the completion of the entire batch
        responses = await asyncio.gather(*tasks)
        
        # Stop the latency timer
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # 4. Performance Assertion: Enforce the 1.0 second maximum latency
        assert elapsed_time < MAX_LATENCY_SECONDS, \
            f"Latency constraint violated! Batch took {elapsed_time:.4f}s (Limit: {MAX_LATENCY_SECONDS}s)"
        
        # 5. Functional Assertion: Validate all 1,200 responses for accuracy
        for response in responses:
            # Ensure the request was successfully processed
            assert response.status_code == 200, \
                f"Unexpected status code: {response.status_code} - Body: {response.text}"
            
            data = response.json()
            
            # Verify the response matches the RecommendationOutput schema structure
            assert "content_id" in data, "Missing content_id in response"
            assert "platform" in data, "Missing platform in response"
            assert "time_slot" in data, "Missing time_slot in response"
            assert "decision" in data, "Missing decision in response"
            
            # Verify data boundaries and enum constraints
            assert data["platform"] in ["Instagram", "YouTube"], "Invalid platform enum"
            assert data["decision"] in ["POST_NOW", "SCHEDULE"], "Invalid decision enum"
            assert 0 <= data["time_slot"] <= 23, "time_slot out of 24-hour bounds"
