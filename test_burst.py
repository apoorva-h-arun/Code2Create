import asyncio
import time
import pytest
import httpx

@pytest.mark.asyncio
async def test_burst_recommendations():
    # Target URL for the local FastAPI server
    url = "http://127.0.0.1:8000/get_recommendation"
    num_requests = 1200
    
    # Generate synthetic payload of 1,200 valid ContentSubmission items
    payloads = [
        {
            "content_id": f"content_{i}",
            "creator_id": i,
            "content_type": "SHORT" if i % 2 == 0 else "LONG",
            "created_timestamp": int(time.time())
        }
        for i in range(num_requests)
    ]
    
    # Configure client limits to allow massive concurrency
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
    
    async with httpx.AsyncClient(limits=limits, timeout=15.0) as client:
        start_time = time.perf_counter()
        
        # Use asyncio.gather to fire all 1,200 requests concurrently
        tasks = [client.post(url, json=p) for p in payloads]
        responses = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # Assert that the total elapsed time is < 1.0 seconds
        assert elapsed_time < 1.0, f"Batch completed in {elapsed_time:.4f} seconds, exceeding 1.0s limit"
        
        # Assert that all 1,200 responses have a status code of 200 and match the RecommendationOutput schema
        for response in responses:
            assert response.status_code == 200, f"Unexpected status code: {response.status_code} - {response.text}"
            data = response.json()
            assert "content_id" in data
            assert "platform" in data
            assert "time_slot" in data
            assert "decision" in data
            assert data["platform"] in ["Instagram", "YouTube"]
            assert data["decision"] in ["POST_NOW", "SCHEDULE"]
            assert 0 <= data["time_slot"] <= 23
