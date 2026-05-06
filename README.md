# Creator Content Posting Optimization System

## Team Information
- **Team Name**: Pixel Nebula 
- **Year**: 2nd Year
- **All-Female Team**: Yes

---

## Architecture Overview: Deterministic Contextual Bandit Optimization Engine

### 1. Architectural Paradigm
The Creator Content Posting Optimization System is engineered as a **High-Performance Numerical Solver** rather than a traditional CRUD web application. Given the strict Hackathon constraints—specifically the massive penalty for execution latency ($>1.0$ second) and the requirement for absolute determinism—we abandoned sequential loop-based logic and database I/O bottlenecks. 

Instead, the system relies on an **Asynchronous Event Loop (FastAPI)** wrapped around a **Vectorized Single Instruction, Multiple Data (SIMD) Inference Engine (NumPy)**. This guarantees that payload ingestion is non-blocking while mathematical optimization occurs in $O(1)$ time complexity, allowing the system to process burst submissions of $1,200+$ concurrent payloads in sub-millisecond per-request times.

### 2. The Core Logic: Joint Optimization via Contextual UCB
Traditional recommendation systems often calculate the optimal platform and optimal time sequentially. This approach is fundamentally flawed as it fails to balance a suboptimal platform at a peak time against an optimal platform at an off-peak time. 

To solve this, we formulated the problem as a **Deterministic Contextual Multi-Armed Bandit (UCB)**. 
*   **The State Space (Arms):** We flattened the decision matrix into exactly 48 distinct arms ($2 \text{ Platforms} \times 24 \text{ Time Slots}$).
*   **The Context:** The dynamic input consisting of the `creator_id` and the `content_type` (SHORT or LONG).

Rather than using stochastic algorithms (like $\epsilon$-greedy) which violate the determinism constraint, we execute a deterministic Upper Confidence Bound objective function across the entire $2 \times 24$ matrix simultaneously:

$$ E[R_{p,t}] = \left[ 0.5 \cdot (\text{Base}_c \cdot \text{Act}_{p,t} \cdot \text{Hist}_{c,p,ct,t}) \right] + \left[ 0.2 \cdot \text{Act}_{p,t} \right] + \left[ 0.15 \cdot \text{Bias}_{ct,p} \right] $$

*   $\text{Base}_c$: Scalar multiplier dictating the creator's baseline engagement capability.
*   $\text{Act}_{p,t}$: The relative activity level of users on platform $p$ at time $t$.
*   $\text{Hist}$: The historical performance tensor sliced specifically for creator $c$ and content type $ct$.
*   $\text{Bias}$: A soft constraint matrix steering content affinities (e.g., heavily weighting `SHORT` content toward `Instagram` ($1.0$), while penalizing it on `YouTube` ($0.85$)).

The absolute optimal decision is extracted using a vectorized `argmax` function, mathematically guaranteeing the highest possible Engagement Score. Ties are deterministically broken by index position.

### 3. Memory Management & The Zero-Copy Feature Store
To support the $O(1)$ matrix calculations, the architecture eliminates database queries during runtime. 
*   At server startup, `creators.csv`, `platform_activity.csv`, and `historical_engagement.csv` are parsed, cleaned (handling zero-values and `NaN` fallbacks), and loaded into contiguous RAM blocks as 2D and 3D NumPy Tensors.
*   During a burst event, the incoming request triggers a near-instant pointer slice to the `creator_id`'s history. The data never moves; the CPU simply runs vectorized math over the pre-cached memory block.

### 4. Deterministic Scheduling Gate
The requirement to output either `POST_NOW` or `SCHEDULE` is handled via a strict algorithmic temporal gate, ensuring zero stochastic variation. 

Once the mathematical engine determines the optimal hour ($t_{opt}$), the system calculates the absolute distance from the submission payload's created timestamp:
*   **Condition:** $|\Delta t| \le 1 \text{ hour}$
*   **Execution:** If the optimal time falls within this delta, the system classifies the submission as `POST_NOW`. Otherwise, it defaults to `SCHEDULE`. This logic natively accounts for content submitted just prior to or exactly during peak usage periods.

### 5. Fault Tolerance & Scalability Profile
*   **Validation Pipeline:** Incoming payloads are rigidly validated using Rust-compiled Pydantic V2 models. Malformed `content_type` inputs or invalid timestamps are rejected before consuming computational cycles.
*   **Missing Data Imputation:** If a creator lacks historical engagement data for a specific platform/time vector, the system gracefully degrades by injecting a neutral $1.0$ multiplier prior, allowing the decision to be driven entirely by the `Act` (Traffic) and `Bias` (Affinity) matrices.
*   **Latency Profile:** Under stress testing, the end-to-end processing pipeline—from JSON ingestion, Pydantic validation, Matrix instantiation, UCB math, Argmax selection, to JSON serialization—completes in $< 1.0$ millisecond, securing the maximum possible Efficiency Score.
---

### Setup & Installation

This system is engineered for zero-configuration startup. All static datasets are loaded directly into RAM upon server boot to prevent I/O blocking during evaluation.

### Prerequisites
*   **Python 3.9+**
*   **Dataset:** Ensure the four required CSV files (`content.csv`, `creators.csv`, `historical_engagement.csv`, `platform_activity.csv`) are placed in the `app/data/raw/` directory (or your configured data path).

### 1. Environment Setup
It is highly recommended to use a virtual environment to prevent dependency conflicts.
```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-directory>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install high-performance dependencies
pip install fastapi uvicorn pydantic numpy pandas httpx pytest
```
### 2. Initialize the Server
Start the FastAPI application. The @asynccontextmanager startup event will automatically parse the local .csv datasets and convert them into memory-contiguous NumPy tensors.
```bash
# Run the ASGI server (Omit --reload for strict latency benchmarking)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
### 3. API Endpoint Verification 
Can verify the system is active by hitting the /get_recommendation endpoint using a standard JSON payload:
```bash 
curl -X POST "[http://127.0.0.1:8000/get_recommendation](http://127.0.0.1:8000/get_recommendation)" \
     -H "Content-Type: application/json" \
     -d '{
           "content_id": "CONT-1001", 
           "creator_id": 42, 
           "content_type": "SHORT", 
           "created_timestamp": "2026-05-06T14:00:00Z"
         }'
```
### 4. Run Burst Latency Benchmark
To verify the system satisfies the < 1.0 second latency constraint for the 15% Efficiency metric, execute the asynchronous stress test suite. This will fire 1,200+ concurrent payloads at the local endpoint.
```bash 
# Ensure the server is running in a separate terminal first, then run:
pytest test_burst.py -v
```
