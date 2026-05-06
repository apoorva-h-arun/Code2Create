from fastapi import FastAPI
from app.data.data_loader import load_all

app = FastAPI()
store = None

@app.on_event("startup")
def startup():
    global store
    store = load_all(
        content_path    = "data/raw/content.csv",
        activity_path   = "data/raw/platform_activity.csv",
        engagement_path = "data/raw/historical_engagement.csv",
        creators_path   = "data/raw/creators.csv",
    )