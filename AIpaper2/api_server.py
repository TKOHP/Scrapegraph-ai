
import sys
import os
import uvicorn
from fastapi import FastAPI, Query
from typing import List, Optional
from pydantic import BaseModel
from time import time

# Ensure we can import from Nodes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Nodes.db_manager import DatabaseManager, AIPaper

app = FastAPI()
db_manager = DatabaseManager()

class PaperResponse(BaseModel):
    id: int
    title: str
    url: str
    pubDate: Optional[str]
    source: Optional[str]
    pdfLink: Optional[str]
    overviewLink: Optional[str]
    analysisLink: Optional[str]
    subject: Optional[str]

class EntireRequest(BaseModel):
    sources: List[str]

@app.get("/api/subscribe_from")
def get_subscribe_from():
    """Get all distinct subscribe_from from the database."""
    with db_manager._get_conn() as conn:
        cursor = conn.execute("SELECT DISTINCT subscribe_from FROM AIpaper")
        subscribe_from = [row[0] for row in cursor.fetchall() if row[0]]
    if not subscribe_from:
        return ["General"]
    return subscribe_from

@app.get("/api/papers")
def get_papers(subscribe_from: Optional[str] = None):
    """Get papers, optionally filtered by subscribe_from."""
    
    if subscribe_from == "General":
        papers = db_manager.list_papers() # Get all
    elif subscribe_from:
        # Use subscribe_from column to filter
        papers = db_manager.list_papers(subscribe_from=subscribe_from)
    else:
        papers = db_manager.list_papers()

    # Convert to response format matching newsnow expectations (mostly)
    # We map AIPaper fields to what frontend will need
    results = []
    for p in papers:
        # Use urlLink as title if subject/title is missing? 
        # Wait, AIPaper doesn't have a 'title' field! 
        # Looking at AIPaper definition: 
        # id, urlLink, source, pdfLink, mdLink, overviewLink, analysisLink, meta, publishTime, subject, type...
        # It seems 'meta' or 'subject' might contain the title? 
        # Or maybe 'urlLink' is the only identifier?
        # Let's check the schema again.
        
        # Schema:
        # urlLink: str
        # subject: str (Topic)
        # meta: str
        
        # Usually papers have titles. 
        # If 'meta' contains JSON or text with title, we might need to parse it.
        # For now, I'll use 'urlLink' or 'meta' as the title.
        
        title = p.meta if p.meta else p.urlLink
        
        results.append({
            "id": p.id,
            "title": title,
            "url": p.urlLink,
            "pubDate": p.publishTime,
            "source": p.source,
            "pdfLink": p.pdfLink,
            "overviewLink": p.overviewLink,
            "analysisLink": p.analysisLink,
            "subject": p.subject,
            "subscribe_from": p.subscribe_from
        })
    
    return {"items": results}

@app.post("/api/s/entire")
def get_entire(payload: EntireRequest):
    out = []
    now_ms = int(time() * 1000)
    for sid in payload.sources:
        papers = db_manager.list_papers(subscribe_from=sid)
        items = []
        for p in papers:
            title = p.meta if p.meta else p.urlLink
            items.append({
                "id": p.id,
                "title": title,
                "url": p.urlLink,
                "pubDate": p.publishTime,
                "source": p.source,
                "pdfLink": p.pdfLink,
                "overviewLink": p.overviewLink,
                "analysisLink": p.analysisLink,
                "subject": p.subject,
                "subscribe_from": p.subscribe_from
            })
        out.append({
            "status": "success",
            "id": sid,
            "updatedTime": now_ms,
            "items": items,
        })
    return out

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
