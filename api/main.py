from __future__ import annotations

import logging

from fastapi import FastAPI

from api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="Intent Miner",
    version="1.0.0",
    description="Describe a solution; get ranked links to people describing the pain it solves.",
)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
