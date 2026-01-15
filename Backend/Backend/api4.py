# main.py
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

app = FastAPI(title="StaticReplyChatAPI")

# Allow CORS from anywhere (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # allow all origins
    allow_credentials=True,
    allow_methods=["*"],            # allow all HTTP methods
    allow_headers=["*"],            # allow all headers
)

@app.post("/generate-report-insights")
async def generate_report_insights(request: Request):
    """
    Accepts any input (JSON or plain text).
    Waits for 10 seconds then responds with the fixed string.
    """
    try:
        _ = await request.json()
    except Exception:
        try:
            _ = await request.body()
        except Exception:
            _ = None

    # Wait for 10 seconds
    await asyncio.sleep(10)

    return PlainTextResponse('''
Here are the recommendations in a more condensed format.

1. Decarbonize Process Energy ☀️
Switch the electrolytic reduction (smelting) process to renewable energy via a Power Purchase Agreement (PPA).
 This directly mitigates your primary Scope 2 emissions hotspot, cutting the product's total Global Warming 
Potential by ~8%.

2. Optimize Material Feedstock ♻️
Implement a circular material flow by specifying the handle be made from 100% secondary (recycled) aluminum. 
This move eliminates the impacts of virgin bauxite for that part, reducing total Mineral Resource Scarcity by 50% 
and Particulate Matter by over 26%.

3. Install Emission Abatement Systems 💨
Retrofit your facility with Best Available Technology (BAT) like scrubbers to control fugitive heavy metal emissions.
 This targets the source of 100% of the Human Carcinogenic Toxicity impact, and a 90% efficient system would cut that 
score by 90%.                             
''')


if __name__ == "__main__":
    print("Starting API server...")
    print("Access API documentation at http://127.0.0.1:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8002)
