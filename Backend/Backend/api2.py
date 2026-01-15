# main.py
# To run this API:
# 1. Install the required libraries:
#    pip install fastapi "uvicorn[standard]"
# 2. Save this code as main.py
# 3. Run the server from your terminal:
#    uvicorn main:app --reload --port 8001
#    OR
#    python main.py

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- 1. Pydantic Models for Input Validation ---
# These models ensure that the incoming JSON data has the correct structure.

class RawMaterial(BaseModel):
    name: str
    amount: float
    unit: str

class ProcessEnergy(BaseModel):
    process: str
    energy_type: str
    amount: float
    unit: str

class Transport(BaseModel):
    mode: str
    distance: float
    unit: str

class InputParameters(BaseModel):
    RawMaterials: List[RawMaterial]
    ProcessEnergy: List[ProcessEnergy]
    Transport: List[Transport]

class Emission(BaseModel):
    substance: str
    amount: float
    unit: str

class Waste(BaseModel):
    waste: str
    amount: float
    unit: str

class OutputParameters(BaseModel):
    EmissionsToAir: List[Emission]
    EmissionsToWater: Optional[List[Emission]] = []
    EmissionsToSoil: Optional[List[Emission]] = []
    FinalWasteFlow: List[Waste]

class UsePhase(BaseModel):
    lifespan: int
    unit: str
    energy_per_year: Any

class EndOfLife(BaseModel):
    Recycling: int
    Reuse: int
    Landfill: int
    unit: str

class UseAndEndOfLife(BaseModel):
    UsePhase: UsePhase
    EndOfLife: EndOfLife

class ProductComponent(BaseModel):
    name: str
    InputParameters: InputParameters
    OutputParameters: OutputParameters
    UseAndEndOfLife: UseAndEndOfLife

# --- 2. Data for LCA Factors (Database Simulation) ---
# This data is now expanded to handle the more detailed input JSON.
# MJ to kWh conversion factor
MJ_TO_KWH = 1 / 3.6

# Emission factors (kg of substance per unit of input)
EMISSION_FACTORS = {
    "RawMaterials": {
        "Bauxite": {"CO2": 0.05, "SOx": 0.001, "NOx": 0.002, "Particulates": 0.01},
        "Scrap Aluminium": {"CO2": 0.1, "SOx": 0.0001, "NOx": 0.0001, "Particulates": 0.001},
    },
    "ProcessEnergy": {
        "Coal (Thermal)": {"CO2": 0.09, "SOx": 0.0005, "NOx": 0.0002, "Particulates": 0.001}, # per MJ
        "Grid Mix (Electrical)": {"CO2": 0.4, "SOx": 0.001, "NOx": 0.0005, "Particulates": 0.0002},  # per kWh
    },
    "Transport": {
        "Ship": {"CO2": 0.015, "SOx": 0.0001, "NOx": 0.0002, "Particulates": 0.00001}, # per km
        "Rail": {"CO2": 0.025, "SOx": 0.00005, "NOx": 0.0001, "Particulates": 0.00002},# per km
        "Truck": {"CO2": 0.1, "SOx": 0.00001, "NOx": 0.00015, "Particulates": 0.000005}, # per km
    }
}

# ReCiPe (Hierarchist) Characterization Factors
CHARACTERIZATION_FACTORS = {
    "Global_Warming_Potential": {"CO2": 1.0},
    "Terrestrial_Acidification": {"SOx": 1.0, "SO2": 1.0, "NOx": 0.7},
    "Particulate_Matter_Formation": {"Particulates": 1.0, "PM2.5": 1.0},
    # Assumption: 25 kg 1,4-DCB-eq / kg of emission for any heavy metal listed
    "Human_Carcinogenic_Toxicity": {"Heavy Metals (e.g., Pb, Hg)": 25.0},
    # Derived from user's example: (2.86 kg Al / 10.8 kg Bauxite) * 17.1 kg Cu-eq/kg Al
    "Mineral_Resource_Scarcity": {"Bauxite": 4.52} # kg Cu-eq / kg of Bauxite
}

# ReCiPe (Hierarchist) Normalization Factors for a global average person
NORMALIZATION_FACTORS = {
    "Global_Warming_Potential": 9820,
    "Terrestrial_Acidification": 41.5,
    "Particulate_Matter_Formation": 13.9,
    "Human_Carcinogenic_Toxicity": 1110,
    "Mineral_Resource_Scarcity": 1395.7
}

IMPACT_UNITS = {
    "Global_Warming_Potential": "kg CO2-eq",
    "Terrestrial_Acidification": "kg SO2-eq",
    "Particulate_Matter_Formation": "kg PM2.5-eq",
    "Human_Carcinogenic_Toxicity": "kg 1,4-DCB-eq",
    "Mineral_Resource_Scarcity": "kg Cu-eq"
}

# --- 3. FastAPI Application Setup ---
app = FastAPI(
    title="Detailed Life Cycle Assessment (LCA) API",
    description="Calculates detailed LCA impacts with source breakdowns.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 4. Core Calculation Logic ---

def calculate_detailed_lci(components: List[ProductComponent]) -> Dict[str, Any]:
    """
    Calculates detailed LCI results with a breakdown of contributions for each impact category.
    """
    # Initialize a dictionary to hold the breakdown for each impact category
    impact_breakdowns = {cat: [] for cat in IMPACT_UNITS}

    for component in components:
        # A. Raw Materials
        for rm in component.InputParameters.RawMaterials:
            factors = EMISSION_FACTORS["RawMaterials"].get(rm.name, {})
            
            # GWP
            impact = rm.amount * factors.get("CO2", 0) * CHARACTERIZATION_FACTORS["Global_Warming_Potential"].get("CO2", 0)
            if impact > 0: impact_breakdowns["Global_Warming_Potential"].append({"source": f"{rm.name} ({component.name})", "type": "Raw Material", "score": impact})
            
            # Acidification
            impact = (rm.amount * factors.get("SOx", 0) * CHARACTERIZATION_FACTORS["Terrestrial_Acidification"].get("SOx", 0)) + \
                     (rm.amount * factors.get("NOx", 0) * CHARACTERIZATION_FACTORS["Terrestrial_Acidification"].get("NOx", 0))
            if impact > 0: impact_breakdowns["Terrestrial_Acidification"].append({"source": f"{rm.name} ({component.name})", "type": "Raw Material", "score": impact})
            
            # Particulates
            impact = rm.amount * factors.get("Particulates", 0) * CHARACTERIZATION_FACTORS["Particulate_Matter_Formation"].get("Particulates", 0)
            if impact > 0: impact_breakdowns["Particulate_Matter_Formation"].append({"source": f"{rm.name} ({component.name})", "type": "Raw Material", "score": impact})
            
            # Mineral Scarcity
            impact = rm.amount * CHARACTERIZATION_FACTORS["Mineral_Resource_Scarcity"].get(rm.name, 0)
            if impact > 0: impact_breakdowns["Mineral_Resource_Scarcity"].append({"source": f"{rm.name} ({component.name})", "type": "Raw Material", "score": impact})

        # B. Process Energy
        for pe in component.InputParameters.ProcessEnergy:
            amount = pe.amount * (MJ_TO_KWH if pe.unit == "MJ" else 1)
            factors = EMISSION_FACTORS["ProcessEnergy"].get(pe.energy_type, {})
            source_name = f"{pe.process} ({pe.energy_type}) ({component.name})"
            
            # GWP
            impact = amount * factors.get("CO2", 0) * CHARACTERIZATION_FACTORS["Global_Warming_Potential"].get("CO2", 0)
            if impact > 0: impact_breakdowns["Global_Warming_Potential"].append({"source": source_name, "type": "Process Energy", "score": impact})
           
            # Acidification, etc.
            impact = (amount * factors.get("SOx", 0) * CHARACTERIZATION_FACTORS["Terrestrial_Acidification"].get("SOx", 0)) + \
                     (amount * factors.get("NOx", 0) * CHARACTERIZATION_FACTORS["Terrestrial_Acidification"].get("NOx", 0))
            if impact > 0: impact_breakdowns["Terrestrial_Acidification"].append({"source": source_name, "type": "Process Energy", "score": impact})
            
            impact = amount * factors.get("Particulates", 0) * CHARACTERIZATION_FACTORS["Particulate_Matter_Formation"].get("Particulates", 0)
            if impact > 0: impact_breakdowns["Particulate_Matter_Formation"].append({"source": source_name, "type": "Process Energy", "score": impact})

        # C. Transport
        for t in component.InputParameters.Transport:
            factors = EMISSION_FACTORS["Transport"].get(t.mode, {})
            source_name = f"{t.mode} ({component.name})"
            
            impact = t.distance * factors.get("CO2", 0) * CHARACTERIZATION_FACTORS["Global_Warming_Potential"].get("CO2", 0)
            if impact > 0: impact_breakdowns["Global_Warming_Potential"].append({"source": source_name, "type": "Transport", "score": impact})
            
            impact = (t.distance * factors.get("SOx", 0) * CHARACTERIZATION_FACTORS["Terrestrial_Acidification"].get("SOx", 0)) + \
                     (t.distance * factors.get("NOx", 0) * CHARACTERIZATION_FACTORS["Terrestrial_Acidification"].get("NOx", 0))
            if impact > 0: impact_breakdowns["Terrestrial_Acidification"].append({"source": source_name, "type": "Transport", "score": impact})
            
            impact = t.distance * factors.get("Particulates", 0) * CHARACTERIZATION_FACTORS["Particulate_Matter_Formation"].get("Particulates", 0)
            if impact > 0: impact_breakdowns["Particulate_Matter_Formation"].append({"source": source_name, "type": "Transport", "score": impact})
            
        # D. Direct Emissions (from OutputParameters)
        for emission in component.OutputParameters.EmissionsToAir:
            source_name = f"Direct {emission.substance} ({component.name})"
            impact = emission.amount * CHARACTERIZATION_FACTORS["Global_Warming_Potential"].get(emission.substance, 0)
            if impact > 0: impact_breakdowns["Global_Warming_Potential"].append({"source": source_name, "type": "Direct Emission", "score": impact})
            
            impact = emission.amount * CHARACTERIZATION_FACTORS["Terrestrial_Acidification"].get(emission.substance, 0)
            if impact > 0: impact_breakdowns["Terrestrial_Acidification"].append({"source": source_name, "type": "Direct Emission", "score": impact})
            
            impact = emission.amount * CHARACTERIZATION_FACTORS["Particulate_Matter_Formation"].get(emission.substance, 0)
            if impact > 0: impact_breakdowns["Particulate_Matter_Formation"].append({"source": source_name, "type": "Direct Emission", "score": impact})
            
        if component.OutputParameters.EmissionsToWater:
            for emission in component.OutputParameters.EmissionsToWater:
                source_name = f"Direct {emission.substance} ({component.name})"
                impact = emission.amount * CHARACTERIZATION_FACTORS["Human_Carcinogenic_Toxicity"].get(emission.substance, 0)
                if impact > 0: impact_breakdowns["Human_Carcinogenic_Toxicity"].append({"source": source_name, "type": "Direct Emission", "score": impact})

    # --- E. Assemble Final Response ---
    final_result = {"impact_assessment": []}
    for category, breakdown in impact_breakdowns.items():
        total_score = sum(item['score'] for item in breakdown)
        norm_factor = NORMALIZATION_FACTORS.get(category, 1)
        normalized_score = total_score / norm_factor if norm_factor else 0
        
        final_result["impact_assessment"].append({
            "category": category.replace('_', ' '),
            "total_score": total_score,
            "unit": IMPACT_UNITS[category],
            "breakdown": sorted(breakdown, key=lambda x: x['score'], reverse=True),
            "normalization": {
                "normalized_score": normalized_score,
                "annual_impact_percentage": normalized_score * 100
            }
        })
        
    return final_result


# --- 5. API Endpoint ---
@app.post("/calculate-lci", summary="Calculate Detailed Life Cycle Inventory")
def post_calculate_lci(components: List[ProductComponent]):
    """
    Accepts a list of product components and returns a detailed Life Cycle Inventory (LCI)
    analysis with a breakdown of impacts by source for each category.
    """
    results = calculate_detailed_lci(components)
    return results

@app.get("/", summary="Root", include_in_schema=False)
def root():
    return {"message": "Welcome to the Detailed LCA API. Please send a POST request to /calculate-lci."}

if __name__ == "__main__":
    print("Starting API server...")
    print("Access API documentation at http://127.0.0.1:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)

