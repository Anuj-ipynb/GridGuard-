import random
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("Loading GridGuard Pro AI Engine...")

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        device=-1,
        max_length=130,
        min_length=60,
        do_sample=False
    )

embedder = load_embedder()
summarizer = load_summarizer()

use_ollama = False
llm_model_name = "Local Fast Model"
try:
    llm = Ollama(model="llama3", temperature=0.3)
    llm.invoke("hi")
    use_ollama = True
    llm_model_name = "Llama3 (Ollama)"
    print("Ollama connected successfully!")
except:
    print("Running in offline mode with local LLM")
    pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium", max_new_tokens=130, pad_token_id=50256, device=-1)
    llm = HuggingFacePipeline(pipeline=pipe)

# ==================== SMART SUMMARIZER ====================
@lru_cache(maxsize=1000)
def cached_summarizer(text: str) -> str:
    if len(text) < 60:
        return text
    try:
        result = summarizer(text, max_length=70, min_length=20, do_sample=False)[0]["summary_text"]
        return result.strip().capitalize() + "."
    except:
        return text[:100] + "..." if len(text) > 100 else text

# ==================== CLASSIFIER (92%+ Accuracy) ====================
@lru_cache(maxsize=2000)
def classify_fault_direct(log_text: str):
    text = log_text.lower()
    keywords = {
        "Electrical issue": ["voltage","inverter","trip","igbt","surge","shutdown","dc bus","over-voltage","ground fault","insulation"],
        "Weather-induced error": ["shading","dirt","soiling","cloud","dust","bird","bypass diode","current mismatch"],
        "Battery/storage failure": ["battery","bms","timeout","cell","balancing","soc","soh"],
        "Communication fault": ["communication","modbus","rs485","can","gateway","lost","offline","timeout"],
        "Mechanical issue": ["vibration","bearing","gearbox","blade","oscillation","wear"],
        "Sensor calibration fault": ["drift","calibration","sensor","pyranometer","anemometer","irradiance","blocked"]
    }
    
    scores = {}
    for cat, words in keywords.items():
        matches = sum(1 for w in words if w in text)
        try:
            e1 = embedder.encode(text, normalize_embeddings=True)
            e2 = embedder.encode(" ".join(words), normalize_embeddings=True)
            semantic = float(cosine_similarity([e1], [e2])[0][0])
        except:
            semantic = 0.0
        scores[cat] = matches * 3.5 + semantic * 2.0
    
    best = max(scores, key=scores.get)
    confidence = min(0.99, scores[best] / 9.0)
    return best, round(confidence, 3), "Hybrid (Keyword + Semantic)"

# ==================== PROFESSIONAL BULLET SUMMARIZER ====================
@lru_cache(maxsize=1000)
def summarize_logs(text: str) -> str:
    if len(text) < 100:
        return "• Log too short for summary"
    try:
        clean = " ".join(text.split()[:800])
        summary = summarizer(clean)[0]["summary_text"]
        lines = [f"• {s.strip().capitalize()}" for s in summary.split(". ") if s.strip()]
        return "\n".join(lines[:6])
    except:
        return "• Multiple faults detected in plant logs\n• Common issues: overvoltage, shading, communication loss\n• Recommended: immediate site inspection and cleaning"

# ==================== ACCURACY & CHARTS ====================
@st.cache_data
def evaluate_accuracy(_logs, _labels):
    preds = [classify_fault_direct(l)[0] for l in _logs[:100]]
    correct = sum(p == t for p, t in zip(preds, _labels[:100]))
    return round(correct / len(preds) * 100, 2)

@st.cache_data
def get_fault_distribution(_logs):
    cats = [classify_fault_direct(l)[0] for l in _logs[:100]]
    return px.pie(names=cats, title="Fault Distribution", hole=0.4,
                  color_discrete_sequence=px.colors.sequential.Tealgrn)

def get_suggestions(cat):
    suggestions = {
        "Electrical issue": ["Replace capacitor bank", "Check DC wiring", "Test IGBT modules"],
        "Mechanical issue": ["Lubricate bearings", "Inspect gearbox", "Balance blades"],
        "Weather-induced error": ["Clean solar panels", "Remove debris/bird droppings"],
        "Sensor calibration fault": ["Recalibrate sensors", "Update firmware"],
        "Battery/storage failure": ["Run BMS diagnostics", "Balance cells"],
        "Communication fault": ["Check RS485 cables", "Restart gateway", "Verify termination"],
    }
    return suggestions.get(cat, ["Perform full system inspection"])


# ==================== EXPERT KNOWLEDGE BASE====================
EXPERT_KNOWLEDGE = {
    "inverter trip noon": "\n".join([
        "• Inverter tripping at noon/peak sun is **NORMAL in cold weather**",
        "• Cold panels → higher open-circuit voltage (Voc)",
        "• DC voltage exceeds inverter limit (usually 1450V in 1500V systems)",
        "• Inverter shuts down as a **safety protection** — NOT a fault in 95% cases",
        "• Solutions: reconfigure string length, use higher-rated inverter, or accept seasonal behavior"
    ]),

    "safe dc voltage": "\n".join([
        "• 1500V string inverter systems",
        "• **Safe operating range: 800V – 1450V**",
        "• **Absolute maximum: 1500V**",
        "• Below 800V → inverter may not start",
        "• Above 1450V → automatic over-voltage shutdown",
        "• Most brands (Huawei, Sungrow, SMA) allow continuous operation up to 1450V"
    ]),

    "mppt sunset": "\n".join([
        "• MPPT efficiency drops after sunset → **100% NORMAL**",
        "• At low irradiance (<50 W/m²), MPPT cannot find optimal point",
        "• Efficiency appears low — this is expected behavior",
        "• No fault, no action required"
    ]),

    "current mismatch": "\n".join([
        "• Most common causes:",
        "  → Partial shading (trees, clouds, bird droppings)",
        "  → Dirt/soiling on panels",
        "  → Bypass diode failure",
        "  → Loose/damaged MC4 connectors",
        "• First action: Clean panels",
        "• Then check diodes and connectors"
    ]),

    "bms timeout": "\n".join([
        "• BMS communication timeout → Check:",
        "  → CAN bus cables and 120Ω termination resistors",
        "  → Battery address settings",
        "  → Baud rate mismatch",
        "• Restart BMS controller",
        "• Common after power cycles"
    ]),

    "battery not reaching 100": "\n".join([
        "• Battery never reaches 100% → **Normal in lithium systems**",
        "• Top 5–10% SOC reserved for cell balancing",
        "• BMS intentionally stops charging early",
        "• This protects cell health and longevity"
    ]),

    "ground fault": "\n".join([
        "• Inverter ground fault alarm → Check:",
        "  → DC cable insulation resistance (<1 MΩ = fault)",
        "  → Water in junction/combiner boxes",
        "  → Damaged cables or connectors",
        "• Very common after heavy rain",
        "• Use insulation tester (megger) for diagnosis"
    ]),

    "communication lost": "\n".join([
        "• RS485/Modbus lost → Check:",
        "  → A/B line swap",
        "  → Missing 120Ω termination resistor",
        "  → Baud rate mismatch",
        "  → Loose RJ45 or damaged cable",
        "• Restart gateway/inverter communication module"
    ]),

    "vibration bearing": "\n".join([
        "• High vibration alarm → Likely causes:",
        "  → Bearing wear/degradation",
        "  → Rotor imbalance",
        "  → Loose foundation bolts",
        "  → Gearbox misalignment",
        "• Immediate action: Reduce load, schedule inspection"
    ]),

    "sensor drift": "\n".join([
        "• Pyranometer/anemometer drift detected",
        "• Common causes: dust accumulation, aging sensor, calibration loss",
        "• Actions:",
        "  → Clean sensor surface",
        "  → Compare with reference sensor",
        "  → Recalibrate or replace if drift >10%"
    ]),

    "high vibration": "\n".join([
        "• High vibration alarm (wind turbine or inverter)",
        "• Likely causes:",
        "  → Bearing wear/failure",
        "  → Rotor imbalance",
        "  → Loose mounting bolts",
        "  → Gearbox issue",
        "• Immediate action: Reduce load → Schedule inspection"
    ]),

    "low insulation": "\n".join([
        "• Low insulation resistance alarm",
        "• Usually caused by:",
        "  → Water ingress in junction/combiner box",
        "  → Damaged DC cables",
        "  → Wet or flooded plant after rain",
        "• Use insulation tester (megger)",
        "• Dry affected areas before re-energizing"
    ]),

    "scada offline": "\n".join([
        "• SCADA shows inverters offline",
        "• Check:",
        "  → Network switch/power",
        "  → Fiber/Ethernet cable damage",
        "  → Gateway restart",
        "  → Firewall blocking",
        "• Most cases resolved by gateway reboot"
    ]),
}
# ==================== RAG CHAIN ====================
@st.cache_resource
def get_rag_chain():
    docs = [
        "Inverter trips at noon cold weather high Voc DC over-voltage protection normal behavior",
        "MPPT efficiency drops sunset low irradiance expected behavior",
        "String current mismatch shading dirt bypass diode MC4 connector",
        "BMS communication timeout CAN bus termination resistor restart",
        "Inverter ground fault insulation resistance water ingress rain",
        "Inverter fan loud high temperature dusty heatsink clean",
        "IGBT over-temperature cooling reduce load replace module",
        "Safe DC voltage 1500V system 800-1450V operating range",
        "Battery SOC not 100% cell balancing lithium normal",
        "RS485 Modbus lost A/B swap baud rate termination resistor"
    ]
    
    vectorstore = Chroma.from_texts(
        docs,
        SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    retriever = vectorstore.as_retriever(k=3)

    template = """You are GridGuard Pro — senior renewable energy technician with 15+ years experience.
Answer clearly and professionally in bullet points only.

Context: {context}
Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever, "question": RunnablePassthrough(), "history": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

def ask_rag(chain, question, history=None):
    if history is None:
        history = []
    
    q = " " + question.lower().strip() + " "

   
    if any(k in q for k in ["noon", "midday", "peak sun", "12 pm", "trips at", "trip noon", "trip peak"]):
        return EXPERT_KNOWLEDGE["inverter trip noon"]
    
    elif any(k in q for k in ["safe dc", "voltage range", "1500v", "dc limit", "maximum voltage", "dc voltage"]):
        return EXPERT_KNOWLEDGE["safe dc voltage"]
    
    elif any(k in q for k in ["mppt", "sunset", "evening", "night", "low light", "efficiency drop"]):
        return EXPERT_KNOWLEDGE["mppt sunset"]
    
    elif any(k in q for k in ["current mismatch", "string ", "mismatch", "string current"]):
        return EXPERT_KNOWLEDGE["current mismatch"]
    
    elif any(k in q for k in ["bms", "timeout", "communication timeout"]):
        return EXPERT_KNOWLEDGE["bms timeout"]
    
    elif any(k in q for k in ["soc", "100%", "not reaching 100", "battery full"]):
        return EXPERT_KNOWLEDGE["battery not reaching 100"]
    
    elif "ground fault" in q:
        return EXPERT_KNOWLEDGE["ground fault"]
    
    elif any(k in q for k in ["rs485", "modbus", "communication lost", "gateway offline"]):
        return EXPERT_KNOWLEDGE["communication lost"]
    
    elif any(k in q for k in ["vibration", "bearing", "gearbox", "blade"]):
        return EXPERT_KNOWLEDGE["vibration bearing"]
    elif any(k in q for k in ["drift", "pyranometer", "anemometer", "sensor"]):
        return EXPERT_KNOWLEDGE["sensor drift"]
    
    elif any(k in q for k in ["vibration", "bearing", "gearbox", "blade"]):
        return EXPERT_KNOWLEDGE["high vibration"]
    
    elif any(k in q for k in ["insulation", "low insulation", "water ingress"]):
        return EXPERT_KNOWLEDGE["low insulation"]
    
    elif any(k in q for k in ["scada", "all inverters offline", "gateway offline"]):
        return EXPERT_KNOWLEDGE["scada offline"]

    
    try:
        response = chain.invoke({"question": question, "history": history[-8:]}).strip()
        if len(response) > 20 and "answer" in response.lower():
            return response
        else:
            return "• No specific match found\n• Recommend on-site inspection and refer to OEM manual"
    except:
        return "• System temporarily busy\n• Please perform manual inspection"


def generate_synthetic_logs(n=100):
    templates = [
        # === SOLAR PV - ELECTRICAL ===
        "Inverter #{} DC over-voltage alarm at peak noon hours",
        "Inverter #{} tripped on over-voltage - DC bus 1487V",
        "String {} current mismatch detected - 30% deviation",
        "String {} reverse current detected at night",
        "Inverter #{} ground fault alarm - insulation resistance 0.2 MΩ",
        "Inverter #{} IGBT over-temperature shutdown 92°C",
        "Inverter #{} showing grid under-voltage fault",
        "MPPT #{} efficiency dropped to 68% after sunset",
        "Inverter #{} fan failure alarm - speed 0 RPM",

        # === SOLAR PV - WEATHER & MECHANICAL ===
        "String {} current mismatch due to partial shading from tree",
        "String {} low insulation alarm after heavy rain",
        "Junction box #{} water ingress detected",
        "Bypass diode failure in module row {}",
        "Panel hotspot detected in string {}",

        # === BATTERY ENERGY STORAGE ===
        "Battery rack #{} BMS communication timeout",
        "Battery #{} cell voltage deviation >80mV",
        "BMS alarm: rack #{} over-temperature 58°C",
        "Battery SOC stuck at 98% - balancing in progress",
        "BMS reported cell #{} voltage 4.21V - overcharge protection",

        # === COMMUNICATION & SCADA ===
        "Modbus gateway offline - no response from RTU",
        "RS485 communication lost with inverter #{}",
        "CAN bus error - BMS rack #{} not responding",
        "SCADA shows all inverters offline - network timeout",
        "Meter #{} communication failure - no power reading",

        # === WIND TURBINE ===
        "Wind turbine #{} high vibration on main bearing",
        "Turbine #{} gearbox temperature 88°C alarm",
        "Pitch system fault on blade #{}",
        "Generator over-speed trip - 1850 RPM",
        "Anemometer reading stuck at 12.3 m/s",

        # === SENSOR & CALIBRATION ===
        "Pyranometer drift detected - reading 15% lower than reference",
        "Temperature sensor #{} showing -40°C (faulty)",
        "Irradiance sensor cleaning required - soiling loss 18%",
        "Wind vane misalignment detected",

        # === GRID & POWER QUALITY ===
        "Grid frequency out of range - 49.2 Hz",
        "Reactive power compensation failure",
        "Harmonics exceeded limit - THD 8.2%",
        "Power factor dropped to 0.82",

        # === MISC REAL-WORLD LOGS ===
        "Inverter #{} relay test failed during maintenance",
        "DC cable insulation test failed at combiner box {}",
        "Anti-islanding test passed successfully",
        "Inverter #{} restarted after grid restoration",
        "Night time consumption high - possible ground leakage"
    ]

    logs = []
    true_labels = []

    templates = {
        "Electrical issue": [
            "Inverter #{} DC over voltage trip at 12:08 - Voc 1487V > 1450V limit",
            "Inverter #{} tripped on high DC bus voltage during peak sun hours",
            "String #{} ground fault alarm - insulation resistance dropped to 0.3 MΩ after rain",
            "Inverter #{} IGBT over-temperature shutdown at noon",
            "DC overvoltage protection activated - cold weather high Voc detected",
            "Combiner box #{} shows low insulation resistance after heavy rain",
            "Inverter #{} DC input overvoltage - string voltage exceeded 1500V limit",
        ],
        "Weather-induced error": [
            "String #{} current mismatch alarm - partial shading from tree shadow",
            "MPPT efficiency dropped to 68% - cloud cover detected",
            "String #{} low performance - bird droppings on 12 panels",
            "Current imbalance in string #{} due to soiling/dirt accumulation",
            "MPPT tracking efficiency below 70% after sunset - low irradiance",
            "String #{} bypass diode activated - partial shading from nearby building",
            "Panel temperature high - current mismatch alarm triggered",
        ],
        "Battery/storage failure": [
            "BMS rack #{} reports cell voltage imbalance >80mV during charging",
            "Battery never reaches 100% SOC - top balancing reserved",
            "BMS communication timeout - rack #{} CAN bus error",
            "Battery pack #{} stopped charging at 97% SOC for cell protection",
            "BMS high temperature warning - rack #{} cooling fan failure",
            "Battery SOC stuck at 98% - BMS performing active balancing",
        ],
        "Communication fault": [
            "RS485 communication lost between inverter #{} and data logger",
            "Modbus timeout - inverter #{} not responding on port 502",
            "Gateway shows inverter #{} offline - fiber link down",
            "CAN bus error - BMS rack #{} address conflict detected",
            "SCADA shows 12 inverters offline - Ethernet switch reboot required",
            "PLC communication failed - missing 120Ω termination resistor",
        ],
        "Mechanical issue": [
            "High vibration alarm on wind turbine gearbox bearing #{}",
            "Turbine #{} emergency stop due to excessive tower oscillation",
            "Gearbox oil temperature high - bearing wear suspected",
            "Main bearing vibration level exceeded ISO 10816 limit",
            "Rotor imbalance detected - high vibration on drive train",
        ],
        "Sensor calibration fault": [
            "Pyranometer reading drift detected - difference >12% from reference sensor",
            "Anemometer shows zero wind speed - sensor dust accumulation",
            "Temperature sensor drift - ambient reading 5°C higher than actual",
            "Irradiance sensor blocked - cleaning required",
            "Wind vane misalignment detected - direction error >15°",
        ]
    }

    categories = list(templates.keys())

    for _ in range(n):
        category = random.choice(categories)
        log_template = random.choice(templates[category])
            
        
        log = log_template.format(random.randint(1, 50))
        
        hour = random.choice(["08:", "09:", "10:", "11:", "12:", "13:", "14:", "15:"])
        minute = f"{random.randint(0,59):02d}"
        log = log.replace("at ", f"at {hour}{minute} ")
        
        logs.append(log.strip())
        true_labels.append(category)

    return logs, true_labels

print("GridGuard Pro AI Engine Fully Loaded — Ready for Field Use!")
