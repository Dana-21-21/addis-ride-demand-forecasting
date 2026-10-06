"""Streamlit demo — ops manager picks zone + date, gets hourly forecast."""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

ZONES = [
    "Ayat",
    "Bole",
    "CMC",
    "Gerji",
    "Kazanchis",
    "Lideta",
    "Megenagna",
    "Mexico",
    "Piassa",
    "Saris",
    "Summit",
    "Tor Hailoch",
]

st.set_page_config(page_title="Addis Ride Demand Forecast", layout="wide")
st.title("Addis Ride Demand Forecast")
st.caption("Pick a zone and a date in 1–14 November 2025. Weather and events are looked up for you.")

col1, col2 = st.columns(2)
with col1:
    zone = st.selectbox("Zone", ZONES)
with col2:
    date = st.date_input(
        "Date",
        value=pd.Timestamp("2025-11-01").date(),
        min_value=pd.Timestamp("2025-11-01").date(),
        max_value=pd.Timestamp("2025-11-14").date(),
    )

if st.button("Forecast", type="primary"):
    st.info(
        f"Forecast for **{zone}** on **{date}** will appear here once the model and "
        "`data/processed/` tables are ready."
    )
    st.write("Outputs to show next:")
    st.markdown(
        """
        - 24-hour trips table + curve  
        - Peak hour  
        - Drivers needed (trips ÷ ~1.3)  
        - Expected gross fares  
        - Weather / event context line  
        """
    )
else:
    st.write("Choose a zone and date, then click **Forecast**.")
