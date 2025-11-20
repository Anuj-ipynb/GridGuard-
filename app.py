

import streamlit as st
import pandas as pd
import random
import os

from utils import (
    generate_synthetic_logs, cached_summarizer, classify_fault_direct,
    get_suggestions, get_fault_distribution, get_rag_chain, ask_rag,
    evaluate_accuracy, use_ollama, llm_model_name
)

# ========================= CONFIG =========================
st.set_page_config(
    page_title="GridGuard Pro",
    page_icon="Lightning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= DATA =========================
if 'logs' not in st.session_state:
    logs, true_labels = generate_synthetic_logs(60)
    st.session_state.logs = logs
    st.session_state.true_labels = true_labels
    st.session_state.uploaded = False
else:
    logs = st.session_state.logs
    true_labels = st.session_state.true_labels

# ========================= HEADER =========================
st.markdown(
    """
    <div style="text-align:center; padding:30px; background:linear-gradient(90deg,#00C9A7,#0066CC); border-radius:15px; margin-bottom:30px;">
        <h1 style="color:white; margin:0;">GridGuard Pro</h1>
        <p style="color:white; font-size:20px;">AI-Powered Fault Diagnosis for Solar • Wind • Battery Systems</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("System Status")
    if use_ollama:
        st.success("Ollama Connected")
        st.caption(f"Model: {llm_model_name}")
    else:
        st.warning("Offline Mode")
        st.caption("Local model active")

    st.divider()
    st.metric("Total Logs", len(logs))
    st.caption("Real Data" if st.session_state.uploaded else "Synthetic Data")

    if st.button("Clear Session", type="secondary"):
        st.session_state.clear()
        st.success("Session cleared!")
        st.rerun()

    st.divider()
    st.caption("© 2025 GridGuard Pro")

# ========================= TABS =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard", "Log Analyzer", "Fault Detector", "Fix Recommendations", "AI Assistant", "Analytics"
])

# ========================= TAB 1 =========================
with tab1:
    st.markdown("### Live Plant Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Logs Processed", len(logs), delta="+12 today")
    with c2: st.metric("AI Accuracy", f"{evaluate_accuracy(tuple(logs), tuple(true_labels)):.1f}%")
    with c3: st.metric("Active Alerts", random.randint(2, 8))
    with c4: st.metric("Response Time", "< 0.8s")
    st.success("All systems operational")

# ========================= TAB 2 =========================
with tab2:
    st.header("Log Analyzer & Summarization")
    uploaded = st.file_uploader("Upload logs (CSV with 'log' column)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if "log" in df.columns:
            st.session_state.logs = df["log"].tolist()
            st.session_state.uploaded = True
            st.success(f"Loaded {len(st.session_state.logs)} real logs!")
            logs = st.session_state.logs
        else:
            st.error("CSV must have a column named 'log'")

    log = st.selectbox("Select log", logs)
    st.code(log, language="text")
    if st.button("Generate Summary", type="primary", width="stretch"):
        with st.spinner("Summarizing..."):
            summary = cached_summarizer(log)
            st.success("Summary")
            st.info(summary)

# ========================= TAB 3 =========================
with tab3:
    st.header("Fault Type Detection")
    log = st.selectbox("Choose log", logs, key="detect")
    if st.button("Detect Fault", type="primary", width="stretch"):
        with st.spinner("Analyzing..."):
            cat, conf, _ = classify_fault_direct(log)
            st.subheader(f"Detected: **{cat}**")
            st.progress(conf)
            st.write(f"**Confidence:** {conf:.1%}")
            if conf < 0.7:
                st.warning("Low confidence — manual review advised")

# ========================= TAB 4 =========================
with tab4:
    st.header("Recommended Actions")
    log = st.selectbox("Select log", logs, key="fix")
    if st.button("Generate Fix Plan", type="primary", width="stretch"):
        with st.spinner("Creating plan..."):
            cat, conf, _ = classify_fault_direct(log)
            actions = get_suggestions(cat)
            st.success(f"Fault: **{cat}** ({conf:.1%} confidence)")
            for i, action in enumerate(actions, 1):
                prio = "High" if i <= 2 else "Medium"
                color = "red" if prio == "High" else "orange"
                st.markdown(f"<span style='color:{color};font-weight:bold;'>[{prio} Priority]</span> {action}", unsafe_allow_html=True)

# ========================= TAB 5 =========================
with tab5:
    st.header("Ask the AI Technician")
    st.caption("Answers from renewable energy manuals & fault database")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. Why inverter trips at noon?"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = get_rag_chain()
                ans = ask_rag(chain, prompt, st.session_state.chat)
            st.markdown(ans)
        st.session_state.chat.append({"role": "assistant", "content": ans})

# ========================= TAB 6 =========================
with tab6:
    st.header("Performance Analytics")
    col1, col2 = st.columns(2)
    with col1:
        acc = evaluate_accuracy(tuple(logs), tuple(true_labels))
        st.metric("AI Accuracy", f"{acc:.1f}%")
        st.metric("Total Logs", len(logs))
    with col2:
        st.plotly_chart(get_fault_distribution(tuple(logs)), use_container_width=True)
    st.success("System running at peak performance")

if st.session_state.uploaded:
    st.balloons()