import streamlit as st

st.set_page_config(page_title="Test", page_icon="🤖")

st.title("🔧 Diagnostic Test")

# Test 1: Basic Streamlit
st.success("✅ Step 1: Streamlit is working!")

# Test 2: Imports
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    st.success("✅ Step 2: All basic packages imported!")
except Exception as e:
    st.error(f"❌ Step 2 Failed: {e}")

# Test 3: OpenAI import
try:
    from openai import OpenAI
    st.success("✅ Step 3: OpenAI package imported!")
except Exception as e:
    st.error(f"❌ Step 3 Failed: {e}")

# Test 4: Secrets
try:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if api_key and api_key.startswith("sk-"):
        st.success(f"✅ Step 4: API key found! (starts with {api_key[:15]}...)")
    else:
        st.error("❌ Step 4: No valid API key found")
except Exception as e:
    st.error(f"❌ Step 4 Failed: {e}")

# Test 5: Create OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
    st.success("✅ Step 5: OpenAI client created!")
except Exception as e:
    st.error(f"❌ Step 5 Failed: {e}")

st.info("If all 5 steps passed ✅, your setup is correct and we can add the full app back!")
