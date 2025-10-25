import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import ast
import io
import sys

st.set_page_config(page_title="Data Copilot", page_icon="🤖", layout="wide")

# CSS
st.markdown('''<style>
.main-header {font-size: 2.5rem; text-align: center; color: #1f77b4;}
.chat-user {background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;}
.chat-bot {background: #f5f5f5; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;}
</style>''', unsafe_allow_html=True)

# Init
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None

# Safe client initialization
try:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Please add it in Streamlit Cloud Settings → Secrets")
        st.stop()
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"Error initializing OpenAI: {e}")
    st.stop()

def get_context(df):
    return f"Columns: {list(df.columns)}\nShape: {df.shape}\nSample:\n{df.head(2).to_string()}"

def check_relevant(q, ctx):
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Dataset:{ctx}\n\nQuestion:{q}\n\nRelevant to data? YES/NO"}],
            max_tokens=50
        )
        ans = r.choices[0].message.content
        return ans.startswith("YES")
    except Exception as e:
        st.error(f"Error checking relevance: {e}")
        return True

def analyze(q, df, ctx):
    if not check_relevant(q, ctx):
        return {'msg': "🤔 That question isn't about this dataset. Try asking about the data!", 'code': None, 'fig': None, 'data': None}
    
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Dataset:{ctx}\n\nQ:{q}\n\nGenerate pandas+matplotlib code. Store in result_df, end with fig=plt.gcf()"}],
            max_tokens=800
        )
        
        code = r.choices[0].message.content.strip()
        code = code.replace("```python","").replace("```","").strip()
        
        ns = {'pd':pd, 'np':np, 'plt':plt, 'df':df.copy(), 'result_df':None, 'fig':None}
        
        exec(code, ns)
        return {'msg': "✅ Analysis complete!", 'code': code, 'fig': ns.get('fig'), 'data': ns.get('result_df')}
    except Exception as e:
        return {'msg': f"❌ Error: {e}", 'code': None, 'fig': None, 'data': None}

# UI
st.markdown('<h1 class="main-header">🤖 Data-to-Insight Copilot</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Upload Data")
    f = st.file_uploader("CSV File", type=['csv'])
    
    if f and st.session_state.df is None:
        st.session_state.df = pd.read_csv(f)
        st.success("✅ Loaded!")
    
    if st.session_state.df is not None:
        st.metric("Rows", st.session_state.df.shape[0])
        st.metric("Cols", st.session_state.df.shape[1])
        
        if st.button("🔄 Clear"):
            st.session_state.messages = []
            st.rerun()

if st.session_state.df is None:
    st.info("👈 Upload CSV to start")
else:
    for m in st.session_state.messages:
        if m['role'] == 'user':
            st.markdown(f'<div class="chat-user">**You:** {m["q"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">**🤖:** {m["msg"]}</div>', unsafe_allow_html=True)
            if m.get('fig'):
                st.pyplot(m['fig'])
            if m.get('data') is not None:
                with st.expander("Data"):
                    st.dataframe(m['data'])
            if m.get('code'):
                with st.expander("Code"):
                    st.code(m['code'])
    
    q = st.chat_input("Ask about your data...")
    
    if q:
        st.session_state.messages.append({'role':'user','q':q})
        
        with st.spinner("Thinking..."):
            ctx = get_context(st.session_state.df)
            res = analyze(q, st.session_state.df, ctx)
            st.session_state.messages.append({'role':'bot','msg':res['msg'],'code':res['code'],'fig':res['fig'],'data':res['data']})
        
        st.rerun()
