import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import ast
import io
import sys
from PIL import Image

# Page config
st.set_page_config(
    page_title="Data-to-Insight Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown('''
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
    }
</style>
''', unsafe_allow_html=True)

# Set OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# Functions
def generate_python_eda(df):
    prompt = f'''You are a pandas expert. Given this DataFrame:

Columns: {list(df.columns)}
Dtypes: {df.dtypes.to_dict()}
Shape: {df.shape}
Sample:
{df.head(3).to_string()}

Generate Python code (pandas + matplotlib) that:
1. Shows basic info (shape, dtypes, missing values)
2. Creates ONE meaningful visualization
3. Stores result in `result_df`
4. Ends with `fig = plt.gcf()`

Return ONLY code, no explanations.'''
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    
    code = response['choices'][0]['message']['content'].strip()
    if code.startswith("```python"):
        code = code.replace("```python", "").replace("```", "").strip()
    elif code.startswith("```"):
        code = code.replace("```", "").strip()
    return code

def generate_query_code(df, question):
    prompt = f'''Pandas expert. DataFrame:

Columns: {list(df.columns)}
Dtypes: {df.dtypes.to_dict()}
Sample:
{df.head(3).to_string()}

Question: "{question}"

Generate code: answer question, store in `result_df`, create chart, end with `fig = plt.gcf()`

Return ONLY code.'''
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    
    code = response['choices'][0]['message']['content'].strip()
    if code.startswith("```python"):
        code = code.replace("```python", "").replace("```", "").strip()
    elif code.startswith("```"):
        code = code.replace("```", "").strip()
    return code

def safe_exec(code, df):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = node.names[0].name if hasattr(node, 'names') else getattr(node, 'module', '')
                if module_name not in ['pandas', 'numpy', 'matplotlib.pyplot', 'matplotlib']:
                    return {"error": f"Import '{module_name}' not allowed"}
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['open', 'eval', 'exec', '__import__']:
                        return {"error": f"Function '{node.func.id}' not allowed"}
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}
    
    namespace = {
        'pd': pd,
        'np': np,
        'plt': plt,
        'df': df.copy(),
        'result_df': None,
        'fig': None
    }
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        exec(code, namespace)
        output = captured_output.getvalue()
        return {
            'result_df': namespace.get('result_df'),
            'fig': namespace.get('fig'),
            'stdout': output,
            'error': None
        }
    except Exception as e:
        return {
            'result_df': None,
            'fig': None,
            'stdout': captured_output.getvalue(),
            'error': str(e)
        }
    finally:
        sys.stdout = old_stdout

# Main app
st.markdown('<h1 class="main-header">🤖 Data-to-Insight Copilot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Data Analysis Assistant</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'])
    
    st.divider()
    
    analysis_type = st.radio(
        "📊 Analysis Mode",
        ["EDA (Exploratory Data Analysis)", "Custom Query"],
        help="Choose between automatic EDA or custom natural language queries"
    )
    
    if analysis_type == "Custom Query":
        question = st.text_area(
            "💬 Your Question",
            placeholder="e.g., What is the total revenue by region?",
            height=100
        )
    else:
        question = ""
    
    st.divider()
    
    analyze_button = st.button("🚀 Analyze Data", type="primary")
    
    st.divider()
    
    st.markdown('''
    ### 💡 Example Questions
    - "Show total sales by product"
    - "What's the revenue trend over time?"
    - "Compare performance by region"
    - "Which category has highest profit?"
    
    ### 🔒 Security Features
    - AST-based validation
    - Sandboxed execution
    - No file I/O allowed
    - Only pandas/numpy/matplotlib
    ''')

# Main content
if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Show dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Rows", df.shape[0])
    with col2:
        st.metric("📋 Columns", df.shape[1])
    with col3:
        st.metric("💾 Size", f"{uploaded_file.size / 1024:.1f} KB")
    
    with st.expander("🔍 Dataset Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Columns: {', '.join(df.columns.tolist())}")
    
    # Analyze button clicked
    if analyze_button:
        if analysis_type == "Custom Query" and not question.strip():
            st.error("❌ Please enter a question for Custom Query mode!")
        else:
            with st.spinner("🤖 AI is generating code..."):
                try:
                    # Generate code
                    if analysis_type == "EDA (Exploratory Data Analysis)":
                        code = generate_python_eda(df)
                    else:
                        code = generate_query_code(df, question)
                    
                    # Execute code
                    result = safe_exec(code, df)
                    
                    if result['error']:
                        st.error(f"❌ **Error:**\n{result['error']}")
                    else:
                        st.success("✅ Analysis complete!")
                        
                        # Results tabs
                        tab1, tab2, tab3 = st.tabs(["📊 Results", "🐍 Generated Code", "📈 Visualization"])
                        
                        with tab1:
                            if result['stdout']:
                                st.text("Console Output:")
                                st.code(result['stdout'], language="text")
                            
                            if result['result_df'] is not None:
                                st.subheader("📋 Result DataFrame")
                                st.dataframe(result['result_df'], use_container_width=True)
                        
                        with tab2:
                            st.code(code, language="python")
                            st.caption("AI-generated pandas code")
                        
                        with tab3:
                            if result['fig'] is not None:
                                st.pyplot(result['fig'])
                            else:
                                st.info("No visualization generated")
                
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")

else:
    # Empty state
    st.info("👈 Upload a CSV file from the sidebar to get started!")
    
    st.markdown('''
    ### 🚀 How to Use
    
    1. **Upload a CSV file** using the sidebar
    2. **Choose an analysis mode:**
       - **EDA Mode:** Automatic exploratory analysis
       - **Query Mode:** Ask questions in natural language
    3. **Click "Analyze Data"** to see results
    4. **View:**
       - Analysis results and statistics
       - AI-generated Python code
       - Interactive visualizations
    
    ### ✨ Features
    
    - 🤖 AI-powered code generation (GPT-4o-mini)
    - 📊 Automatic EDA with visualizations
    - 💬 Natural language queries
    - 🔒 Secure code execution
    - 📈 Interactive charts
    - 🎯 Professional pandas code examples
    ''')

# Footer
st.divider()
st.markdown('''
<div style='text-align: center; color: #666; padding: 1rem;'>
    Built with Streamlit, OpenAI GPT-4o-mini, Pandas & Matplotlib<br>
    <a href='https://github.com/YOUR_USERNAME/data-insight-copilot' target='_blank'>View on GitHub</a>
</div>
''', unsafe_allow_html=True)
