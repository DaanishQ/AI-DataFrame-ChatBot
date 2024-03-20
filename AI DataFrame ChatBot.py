from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import os

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


st.set_page_config(page_title="AI DataFrame ChatBot", page_icon="🌏", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

st.title("AI DataFrame ChatBot 🌏")

uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are Support",
    on_change=clear_submit,
)

with st.sidebar:
    st.sidebar.image("image/57045756.png", use_column_width=True)
    st.sidebar.markdown(
    f"""
    Powered by the GitHub: [DaanishQ](https://github.com/DaanishQ)
    """, unsafe_allow_html=True)
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not uploaded_file:
    st.warning(
        "Please upload the file, otherwise you don't able to talk with AI DataFrame ChatBot!"
    )
    st.stop()
    
if not openai_api_key:
    st.warning("Please add your OpenAI API key to continue the GPT-3.5 Turbo conversation.")
    st.stop()

if uploaded_file:
    df = load_data(uploaded_file)


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
