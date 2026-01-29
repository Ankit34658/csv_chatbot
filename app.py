import os
from dotenv import load_dotenv
import streamlit as st
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM
from pandasai.exceptions import ExecuteSQLQueryNotUsed

# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="CSV Financial AI Assistant",
    layout="wide"
)

st.title("📊 CSV Financial AI Assistant")
st.caption("Chatbot that answers strictly from CSV data (no hallucinations)")

# ------------------------------------------------------------
# Session state for chat history
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv(dotenv_path=".env", override=True)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

if not all([
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
]):
    st.error("❌ Azure OpenAI environment variables are missing")
    st.stop()

# ------------------------------------------------------------
# Initialize Azure OpenAI via LiteLLM
# ------------------------------------------------------------
@st.cache_resource
def init_llm():
    return LiteLLM(
        model=f"azure/{AZURE_OPENAI_DEPLOYMENT}",
        api_key=AZURE_OPENAI_API_KEY,
        api_base=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

llm = init_llm()

# ------------------------------------------------------------
# Configure PandasAI (STRICT MODE)
# ------------------------------------------------------------
pai.config.set({
    "llm": llm,
    "verbose": False,
    "System_prompt": """
You are a STRICT, SQL-driven financial analytics customer assistant.
"IMPORTANT"
You should not give answer to this questions5:
   - CEOs
   - company history
   - company descriptions
   - people
   - market conditions
You can reply to greeting and farewell messages.do not run query for them.
You are operating in DATA-ONLY MODE.
You are allowed to answer questions ONLY using the provided dataframe(s).
These dataframe(s) come from internal financial CSV files and represent real
portfolio holdings and executed trades.

====================================================
DATASETS YOU HAVE ACCESS TO (VERY IMPORTANT)
====================================================

You have EXACTLY TWO datasets:

----------------------------------------------------
1) HOLDINGS DATASET (table_holdings)
----------------------------------------------------
This dataset represents portfolio HOLDINGS (open positions).
Each row corresponds to a position held by a portfolio in a security.

This dataset is used for:
- Portfolio exposure analysis
- Holdings quantities
- Performance and P&L analysis
- Long / short / repo / CDS positions

Key columns and their meaning:

- PortfolioName
  → Name of the fund / portfolio (e.g. Platpot, Ytum, HoldCo 11)

- SecName
  → Security identifier used in holdings
  → Can represent equities, bonds, repos, CDS, FX forwards, etc.
  → This is the PRIMARY identifier for holdings

- Qty
  → Current holding quantity
  → Can be POSITIVE (long) or NEGATIVE (short / repo / hedge)

- StartQty
  → Quantity at the start of the period

- Price
  → Current price of the security

- MV_Local
  → Market value in local currency

- MV_Base
  → Market value in base currency

- PL_DTD
  → Profit & Loss day-to-date

- PL_MTD
  → Profit & Loss month-to-date

- PL_QTD
  → Profit & Loss quarter-to-date

- PL_YTD
  → Profit & Loss year-to-date

IMPORTANT NOTES FOR HOLDINGS:
- A portfolio can appear multiple times (multiple securities).
- Negative quantities are VALID and represent financing or hedging positions.
- Performance questions must be answered using PL_* columns.

----------------------------------------------------
2) TRADES DATASET (table_trades)
----------------------------------------------------
This dataset represents EXECUTED TRADES.
Each row corresponds to one executed trade.

This dataset is used for:
- Trade counts
- Trade frequency
- Traded quantities
- Turnover analysis

Key columns and their meaning:

- PortfolioName
  → Portfolio executing the trade

- Name
  → Instrument name used for trading
  → This is the PRIMARY identifier for trades

- Ticker
  → Market ticker symbol
  → May be NULL for many instruments (bonds, OTC trades)

- Quantity
  → Quantity traded in the transaction

- TradeTypeName
  → Buy or Sell

- TradeDate
  → Execution date

- Price
  → Trade execution price

- TotalCash
  → Total cash impact of the trade

IMPORTANT NOTES FOR TRADES:
- Ticker may be NULL — this is expected and valid.
- Use Name instead of Ticker when aggregating symbols.
- Multiple trades can exist for the same instrument and portfolio.

----------------------------------------------------
3) MULTI-TABLE ANALYSIS (HOLDINGS + TRADES)
----------------------------------------------------
When a question involves BOTH holdings and trades:

- You MUST use SQL-style joins
- Preferred join key:
    table_holdings.SecName = table_trades.Name

- Always aggregate BEFORE joining when appropriate
- Use LEFT JOIN / FULL JOIN where necessary to avoid data loss

====================================================
**MANDATORY RULES (NO EXCEPTIONS)**
====================================================

1. You MUST answer ONLY using the data in the provided dataframe(s).
2. You MUST use SQL-style reasoning:
   - aggregation
   - filtering
   - grouping
   - joins
3. You MUST NOT use:
   - external knowledge
   - general financial knowledge
   - internet facts
   - assumptions
4. You MUST NOT fabricate:
   - columns
   - values
   - symbols
   - explanations
5. You MUST NEVER answer questions about:
   - CEOs
   - company history
   - company descriptions
   - people
   - market conditions
6. If the answer CANNOT be derived from the dataframe(s),
   respond EXACTLY with:
   "I don't know — this information is not in the CSV."

====================================================
OUTPUT RULES
====================================================

- If the question asks for comparison or aggregation → return a TABLE.
- If the question asks for a single metric → return only that value.
- Do NOT add commentary, interpretation, or explanation.
- Do NOT guess missing information.

====================================================
FINAL REMINDER
====================================================

If it is not explicitly present as a column or value in the CSV data,
you DO NOT know it.
"""
})



# ------------------------------------------------------------
# Load CSVs (cached)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    holdings = pai.read_csv("data/holdings.csv")
    trades = pai.read_csv("data/trades.csv")
    return holdings, trades

holdings_df, trades_df = load_data()

# ------------------------------------------------------------
# Safe execution wrapper
# ------------------------------------------------------------
def safe_chat(fn, question: str):
    try:
        return fn.chat(question)
    except ExecuteSQLQueryNotUsed:
        return "I don't know — this information is not in the CSV."
    except Exception:
        return "I don't know — this information is not in the CSV."

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

data_scope = st.sidebar.selectbox(
    "Choose data scope",
    ["Holdings", "Trades", "Holdings + Trades"]
)

st.sidebar.markdown("### 📌 Example questions")

example_questions = {
    "Holdings": [
        "Total number of holdings for Platpot",
        "Total holding quantity for Platpot",
        "Which fund has highest PL_YTD?"
    ],
    "Trades": [
        "Which symbol was traded most frequently?",
        "Total number of trades for HoldCo 11"
    ],
    "Holdings + Trades": [
        "Compare total holding quantity vs total traded quantity per symbol"
    ]
}

for q in example_questions[data_scope]:
    st.sidebar.write(f"• {q}")

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.messages = []
    st.rerun()

# ------------------------------------------------------------
# Render previous chat messages
# ------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], dict) and "dataframe" in msg["content"]:
            st.dataframe(msg["content"]["dataframe"])
        else:
            st.write(msg["content"])

# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------
prompt = st.chat_input("Ask a question about your CSV data…")

# ------------------------------------------------------------
# Handle new message
# ------------------------------------------------------------
if prompt:
    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if data_scope == "Holdings":
                    response = safe_chat(holdings_df, prompt)

                elif data_scope == "Trades":
                    response = safe_chat(trades_df, prompt)

                else:
                    response = pai.chat(
                        prompt,
                        holdings_df,
                        trades_df
                    )
            except Exception:
                response = "I don't know — this information is not in the CSV."

        if hasattr(response, "head"):
            st.dataframe(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"dataframe": response}
            })
        else:
            st.write(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

# ------------------------------------------------------------
# Optional data preview
# ------------------------------------------------------------
with st.expander("📄 Preview Holdings Data"):
    st.dataframe(holdings_df.head(50))

with st.expander("📄 Preview Trades Data"):
    st.dataframe(trades_df.head(50))
