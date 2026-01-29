import os
import pandas as pd
from dotenv import load_dotenv

# -----------------------------
# LangChain (LATEST IMPORTS)
# -----------------------------
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()

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
    raise ValueError("❌ Missing Azure OpenAI environment variables")

print("✅ Azure environment loaded")

# -----------------------------
# 2. Load CSV files
# -----------------------------
DATA_DIR = "data"

holding_df = pd.read_csv(os.path.join(DATA_DIR, "holdings.csv"), nrows=1023)
trade_df = pd.read_csv(os.path.join(DATA_DIR, "trades.csv"), nrows=700)

print(f"📄 Holding rows loaded: {len(holding_df)}")
print(f"📄 Trade rows loaded: {len(trade_df)}")

# -----------------------------
# 3. Convert rows → text
# -----------------------------
def create_readable_text_from_row(row):
    parts = []
    for col, val in row.items():
        if pd.notna(val):
            parts.append(f"{col}: {val}")
    return ". ".join(parts) + "."

def df_to_documents(df, source):
    docs = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=create_readable_text_from_row(row),
                metadata={"source": source}
            )
        )
    return docs

documents = (
    df_to_documents(holding_df, "holding") +
    df_to_documents(trade_df, "trade")
)

print(f"📚 Total documents created: {len(documents)}")

# -----------------------------
# 4. LOCAL embeddings (NO AZURE REQUIRED)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Local embeddings initialized")

# -----------------------------
# 5. FAISS Vector Store
# -----------------------------
vector_store = FAISS.from_documents(documents, embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 100}
)

print("✅ FAISS vector store ready")

# -----------------------------
# 6. Azure Chat Model (LLM)
# -----------------------------
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    temperature=0
)

print("✅ Azure chat model initialized")

# -----------------------------
# 7. Prompt Template
# -----------------------------
prompt = PromptTemplate.from_template("""
You are a helpful data analyst.

Rules:
- Use ONLY the information in the context
- If the answer is not present, say: "I don't have that information in the data"
- Mention whether the data came from holding or trade if relevant
- Do NOT guess or hallucinate

Context:
{context}

Question:
{question}

Answer:
""")

# -----------------------------
# 8. RAG Chain (LCEL)
# -----------------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\n🚀 RAG system READY")
print("⚠️ Using first 100 rows from each CSV")
print("Type 'exit' to quit\n")

# -----------------------------
# 9. Interactive Q&A
# -----------------------------
while True:
    user_question = input("You: ")

    if user_question.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break

    try:
        answer = rag_chain.invoke(user_question)
        print(f"\nAI: {answer}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}\n")
