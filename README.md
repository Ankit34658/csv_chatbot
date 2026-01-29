# CSV Chatbot

A chatbot application that answers questions based on CSV file data using two different approaches: **PandasAI** and **RAG (Retrieval-Augmented Generation)**.

## 🎯 Overview

This project implements two methods for querying CSV data using natural language:

1. **PandasAI Method** - An LLM generates and executes pandas queries to retrieve answers directly from the CSV file
2. **RAG Method** - Uses vector database search to find relevant information and generate responses

**Performance Note:** The PandasAI approach currently provides better results for this use case.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)


## 💻 Usage

### Running the PandasAI Chatbot (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

This will open a web interface where you can:
- Upload your CSV file
- Ask questions in natural language
- Get answers based on data analysis

### Running the RAG Chatbot

Execute the RAG-based approach:

```bash
python rag.py
```

## 🏗️ Architecture

### PandasAI Approach
```
User Question → LLM → Generate Pandas Query → Execute on CSV → Return Answer
```

### RAG Approach
```
CSV Data → Vector Embeddings → Vector Database → Similarity Search → LLM → Answer
```

## 📊 Features

- Natural language querying of CSV data
- Two different AI approaches for comparison
- Interactive web interface (PandasAI)
- Support for various CSV file formats

## 🔧 Technologies Used

- **PandasAI** - For intelligent data analysis
- **Streamlit** - Web interface framework
- **RAG** - Retrieval-Augmented Generation
- **Vector Database** - For semantic search
- **Python** - Core programming language

```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ankit34658**
- GitHub: [@Ankit34658](https://github.com/Ankit34658)

## 🙏 Acknowledgments

- PandasAI library for intelligent data analysis
- Streamlit for the easy-to-use web framework
- The open-source community

---

**Note:** If you encounter any issues or have questions, please open an issue in the repository.
