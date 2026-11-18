#!/bin/bash

# setup for EPAS Agent Challenge
# Ubuntu 22.04 - Python Environment

echo "=== EPAS Agent Setup - UiPath Challenge ==="
echo ""

# 1. Verify Python
echo "1. Verify Python's version..."
python3 --version
pip3 --version

# 2. Update pip
echo ""
echo "2. Updating pip..."
pip3 install --upgrade pip

# 3. Set up core dependencies
echo ""
echo "3. Set up core dependencies..."
pip3 install uipath
pip3 install langchain
pip3 install langchain-community
pip3 install langchain-openai

# 4. Set up RAG e PDF
echo ""
echo "4. Set up RAG and processing PDF..."
pip3 install pypdf
pip3 install faiss-cpu  # Vector store per RAG
pip3 install sentence-transformers  # Per embeddings
pip3 install tiktoken  # Token counter per OpenAI

# 5. Set up utility
echo ""
echo "5. Set up utility..."
pip3 install python-dotenv  # Env variables
pip3 install requests
pip3 install pydantic

# 6. Set up optional
echo ""
echo "6. Set up optional..."
pip3 install chromadb  # Alternative vector store
pip3 install openai  # OpenAI Client 

# 7. Verify installation
echo ""
echo "7. Verify installation..."
python3 -c "import uipath; print(f'✓ UiPath SDK: {uipath.__version__}')" 2>/dev/null || echo "✗ UiPath SDK not installed"
python3 -c "import langchain; print('✓ LangChain installed')" 2>/dev/null || echo "✗ LangChain not installed"
python3 -c "import faiss; print('✓ FAISS installed')" 2>/dev/null || echo "✗ FAISS not installed"

echo ""
echo "=== Setup completed! ==="
echo ""
echo "Next steps:"
echo "1. Create a .env file with API keys"
echo "2. Put EASA PDF into data/"
echo "3. Execute script for knowledge base"
