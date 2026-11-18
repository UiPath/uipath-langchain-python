# 🚀 Getting Started with EPAS Agent

## Step-by-Step Setup Guide for Ubuntu 22.04

This guide will walk you through setting up and running the EPAS Agent from scratch.

---

## Prerequisites Check

Before starting, verify you have:

```bash
# Check Python version (need 3.10+)
python3 --version

# Check pip
pip3 --version

# Check available RAM (need 8GB+)
free -h

# Check disk space (need ~5GB)
df -h
```

---

## Step 1: Download or Clone Project

```bash
# Create project directory
mkdir -p ~/projects
cd ~/projects

# If you have the code, extract it here
# Otherwise, create the structure:
mkdir epas-agent-challenge
cd epas-agent-challenge
```

---

## Step 2: Create Project Structure

```bash
# Create all necessary directories
mkdir -p data/raw data/processed data/vectorstore
mkdir -p src/config src/data_processing src/rag src/agent src/api
mkdir -p scripts tests docs logs

# Make scripts executable
chmod +x scripts/*.py
```

---

## Step 3: Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify activation (should show venv path)
which python

# Upgrade pip
pip install --upgrade pip
```

---

## Step 4: Install Dependencies

```bash
# Run the setup script
bash setup_environment.sh

# Or install manually:
pip install -r requirements.txt

# This will install:
# - uipath
# - langchain and related packages
# - sentence-transformers
# - faiss-cpu
# - pypdf
# - and more...

# Verify installations
python -c "import langchain; print('✓ LangChain')"
python -c "import faiss; print('✓ FAISS')"
python -c "import sentence_transformers; print('✓ Sentence Transformers')"
```

---

## Step 5: Configure Environment

```bash
# Copy environment template
cp .env.template .env

# Edit with your favorite editor
nano .env
# OR
vim .env
# OR
code .env  # if you have VS Code

# You MUST set:
# OPENAI_API_KEY=sk-your-actual-key-here
```

### Getting OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy it and paste in .env file
4. Save the file

---

## Step 6: Obtain EPAS PDF Files

You need the 3 EPAS volumes from EASA:

### Option A: Download from EASA

1. Go to EASA website: https://www.easa.europa.eu/
2. Navigate to EPAS 2024-2028 section
3. Download the three volumes:
   - Easy Access Rules for Standardisation
   - EPAS Actions Volume
   - EPAS Safety Risk Portfolio

### Option B: If You Already Have Them

```bash
# Place them in data/raw/ with these exact names:
# data/raw/volume_1_regulations.pdf
# data/raw/volume_2_actions.pdf
# data/raw/volume_3_safety_risk.pdf

# Or rename your files:
cp /path/to/your/easa_volume1.pdf data/raw/volume_1_regulations.pdf
cp /path/to/your/easa_volume2.pdf data/raw/volume_2_actions.pdf
cp /path/to/your/easa_volume3.pdf data/raw/volume_3_safety_risk.pdf
```

### Verify PDFs

```bash
# Check if files exist
ls -lh data/raw/

# Should show:
# volume_1_regulations.pdf
# volume_2_actions.pdf
# volume_3_safety_risk.pdf
```

---

## Step 7: Test Configuration

```bash
# Test that configuration is correct
python -c "from src.config.settings import print_config_summary; print_config_summary()"

# Should output:
# ✓ Volume I: <path>
# ✓ Volume II: <path>
# ✓ Volume III: <path>
```

---

## Step 8: Setup Knowledge Base

This is the most important step - it processes the PDFs and creates the vector store.

```bash
# Run setup script (takes ~10 minutes)
python scripts/setup_knowledge_base.py
```

**What happens**:
1. ✅ Loads PDFs and extracts text
2. ✅ Splits into intelligent chunks
3. ✅ Generates embeddings (this takes time)
4. ✅ Creates FAISS vector store
5. ✅ Saves everything to disk

**Output should show**:
```
Loading PDF: data/raw/volume_1_regulations.pdf
Extracted 500 pages from Volume I
...
Created 4000 chunks from 3 volumes
...
Embedding 4000 chunks...
100%|████████████████| 125/125 [05:30<00:00]
...
✅ KNOWLEDGE BASE SETUP COMPLETE!
```

**If it fails**:
- Check PDF files are in correct location
- Check you have enough disk space (~2GB needed)
- Check RAM usage (close other applications)
- Check logs in `logs/setup_kb.log`

---

## Step 9: Test the System

### Test 1: Test RAG System

```bash
python scripts/test_rag.py

# Choose option 3 (Interactive Mode)
```

Try these test questions:
```
What are the strategic priorities in EPAS?
vol:I What are maintenance requirements?
Find cross-references for CAT.GEN.MPA.210
```

### Test 2: Run the Agent

```bash
python scripts/run_agent.py

# Choose option 1 (Interactive Mode)
```

Now you can ask questions!

---

## Step 10: Verify Everything Works

### Quick Verification Checklist

```bash
# 1. Check vector store exists
ls -lh data/vectorstore/
# Should show: faiss_index.bin, chunks.pkl, metadata.pkl

# 2. Check processed data
ls -lh data/processed/
# Should show: volume_*_embedded.pkl files

# 3. Test a simple query
python -c "
from src.rag.vectorstore import EPASVectorStore
store = EPASVectorStore.load('data/vectorstore')
print(f'✓ Vector store loaded: {store.get_statistics()}')
"
```

---

## Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'uipath'"

**Solution**:
```bash
pip install uipath-sdk
# If not available, the system will work in mock mode
```

### Issue 2: "FileNotFoundError: PDF not found"

**Solution**:
```bash
# Check PDF files are in correct location
ls data/raw/

# If missing, place PDFs there with correct names
```

### Issue 3: "openai.error.AuthenticationError"

**Solution**:
```bash
# Check API key in .env file
cat .env | grep OPENAI_API_KEY

# Make sure it starts with sk- and has no quotes
```

### Issue 4: Out of Memory

**Solution**:
```bash
# Close other applications
# Or reduce batch size in config:
# Edit src/config/settings.py
# Change: chunk_size = 500 to chunk_size = 300
```

### Issue 5: Slow Embedding Generation

**Solution**:
```bash
# This is normal! Embedding 4000 chunks takes 5-10 minutes
# Wait for it to complete
# You can monitor progress in logs/setup_kb.log
```

---

## Next Steps

Once everything is working:

1. **Read the Documentation**
   ```bash
   cat docs/SUBMISSION.md
   cat docs/ARCHITECTURE.md
   ```

2. **Try Different Queries**
   - Regulatory questions
   - Safety action queries
   - Risk assessment questions
   - Cross-reference searches

3. **Customize for Your Needs**
   - Adjust prompts in `src/agent/epas_agent.py`
   - Modify retrieval parameters in `src/config/settings.py`
   - Add custom tools

4. **Integration**
   - Connect to your existing EPASApp
   - Create REST API endpoint
   - Build web interface

---

## Useful Commands

```bash
# Activate environment
source venv/bin/activate

# Deactivate environment
deactivate

# View logs
tail -f logs/epas_agent.log

# Clean and rebuild
rm -rf data/vectorstore data/processed
python scripts/setup_knowledge_base.py

# Update dependencies
pip install -r requirements.txt --upgrade

# Check disk usage
du -sh data/
```

---

## Getting Help

1. **Check logs**: `logs/setup_kb.log`, `logs/epas_agent.log`
2. **Read docs**: `docs/` folder
3. **Test components**: Use individual test scripts
4. **Debug mode**: Set `DEBUG_MODE=true` in .env

---

## Summary

✅ Environment setup  
✅ Dependencies installed  
✅ Configuration complete  
✅ PDFs processed  
✅ Vector store created  
✅ System tested  
✅ Ready to use!

**You're all set! Start asking questions about aviation safety!** 🛩️

---

## Quick Reference Card

```bash
# Daily usage commands:

# 1. Activate environment
source venv/bin/activate

# 2. Run agent
python scripts/run_agent.py

# 3. Test RAG
python scripts/test_rag.py

# 4. View logs
tail -f logs/epas_agent.log

# 5. Deactivate when done
deactivate
```

---

**Need more help? Check docs/ARCHITECTURE.md for technical details!**
