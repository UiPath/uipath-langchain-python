#!/usr/bin/env python3
"""
Setup Knowledge Base Script
Processes EPAS PDFs and creates vector store
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config.settings import settings, print_config_summary
from src.data_processing.pdf_loader import load_all_volumes
from src.data_processing.chunker import chunk_all_volumes
from src.data_processing.embedder import embed_all_volumes, save_embeddings
from src.rag.vectorstore import create_vectorstore_from_embeddings


def setup_logging():
    """Setup logging configuration"""
    log_file = settings.log_file.parent / "setup_kb.log"
    logger.add(
        log_file,
        rotation="10 MB",
        retention="1 week",
        level=settings.log_level
    )
    logger.info("="*80)
    logger.info("EPAS Knowledge Base Setup")
    logger.info("="*80)


def validate_environment():
    """Validate that all requirements are met"""
    logger.info("\n📋 Validating Environment...")
    
    # Check PDFs exist
    pdf_status = settings.validate_pdfs()
    all_exist = all(status['exists'] for status in pdf_status.values())
    
    if not all_exist:
        logger.error("❌ Not all PDF files found!")
        logger.info("\nExpected PDFs:")
        for volume, status in pdf_status.items():
            symbol = "✓" if status['exists'] else "✗"
            logger.info(f"  {symbol} Volume {volume}: {status['path']}")
        
        logger.info("\n📝 Instructions:")
        logger.info(f"1. Download the 3 EPAS volumes from EASA")
        logger.info(f"2. Place them in: {settings.raw_pdf_dir}")
        logger.info(f"3. Rename them as:")
        for volume, info in settings.volumes.items():
            logger.info(f"   - Volume {volume}: {info['filename']}")
        
        return False
    
    logger.info("✓ All PDF files found")
    
    # Check API key
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        logger.error("❌ OpenAI API key not configured!")
        logger.info("Set OPENAI_API_KEY in .env file")
        return False
    
    logger.info("✓ OpenAI API key configured")
    
    return True


def main():
    """Main setup process"""
    setup_logging()
    
    # Print configuration
    print_config_summary()
    
    # Validate environment
    if not validate_environment():
        logger.error("\n❌ Environment validation failed. Please fix the issues above.")
        sys.exit(1)
    
    logger.info("\n✓ Environment validation passed\n")
    
    try:
        # Step 1: Load PDFs
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Loading PDF Documents")
        logger.info("="*80)
        
        all_volumes = load_all_volumes(settings)
        
        total_sections = sum(len(sections) for sections in all_volumes.values())
        logger.info(f"\n✓ Loaded {total_sections} sections from {len(all_volumes)} volumes")
        
        # Step 2: Chunk documents
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Chunking Documents")
        logger.info("="*80)
        
        chunked_volumes = chunk_all_volumes(
            all_volumes,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        total_chunks = sum(len(chunks) for chunks in chunked_volumes.values())
        logger.info(f"\n✓ Created {total_chunks} chunks")
        
        # Step 3: Generate embeddings
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Generating Embeddings")
        logger.info("="*80)
        logger.info("⚠️  This may take several minutes...")
        
        embedded_volumes = embed_all_volumes(
            chunked_volumes,
            model_name=settings.embedding_model,
            batch_size=32
        )
        
        logger.info(f"\n✓ Generated embeddings for {total_chunks} chunks")
        
        # Step 4: Save embeddings (optional, for backup)
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Saving Embeddings")
        logger.info("="*80)
        
        save_embeddings(embedded_volumes, str(settings.processed_dir))
        logger.info(f"✓ Embeddings saved to {settings.processed_dir}")
        
        # Step 5: Create vector store
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Creating Vector Store")
        logger.info("="*80)
        
        vectorstore = create_vectorstore_from_embeddings(
            embedded_volumes,
            embedding_dim=settings.embedding_dimension
        )
        
        # Step 6: Save vector store
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Saving Vector Store")
        logger.info("="*80)
        
        vectorstore.save(str(settings.vectorstore_dir))
        logger.info(f"✓ Vector store saved to {settings.vectorstore_dir}")
        
        # Print statistics
        stats = vectorstore.get_statistics()
        logger.info("\n" + "="*80)
        logger.info("📊 FINAL STATISTICS")
        logger.info("="*80)
        logger.info(f"Total chunks: {stats['total_chunks']:,}")
        logger.info(f"Embedding dimension: {stats['embedding_dim']}")
        logger.info(f"Vector store size: {stats['index_size']:,} vectors")
        logger.info("\nChunks per volume:")
        for volume, count in stats['volumes'].items():
            logger.info(f"  - Volume {volume}: {count:,} chunks")
        
        logger.info("\n" + "="*80)
        logger.info("✅ KNOWLEDGE BASE SETUP COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nVector store location: {settings.vectorstore_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Test the RAG system: python scripts/test_rag.py")
        logger.info("2. Run the agent: python scripts/run_agent.py")
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Setup failed: {str(e)}")
        logger.exception("Full error traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
