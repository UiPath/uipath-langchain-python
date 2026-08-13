"""
Prompt Loader for ReAct Claims Processing Agent
Manages loading and caching of system prompts from external files.

File Naming Convention:
    Prompt files should be named: {prompt_name}.txt
    Example: react_system.txt, validation_prompt.txt

Supported Prompts:
    - react_system: Main ReAct agent system prompt
    - react_system_prompt: Alias for react_system
    
Usage:
    >>> from src.agents.prompts.prompt_loader import PromptLoader
    >>> prompt = PromptLoader.load_prompt("react_system")
    >>> print(prompt[:50])
    
Thread Safety:
    This class uses class-level caching with threading locks for thread-safe operation.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, List

from .fallback_prompts import REACT_SYSTEM_FALLBACK, DEFAULT_FALLBACK

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Loads and manages system prompts for the ReAct agent.
    
    Features:
    - Loads prompts from external .txt files
    - Caches loaded prompts to avoid repeated file reads
    - Provides fallback embedded prompts when files are missing
    - Thread-safe caching with locks
    - Validates prompt content before caching
    """
    
    # Class-level cache for loaded prompts
    _cache: Dict[str, str] = {}
    
    # Thread lock for cache operations
    _cache_lock = threading.Lock()
    
    # Directory containing prompt files
    _prompts_dir = Path(__file__).parent.resolve()
    
    @classmethod
    def load_prompt(cls, prompt_name: str) -> str:
        """
        Load a prompt by name with caching and thread safety.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            
        Returns:
            The prompt text as a string
            
        Example:
            >>> prompt = PromptLoader.load_prompt("react_system")
            >>> print(prompt[:50])
            You are an expert LTL Claims Processing Agent...
        """
        # Check cache first (with lock)
        with cls._cache_lock:
            if prompt_name in cls._cache:
                logger.debug(f"Loaded prompt '{prompt_name}' from cache")
                return cls._cache[prompt_name]
        
        # Construct file path
        prompt_file = cls._prompts_dir / f"{prompt_name}.txt"
        
        # Try to load from file (outside lock for better concurrency)
        if prompt_file.exists():
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                
                # Validate prompt is not empty
                if not prompt or not prompt.strip():
                    logger.warning(
                        f"Prompt file {prompt_file} is empty. Falling back to embedded prompt."
                    )
                    fallback_prompt = cls._get_fallback_prompt(prompt_name)
                    with cls._cache_lock:
                        cls._cache[prompt_name] = fallback_prompt
                    return fallback_prompt
                
                # Cache the loaded prompt (with lock)
                with cls._cache_lock:
                    cls._cache[prompt_name] = prompt
                
                logger.info(f"Loaded prompt '{prompt_name}' from file: {prompt_file}")
                return prompt
                
            except (IOError, OSError, UnicodeDecodeError) as e:
                logger.warning(
                    f"Failed to read prompt file {prompt_file}: {type(e).__name__}: {e}. "
                    f"Falling back to embedded prompt."
                )
        else:
            logger.warning(
                f"Prompt file not found: {prompt_file}. "
                f"Falling back to embedded prompt."
            )
        
        # Fallback to embedded prompt (with lock)
        fallback_prompt = cls._get_fallback_prompt(prompt_name)
        with cls._cache_lock:
            cls._cache[prompt_name] = fallback_prompt
        return fallback_prompt
    
    @classmethod
    def _get_fallback_prompt(cls, prompt_name: str) -> str:
        """
        Get fallback embedded prompt when file cannot be loaded.
        
        Args:
            prompt_name: Name of the prompt
            
        Returns:
            Fallback prompt text
        """
        fallbacks = {
            "react_system": REACT_SYSTEM_FALLBACK,
            "react_system_prompt": REACT_SYSTEM_FALLBACK,  # Alias
        }
        
        prompt = fallbacks.get(prompt_name, DEFAULT_FALLBACK)
        logger.info(f"Using fallback embedded prompt for '{prompt_name}'")
        return prompt
    
    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the prompt cache (thread-safe).
        
        Useful for testing or when prompts are updated and need to be reloaded.
        """
        with cls._cache_lock:
            cls._cache.clear()
        logger.info("Prompt cache cleared")
    
    @classmethod
    def get_cached_prompts(cls) -> Dict[str, str]:
        """
        Get all currently cached prompts (thread-safe).
        
        Returns:
            Dictionary of cached prompt names and their content
        """
        with cls._cache_lock:
            return cls._cache.copy()
    
    @classmethod
    def preload_prompts(cls, prompt_names: List[str]) -> None:
        """
        Preload multiple prompts into cache.
        
        Args:
            prompt_names: List of prompt names to preload
            
        Example:
            >>> PromptLoader.preload_prompts(["react_system", "validation_prompt"])
        """
        for prompt_name in prompt_names:
            cls.load_prompt(prompt_name)
        
        logger.info(f"Preloaded {len(prompt_names)} prompts into cache")
    
    @classmethod
    def register_prompt(cls, prompt_name: str, prompt_text: str) -> None:
        """
        Register a prompt directly without file I/O (useful for testing).
        
        Args:
            prompt_name: Name of the prompt
            prompt_text: Prompt content
            
        Example:
            >>> PromptLoader.register_prompt("test_prompt", "Test content")
        """
        with cls._cache_lock:
            cls._cache[prompt_name] = prompt_text
        logger.debug(f"Registered prompt '{prompt_name}' directly")
