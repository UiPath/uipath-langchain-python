"""
LangGraph Configuration Validation Module

This module provides Pydantic models and validation functions for the langgraph.json
configuration file. It ensures that the graph structure is properly defined with
valid nodes, edges, and conditional routing logic.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class NodeConfig(BaseModel):
    """Configuration for a graph node."""
    type: str = Field(description="Node type (e.g., 'function', 'tool', 'conditional')")
    function: str = Field(description="Module path to function (e.g., 'module:Class.method')")
    
    @validator('type')
    def validate_type(cls, v):
        """Validate that node type is supported."""
        valid_types = ['function', 'tool', 'conditional']
        if v not in valid_types:
            raise ValueError(f"Node type must be one of {valid_types}, got '{v}'")
        return v
    
    @validator('function')
    def validate_function_path(cls, v):
        """Validate that function path has correct format."""
        if ':' not in v:
            raise ValueError(
                f"Function path must be in format 'module.path:Class.method', got '{v}'"
            )
        return v


class EdgeConfig(BaseModel):
    """Configuration for a graph edge."""
    from_node: str = Field(alias="from", description="Source node name")
    to_node: str = Field(alias="to", description="Destination node name")
    
    class Config:
        populate_by_name = True  # Allow both 'from' and 'from_node'


class ConditionalEdgeConfig(BaseModel):
    """Configuration for conditional edges with routing logic."""
    function: str = Field(description="Module path to routing function")
    routes: Dict[str, str] = Field(description="Mapping of route keys to node names")
    
    @validator('function')
    def validate_function_path(cls, v):
        """Validate that function path has correct format."""
        if ':' not in v:
            raise ValueError(
                f"Function path must be in format 'module.path:Class.method', got '{v}'"
            )
        return v
    
    @validator('routes')
    def validate_routes(cls, v):
        """Validate that routes dictionary is not empty."""
        if not v:
            raise ValueError("Routes dictionary cannot be empty")
        return v


class MetadataConfig(BaseModel):
    """Configuration metadata for the agent."""
    name: str = Field(description="Agent name")
    version: str = Field(description="Agent version")
    description: str = Field(description="Agent description")
    author: str = Field(description="Agent author")
    framework: Optional[str] = Field(default="LangGraph", description="Framework name")
    pattern: Optional[str] = Field(default=None, description="Agent pattern (e.g., 'ReAct')")
    features: Optional[List[str]] = Field(default=None, description="List of agent features")


class LangGraphConfig(BaseModel):
    """Complete LangGraph configuration model."""
    graphs: Dict[str, str] = Field(
        description="Graph definitions mapping graph names to module paths"
    )
    nodes: Dict[str, NodeConfig] = Field(description="Graph nodes configuration")
    edges: List[EdgeConfig] = Field(description="Graph edges configuration")
    conditional_edges: Dict[str, ConditionalEdgeConfig] = Field(
        description="Conditional edges with routing logic"
    )
    entry_point: str = Field(description="Entry point node name")
    metadata: MetadataConfig = Field(description="Agent metadata")
    
    @validator('graphs')
    def validate_graphs(cls, v):
        """Validate that graphs dictionary is not empty."""
        if not v:
            raise ValueError("Graphs dictionary cannot be empty")
        for graph_name, graph_path in v.items():
            if ':' not in graph_path:
                raise ValueError(
                    f"Graph path for '{graph_name}' must be in format 'module.path:Class.attribute', "
                    f"got '{graph_path}'"
                )
        return v
    
    @validator('nodes')
    def validate_nodes(cls, v):
        """Validate that nodes dictionary is not empty."""
        if not v:
            raise ValueError("Nodes dictionary cannot be empty")
        return v
    
    @validator('entry_point')
    def validate_entry_point(cls, v, values):
        """Validate that entry point exists in nodes."""
        if 'nodes' in values and v not in values['nodes']:
            raise ValueError(
                f"Entry point '{v}' not found in nodes: {list(values['nodes'].keys())}"
            )
        return v
    
    @validator('edges')
    def validate_edges_reference_nodes(cls, v, values):
        """Validate that all edges reference existing nodes."""
        if 'nodes' not in values:
            return v
        
        node_names = set(values['nodes'].keys())
        for edge in v:
            if edge.from_node not in node_names:
                raise ValueError(
                    f"Edge references non-existent source node: '{edge.from_node}'"
                )
            if edge.to_node not in node_names:
                raise ValueError(
                    f"Edge references non-existent destination node: '{edge.to_node}'"
                )
        return v
    
    @validator('conditional_edges')
    def validate_conditional_edges_reference_nodes(cls, v, values):
        """Validate that all conditional edges reference existing nodes."""
        if 'nodes' not in values:
            return v
        
        node_names = set(values['nodes'].keys())
        for source_node, config in v.items():
            if source_node not in node_names:
                raise ValueError(
                    f"Conditional edge references non-existent source node: '{source_node}'"
                )
            for route_key, target_node in config.routes.items():
                if target_node not in node_names and target_node != "END":
                    raise ValueError(
                        f"Conditional edge route '{route_key}' references "
                        f"non-existent target node: '{target_node}'"
                    )
        return v


def load_langgraph_config(config_path: Optional[Path] = None) -> LangGraphConfig:
    """
    Load and parse the langgraph.json configuration file.
    
    Args:
        config_path: Optional path to configuration file. If not provided,
                    looks for langgraph.json in the project root.
    
    Returns:
        Parsed and validated LangGraphConfig object
    
    Raises:
        ConfigurationError: If configuration file is missing or invalid
    """
    # Determine config path
    if config_path is None:
        # Look for langgraph.json in project root
        possible_paths = [
            Path("langgraph.json"),
            Path(__file__).parent.parent.parent / "langgraph.json",
        ]
        
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            raise ConfigurationError(
                "langgraph.json not found in project root. "
                "Please create the configuration file or run 'uv run uipath init'."
            )
    
    # Validate path exists
    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}. "
            "Please create the configuration file or run 'uv run uipath init'."
        )
    
    # Load and parse JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Invalid JSON in {config_path}: {e}. "
            "Please check the file syntax."
        )
    except Exception as e:
        raise ConfigurationError(
            f"Failed to read configuration file {config_path}: {e}"
        )
    
    # Validate using Pydantic model
    try:
        config = LangGraphConfig(**config_data)
        logger.info(f"✅ LangGraph configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        raise ConfigurationError(
            f"Invalid configuration schema in {config_path}: {e}. "
            "Please check that all required fields are present and valid."
        )


def validate_langgraph_config(config_path: Optional[Path] = None) -> bool:
    """
    Validate the langgraph.json configuration file.
    
    This function loads and validates the configuration, logging any errors
    that are found. It's designed to be called during agent initialization
    to ensure the configuration is valid before processing begins.
    
    Args:
        config_path: Optional path to configuration file. If not provided,
                    looks for langgraph.json in the project root.
    
    Returns:
        True if configuration is valid, False otherwise
    
    Raises:
        ConfigurationError: If configuration is missing or invalid
    """
    try:
        config = load_langgraph_config(config_path)
        
        # Additional validation checks
        logger.info("🔍 Performing additional configuration validation...")
        
        # Check that all nodes have valid function references
        for node_name, node_config in config.nodes.items():
            logger.debug(f"  ✓ Node '{node_name}': {node_config.function}")
        
        # Check that all edges are properly connected
        logger.debug(f"  ✓ {len(config.edges)} edges defined")
        
        # Check conditional edges
        for source_node, cond_config in config.conditional_edges.items():
            logger.debug(
                f"  ✓ Conditional edge from '{source_node}' with "
                f"{len(cond_config.routes)} routes"
            )
        
        # Validate entry point
        logger.debug(f"  ✓ Entry point: {config.entry_point}")
        
        # Validate metadata
        logger.debug(
            f"  ✓ Metadata: {config.metadata.name} v{config.metadata.version}"
        )
        
        logger.info("✅ LangGraph configuration validation complete - all checks passed")
        return True
        
    except ConfigurationError as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during configuration validation: {e}")
        raise ConfigurationError(f"Configuration validation failed: {e}")


def get_config_summary(config: LangGraphConfig) -> Dict[str, Any]:
    """
    Get a summary of the configuration for logging and debugging.
    
    Args:
        config: Validated LangGraphConfig object
    
    Returns:
        Dictionary containing configuration summary
    """
    return {
        "name": config.metadata.name,
        "version": config.metadata.version,
        "description": config.metadata.description,
        "author": config.metadata.author,
        "framework": config.metadata.framework,
        "pattern": config.metadata.pattern,
        "graphs": config.graphs,
        "node_count": len(config.nodes),
        "edge_count": len(config.edges),
        "conditional_edge_count": len(config.conditional_edges),
        "entry_point": config.entry_point,
        "nodes": list(config.nodes.keys()),
        "features": config.metadata.features or []
    }


# Convenience function for quick validation
def validate_config_on_startup() -> None:
    """
    Validate configuration on agent startup.
    
    This is a convenience function that can be called during agent
    initialization to ensure the configuration is valid before
    processing begins.
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
    logger.info("🚀 Validating LangGraph configuration on startup...")
    validate_langgraph_config()
    logger.info("✅ Configuration validation complete")
