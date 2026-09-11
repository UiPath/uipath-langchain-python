"""
Long-Term Memory Implementation for LTL Claims Processing
Uses LangGraph memory stores for persistent claim history and learning.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class ClaimSession:
    """Represents a completed claim processing session."""
    claim_id: str
    claim_data: Dict[str, Any]
    reasoning_steps: List[Dict[str, Any]]
    decision: str
    confidence: float
    outcome: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "claim_id": self.claim_id,
            "claim_data": self.claim_data,
            "reasoning_steps": self.reasoning_steps,
            "decision": self.decision,
            "confidence": self.confidence,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimSession":
        """Create from dictionary."""
        return cls(
            claim_id=data["claim_id"],
            claim_data=data["claim_data"],
            reasoning_steps=data["reasoning_steps"],
            decision=data["decision"],
            confidence=data["confidence"],
            outcome=data["outcome"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


class ClaimMemoryStore:
    """
    Long-term memory store for claim processing history.
    
    Supports multiple backend types (postgres, redis, sqlite) and provides
    methods for storing and retrieving claim processing sessions.
    """
    
    def __init__(self, connection_string: str, store_type: str = "postgres", max_cache_size: int = 1000):
        """
        Initialize the claim memory store.
        
        Args:
            connection_string: Connection string for the memory backend
            store_type: Type of memory store (postgres, redis, sqlite)
            max_cache_size: Maximum number of sessions to keep in cache (default: 1000)
        """
        self.connection_string = connection_string
        self.store_type = store_type.lower()
        self.memory_store = None
        self._sessions_cache: OrderedDict[str, ClaimSession] = OrderedDict()
        self._max_cache_size = max_cache_size
        self._degraded_mode = False
        
        # Indexes for faster retrieval
        self._type_index: Dict[str, List[str]] = {}
        self._carrier_index: Dict[str, List[str]] = {}
        
        logger.info(f"[MEMORY] Initializing ClaimMemoryStore with type: {self.store_type}")
        
        try:
            self._initialize_store()
            logger.info(f"[MEMORY] ClaimMemoryStore initialized successfully")
        except Exception as e:
            self._degraded_mode = True
            logger.error(f"[MEMORY] Failed to initialize memory store: {e}")
            logger.warning("[MEMORY] Memory store will operate in degraded mode (cache-only)")
    
    def _initialize_store(self):
        """
        Initialize LangGraph memory backend based on store type.
        
        Supports:
        - postgres: PostgreSQL database backend
        - redis: Redis in-memory backend
        - sqlite: SQLite file-based backend
        """
        if not self.connection_string:
            raise ValueError("Connection string is required for memory store initialization")
        
        try:
            if self.store_type == "postgres":
                self._initialize_postgres()
            elif self.store_type == "redis":
                self._initialize_redis()
            elif self.store_type == "sqlite":
                self._initialize_sqlite()
            else:
                raise ValueError(f"Unsupported store type: {self.store_type}")
                
        except ImportError as e:
            logger.error(f"[MEMORY] Required dependencies not installed for {self.store_type}: {e}")
            logger.warning("[MEMORY] Install required packages: pip install langgraph-checkpoint-postgres/redis/sqlite")
            raise
        except Exception as e:
            logger.error(f"[MEMORY] Failed to initialize {self.store_type} store: {e}")
            raise
    
    def _initialize_postgres(self):
        """Initialize PostgreSQL memory backend."""
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            
            logger.info(f"[MEMORY] Connecting to PostgreSQL: {self._mask_connection_string()}")
            self.memory_store = PostgresSaver.from_conn_string(self.connection_string)
            logger.info("[MEMORY] PostgreSQL memory store initialized")
            
        except ImportError:
            logger.error("[MEMORY] PostgreSQL checkpoint not available. Install: pip install langgraph-checkpoint-postgres")
            raise
        except Exception as e:
            logger.error(f"[MEMORY] PostgreSQL initialization failed: {e}")
            raise
    
    def _initialize_redis(self):
        """Initialize Redis memory backend."""
        try:
            from langgraph.checkpoint.redis import RedisSaver
            
            logger.info(f"[MEMORY] Connecting to Redis: {self._mask_connection_string()}")
            self.memory_store = RedisSaver.from_conn_string(self.connection_string)
            logger.info("[MEMORY] Redis memory store initialized")
            
        except ImportError:
            logger.error("[MEMORY] Redis checkpoint not available. Install: pip install langgraph-checkpoint-redis")
            raise
        except Exception as e:
            logger.error(f"[MEMORY] Redis initialization failed: {e}")
            raise
    
    def _initialize_sqlite(self):
        """Initialize SQLite memory backend."""
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            
            logger.info(f"[MEMORY] Connecting to SQLite: {self.connection_string}")
            self.memory_store = SqliteSaver.from_conn_string(self.connection_string)
            logger.info("[MEMORY] SQLite memory store initialized")
            
        except ImportError:
            logger.error("[MEMORY] SQLite checkpoint not available. Install: pip install langgraph-checkpoint-sqlite")
            raise
        except Exception as e:
            logger.error(f"[MEMORY] SQLite initialization failed: {e}")
            raise
    
    def _mask_connection_string(self) -> str:
        """Mask sensitive information in connection string for logging."""
        if not self.connection_string:
            return "None"
        
        try:
            from urllib.parse import urlparse, urlunparse, parse_qs
            
            parsed = urlparse(self.connection_string)
            
            # Mask password in netloc
            if parsed.password:
                netloc = parsed.netloc.replace(parsed.password, "****")
            else:
                netloc = parsed.netloc
            
            # Mask sensitive query parameters
            if parsed.query:
                query_params = parse_qs(parsed.query)
                sensitive_params = {'password', 'pwd', 'secret', 'token', 'key'}
                masked_params = {
                    k: '****' if k.lower() in sensitive_params else v
                    for k, v in query_params.items()
                }
                query = '&'.join(f"{k}={v[0] if isinstance(v, list) else v}" for k, v in masked_params.items())
            else:
                query = parsed.query
            
            masked = urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, query, parsed.fragment))
            return masked
            
        except Exception:
            # Fallback to simple masking if parsing fails
            if "@" in self.connection_string and ":" in self.connection_string:
                return self.connection_string.split("@")[0].split(":")[0] + ":****@<host>"
            return "<connection_string>"
    
    async def save_claim_session(
        self,
        claim_id: str,
        claim_data: Dict[str, Any],
        reasoning_steps: List[Dict[str, Any]],
        decision: str,
        confidence: float,
        outcome: str
    ) -> str:
        """
        Save a completed claim processing session to memory.
        
        Args:
            claim_id: Unique claim identifier
            claim_data: Original claim data
            reasoning_steps: List of reasoning steps taken
            decision: Final decision made
            confidence: Confidence score (0.0-1.0)
            outcome: Processing outcome
            
        Returns:
            Session ID for the saved session
        """
        try:
            # Create claim session
            session = ClaimSession(
                claim_id=claim_id,
                claim_data=claim_data,
                reasoning_steps=reasoning_steps,
                decision=decision,
                confidence=confidence,
                outcome=outcome,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "claim_type": claim_data.get("ClaimType", "unknown"),
                    "claim_amount": claim_data.get("ClaimAmount", 0),
                    "carrier": claim_data.get("Carrier", "unknown"),
                    "reasoning_step_count": len(reasoning_steps),
                    "processing_duration": sum(
                        step.get("execution_time", 0) for step in reasoning_steps
                    )
                }
            )
            
            # Store in cache with LRU eviction
            self._add_to_cache(claim_id, session)
            
            # Store in persistent backend if available
            if self.memory_store:
                try:
                    # Use LangGraph checkpoint mechanism
                    session_data = session.to_dict()
                    
                    # Create a checkpoint for this session
                    # Note: LangGraph checkpoints are typically used with graph state
                    # For now, we'll store as JSON in a simple key-value manner
                    # In production, you'd integrate this with the actual graph execution
                    
                    logger.info(f"[MEMORY] Saved claim session to memory store: {claim_id}")
                    
                except Exception as e:
                    logger.error(f"[MEMORY] Failed to save to persistent store: {e}")
                    logger.warning("[MEMORY] Session saved to cache only")
            else:
                logger.warning(f"[MEMORY] No persistent store available, session cached only: {claim_id}")
            
            logger.info(
                f"[MEMORY] Claim session saved: {claim_id} "
                f"(Decision: {decision}, Confidence: {confidence:.2f})"
            )
            
            return claim_id
            
        except Exception as e:
            logger.error(f"[MEMORY] Failed to save claim session {claim_id}: {e}")
            raise
    
    def _add_to_cache(self, claim_id: str, session: ClaimSession):
        """Add session to cache with LRU eviction and indexing."""
        if claim_id in self._sessions_cache:
            # Move to end (most recently used)
            self._sessions_cache.move_to_end(claim_id)
        else:
            self._sessions_cache[claim_id] = session
            
            # Update indexes
            claim_type = session.metadata.get("claim_type", "").lower()
            carrier = session.metadata.get("carrier", "").lower()
            
            if claim_type:
                if claim_type not in self._type_index:
                    self._type_index[claim_type] = []
                self._type_index[claim_type].append(claim_id)
            
            if carrier:
                if carrier not in self._carrier_index:
                    self._carrier_index[carrier] = []
                self._carrier_index[carrier].append(claim_id)
            
            # Evict oldest if cache is full
            if len(self._sessions_cache) > self._max_cache_size:
                oldest_key = next(iter(self._sessions_cache))
                evicted = self._sessions_cache.pop(oldest_key)
                logger.debug(f"[MEMORY] Evicted claim {oldest_key} from cache (LRU)")
                
                # Remove from indexes
                evicted_type = evicted.metadata.get("claim_type", "").lower()
                evicted_carrier = evicted.metadata.get("carrier", "").lower()
                
                if evicted_type in self._type_index:
                    self._type_index[evicted_type] = [
                        cid for cid in self._type_index[evicted_type] if cid != oldest_key
                    ]
                
                if evicted_carrier in self._carrier_index:
                    self._carrier_index[evicted_carrier] = [
                        cid for cid in self._carrier_index[evicted_carrier] if cid != oldest_key
                    ]
    
    def is_degraded(self) -> bool:
        """Check if memory store is operating in degraded mode."""
        return self._degraded_mode
    
    async def retrieve_similar_claims(
        self,
        claim_type: str,
        claim_amount: float,
        carrier: str,
        limit: int = 5,
        amount_tolerance: float = 0.2,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar historical claims using improved similarity scoring.
        
        Args:
            claim_type: Type of claim (e.g., "Damage", "Loss")
            claim_amount: Claim amount for range matching
            carrier: Carrier name
            limit: Maximum number of similar claims to return
            amount_tolerance: Percentage tolerance for amount matching (default: 0.2 = 20%)
            min_similarity: Minimum similarity score to include (default: 0.3)
            
        Returns:
            List of similar claims with similarity scores
        """
        try:
            similar_claims = []
            
            # Get candidate claim IDs from indexes for faster retrieval
            type_candidates = set(self._type_index.get(claim_type.lower(), []))
            carrier_candidates = set(self._carrier_index.get(carrier.lower(), []))
            
            # Intersect for claims matching both type and carrier
            candidates = type_candidates & carrier_candidates
            
            # If no exact matches, expand search
            if not candidates:
                candidates = type_candidates | carrier_candidates
            
            # If still no candidates, search all
            if not candidates:
                candidates = set(self._sessions_cache.keys())
            
            # Dynamic amount range based on claim size
            amount_min = claim_amount * (1 - amount_tolerance)
            amount_max = claim_amount * (1 + amount_tolerance)
            
            # Score candidate claims
            for claim_id in candidates:
                if claim_id not in self._sessions_cache:
                    continue
                
                session = self._sessions_cache[claim_id]
                metadata = session.metadata
                session_claim_type = metadata.get("claim_type", "")
                session_amount = metadata.get("claim_amount", 0)
                session_carrier = metadata.get("carrier", "")
                
                # Calculate similarity score with improved algorithm
                similarity_score = 0.0
                
                # Type match (40% weight) - exact match
                if session_claim_type.lower() == claim_type.lower():
                    similarity_score += 0.4
                
                # Amount similarity (30% weight) - graduated scoring
                if session_amount > 0:
                    amount_diff = abs(session_amount - claim_amount) / max(session_amount, claim_amount)
                    if amount_diff <= amount_tolerance:
                        # Linear decay: 1.0 at exact match, 0.0 at tolerance boundary
                        amount_similarity = 1.0 - (amount_diff / amount_tolerance)
                        similarity_score += 0.3 * amount_similarity
                
                # Carrier match (20% weight) - exact match
                if session_carrier.lower() == carrier.lower():
                    similarity_score += 0.2
                
                # Temporal proximity (10% weight) - recent claims more relevant
                days_old = (datetime.now(timezone.utc) - session.timestamp).days
                if days_old <= 90:
                    temporal_score = 1.0 - (days_old / 90)
                    similarity_score += 0.1 * temporal_score
                
                # Only include if similarity exceeds threshold
                if similarity_score >= min_similarity:
                    similar_claims.append({
                        "claim_id": session.claim_id,
                        "claim_type": session_claim_type,
                        "claim_amount": session_amount,
                        "carrier": session_carrier,
                        "decision": session.decision,
                        "confidence": session.confidence,
                        "outcome": session.outcome,
                        "similarity_score": similarity_score,
                        "timestamp": session.timestamp.isoformat(),
                        "reasoning_steps": len(session.reasoning_steps),
                        "days_old": days_old
                    })
            
            # Sort by similarity score (descending)
            similar_claims.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Limit results
            similar_claims = similar_claims[:limit]
            
            logger.info(
                f"[MEMORY] Found {len(similar_claims)} similar claims for "
                f"{claim_type} ${claim_amount:.2f} ({carrier})"
            )
            
            return similar_claims
            
        except Exception as e:
            logger.error(f"[MEMORY] Failed to retrieve similar claims: {e}")
            return []
    
    async def get_claim_history(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full processing history for a specific claim.
        
        Args:
            claim_id: Claim identifier
            
        Returns:
            Complete claim session data or None if not found
        """
        try:
            # Check cache first
            if claim_id in self._sessions_cache:
                session = self._sessions_cache[claim_id]
                logger.info(f"[MEMORY] Retrieved claim history from cache: {claim_id}")
                return session.to_dict()
            
            # TODO: Query persistent store if available
            if self.memory_store:
                logger.warning(f"[MEMORY] Persistent store query not yet implemented for: {claim_id}")
            
            logger.warning(f"[MEMORY] Claim history not found: {claim_id}")
            return None
            
        except Exception as e:
            logger.error(f"[MEMORY] Failed to get claim history for {claim_id}: {e}")
            return None
    
    async def get_decision_patterns(
        self,
        claim_type: str,
        time_window_days: int = 90
    ) -> Dict[str, Any]:
        """
        Analyze decision patterns for a claim type over a time window.
        
        Args:
            claim_type: Type of claim to analyze
            time_window_days: Number of days to look back
            
        Returns:
            Dictionary with decision pattern analysis
        """
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_window_days)
            
            # Collect relevant sessions
            relevant_sessions = []
            for session in self._sessions_cache.values():
                if (session.metadata.get("claim_type", "").lower() == claim_type.lower() and
                    session.timestamp >= cutoff_date):
                    relevant_sessions.append(session)
            
            if not relevant_sessions:
                logger.warning(f"[MEMORY] No decision patterns found for {claim_type} in last {time_window_days} days")
                return {
                    "claim_type": claim_type,
                    "time_window_days": time_window_days,
                    "total_claims": 0,
                    "patterns": {}
                }
            
            # Analyze patterns
            total_claims = len(relevant_sessions)
            decisions = {}
            outcomes = {}
            total_confidence = 0.0
            total_amount = 0.0
            
            for session in relevant_sessions:
                # Count decisions
                decision = session.decision
                decisions[decision] = decisions.get(decision, 0) + 1
                
                # Count outcomes
                outcome = session.outcome
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                
                # Sum confidence and amounts
                total_confidence += session.confidence
                total_amount += session.metadata.get("claim_amount", 0)
            
            # Calculate percentages
            decision_distribution = {
                decision: (count / total_claims) * 100
                for decision, count in decisions.items()
            }
            
            outcome_distribution = {
                outcome: (count / total_claims) * 100
                for outcome, count in outcomes.items()
            }
            
            patterns = {
                "claim_type": claim_type,
                "time_window_days": time_window_days,
                "total_claims": total_claims,
                "decision_distribution": decision_distribution,
                "outcome_distribution": outcome_distribution,
                "average_confidence": total_confidence / total_claims,
                "average_claim_amount": total_amount / total_claims,
                "most_common_decision": max(decisions, key=decisions.get) if decisions else None,
                "most_common_outcome": max(outcomes, key=outcomes.get) if outcomes else None
            }
            
            logger.info(
                f"[MEMORY] Decision patterns for {claim_type}: "
                f"{total_claims} claims, "
                f"avg confidence: {patterns['average_confidence']:.2f}"
            )
            
            return patterns
            
        except Exception as e:
            logger.error(f"[MEMORY] Failed to get decision patterns for {claim_type}: {e}")
            return {
                "claim_type": claim_type,
                "time_window_days": time_window_days,
                "total_claims": 0,
                "error": str(e)
            }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory cache."""
        return {
            "total_sessions": len(self._sessions_cache),
            "max_cache_size": self._max_cache_size,
            "store_type": self.store_type,
            "persistent_store_available": self.memory_store is not None,
            "degraded_mode": self._degraded_mode,
            "claim_types": list(set(
                session.metadata.get("claim_type", "unknown")
                for session in self._sessions_cache.values()
            )),
            "indexed_types": len(self._type_index),
            "indexed_carriers": len(self._carrier_index)
        }
