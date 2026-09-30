"""
Document Analyzer service for parsing and analyzing extracted document information.
Handles damage description extraction, monetary amount parsing, date parsing, and party identification.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


class DocumentAnalyzerError(Exception):
    """Custom exception for document analyzer errors."""
    pass


class DamageInfo(dict):
    """Structured damage information extracted from documents."""
    
    def __init__(
        self,
        damage_type: Optional[str] = None,
        damage_location: Optional[str] = None,
        damage_extent: Optional[str] = None,
        damage_description: Optional[str] = None,
        severity: Optional[str] = None,
        confidence: float = 0.0
    ):
        super().__init__(
            damage_type=damage_type,
            damage_location=damage_location,
            damage_extent=damage_extent,
            damage_description=damage_description,
            severity=severity,
            confidence=confidence
        )


class MonetaryAmount(dict):
    """Structured monetary amount with currency and context."""
    
    def __init__(
        self,
        amount: float,
        currency: str = "USD",
        context: Optional[str] = None,
        confidence: float = 0.0
    ):
        super().__init__(
            amount=amount,
            currency=currency,
            context=context,
            confidence=confidence
        )


class PartyInfo(dict):
    """Structured party/contact information."""
    
    def __init__(
        self,
        party_type: str,
        name: Optional[str] = None,
        company: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        address: Optional[str] = None,
        confidence: float = 0.0
    ):
        super().__init__(
            party_type=party_type,
            name=name,
            company=company,
            email=email,
            phone=phone,
            address=address,
            confidence=confidence
        )


class DocumentAnalyzer:
    """
    Service for analyzing and parsing extracted document information.
    Provides specialized parsing for damage descriptions, monetary amounts, dates, and parties.
    """
    
    def __init__(self):
        """Initialize DocumentAnalyzer with parsing patterns."""
        
        # Damage type keywords
        self.damage_types = {
            "broken": ["broken", "shattered", "cracked", "fractured", "smashed"],
            "dented": ["dented", "dent", "crushed", "compressed"],
            "scratched": ["scratched", "scratch", "scuffed", "abraded"],
            "torn": ["torn", "ripped", "punctured", "hole"],
            "water_damage": ["water damage", "wet", "moisture", "soaked", "damp"],
            "missing": ["missing", "lost", "not received", "absent"],
            "contaminated": ["contaminated", "dirty", "stained", "soiled"],
            "other": ["damaged", "defective", "faulty"]
        }
        
        # Severity keywords
        self.severity_levels = {
            "minor": ["minor", "slight", "small", "minimal"],
            "moderate": ["moderate", "medium", "noticeable"],
            "major": ["major", "significant", "substantial", "severe"],
            "total_loss": ["total loss", "destroyed", "unrepairable", "complete"]
        }
        
        # Currency symbols and codes
        self.currency_patterns = {
            "USD": [r"\$", r"USD", r"US\$"],
            "EUR": [r"€", r"EUR"],
            "GBP": [r"£", r"GBP"],
            "CAD": [r"CAD", r"C\$"]
        }
        
        # Monetary amount patterns
        self.amount_patterns = [
            r"[\$€£]\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",  # $1,234.56
            r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|EUR|GBP|CAD)",  # 1,234.56 USD
            r"(?:amount|total|cost|value|claim)[\s:]+[\$€£]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",  # amount: $1,234.56
        ]
        
        # Date patterns
        self.date_patterns = [
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # MM/DD/YYYY or DD-MM-YYYY
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",  # YYYY-MM-DD
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}",  # Month DD, YYYY
            r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}",  # DD Month YYYY
        ]
        
        # Email pattern
        self.email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        
        # Phone patterns
        self.phone_patterns = [
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # (123) 456-7890 or 123-456-7890
            r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",  # International
        ]
        
        logger.info("DocumentAnalyzer initialized with parsing patterns")
    
    def extract_damage_details(self, text: str, extracted_fields: Optional[Dict[str, Any]] = None) -> DamageInfo:
        """
        Extract damage details from text or extracted fields.
        
        Args:
            text: Raw text to analyze
            extracted_fields: Optional pre-extracted fields from Document Understanding
            
        Returns:
            DamageInfo with structured damage information
        """
        try:
            logger.debug("🔍 Extracting damage details from text")
            
            text_lower = text.lower() if text else ""
            
            # Check extracted fields first
            damage_type = None
            damage_location = None
            damage_extent = None
            damage_description = None
            
            if extracted_fields:
                damage_type = extracted_fields.get("damage_type")
                damage_location = extracted_fields.get("damage_location")
                damage_extent = extracted_fields.get("damage_extent")
                damage_description = extracted_fields.get("damage_description")
            
            # If not in extracted fields, parse from text
            if not damage_type:
                damage_type = self._identify_damage_type(text_lower)
            
            if not damage_description:
                damage_description = self._extract_damage_description(text)
            
            # Determine severity
            severity = self._determine_severity(text_lower)
            
            # Calculate confidence based on how much information we found
            confidence = self._calculate_damage_confidence(
                damage_type, damage_location, damage_extent, damage_description
            )
            
            damage_info = DamageInfo(
                damage_type=damage_type,
                damage_location=damage_location,
                damage_extent=damage_extent,
                damage_description=damage_description,
                severity=severity,
                confidence=confidence
            )
            
            logger.info(f"✅ Damage details extracted: type={damage_type}, severity={severity}, confidence={confidence:.2f}")
            return damage_info
            
        except Exception as e:
            logger.error(f"❌ Failed to extract damage details: {str(e)}")
            return DamageInfo(confidence=0.0)
    
    def extract_monetary_amounts(
        self, 
        text: str, 
        extracted_fields: Optional[Dict[str, Any]] = None
    ) -> List[MonetaryAmount]:
        """
        Extract and validate monetary amounts from text or extracted fields.
        
        Args:
            text: Raw text to analyze
            extracted_fields: Optional pre-extracted fields from Document Understanding
            
        Returns:
            List of MonetaryAmount objects with validation
        """
        try:
            logger.debug("💰 Extracting monetary amounts from text")
            
            amounts = []
            
            # Check extracted fields first
            if extracted_fields:
                for field_name, field_value in extracted_fields.items():
                    if any(keyword in field_name.lower() for keyword in ["amount", "cost", "value", "total", "charge"]):
                        try:
                            amount_value = self._parse_amount_string(str(field_value))
                            if amount_value > 0:
                                amounts.append(MonetaryAmount(
                                    amount=amount_value,
                                    currency="USD",
                                    context=field_name,
                                    confidence=0.9
                                ))
                        except (ValueError, InvalidOperation):
                            pass
            
            # Parse from text if no amounts found in fields
            if not amounts:
                for pattern in self.amount_patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        try:
                            amount_str = match.group(1) if match.lastindex else match.group(0)
                            amount_value = self._parse_amount_string(amount_str)
                            
                            if amount_value > 0:
                                # Get context (surrounding text)
                                context_start = max(0, match.start() - 30)
                                context_end = min(len(text), match.end() + 30)
                                context = text[context_start:context_end].strip()
                                
                                amounts.append(MonetaryAmount(
                                    amount=amount_value,
                                    currency=self._detect_currency(text[max(0, match.start()-10):match.end()+10]),
                                    context=context,
                                    confidence=0.7
                                ))
                        except (ValueError, InvalidOperation):
                            continue
            
            # Remove duplicates and sort by amount
            amounts = self._deduplicate_amounts(amounts)
            
            logger.info(f"✅ Extracted {len(amounts)} monetary amounts")
            return amounts
            
        except Exception as e:
            logger.error(f"❌ Failed to extract monetary amounts: {str(e)}")
            return []
    
    def extract_dates_and_parties(
        self, 
        text: str, 
        extracted_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract dates and party information from text or extracted fields.
        
        Args:
            text: Raw text to analyze
            extracted_fields: Optional pre-extracted fields from Document Understanding
            
        Returns:
            Dictionary with dates and parties information
        """
        try:
            logger.debug("📅 Extracting dates and parties from text")
            
            result = {
                "dates": [],
                "parties": []
            }
            
            # Extract dates
            result["dates"] = self._extract_dates(text, extracted_fields)
            
            # Extract parties
            result["parties"] = self._extract_parties(text, extracted_fields)
            
            logger.info(f"✅ Extracted {len(result['dates'])} dates and {len(result['parties'])} parties")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to extract dates and parties: {str(e)}")
            return {"dates": [], "parties": []}
    
    def _identify_damage_type(self, text_lower: str) -> Optional[str]:
        """Identify damage type from text."""
        for damage_type, keywords in self.damage_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return damage_type
        return None
    
    def _extract_damage_description(self, text: str) -> Optional[str]:
        """Extract damage description from text."""
        # Look for sentences containing damage keywords
        sentences = re.split(r'[.!?]+', text)
        damage_keywords = ["damage", "broken", "torn", "dent", "scratch", "missing"]
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in damage_keywords):
                return sentence.strip()
        
        return None
    
    def _determine_severity(self, text_lower: str) -> Optional[str]:
        """Determine damage severity from text."""
        for severity, keywords in self.severity_levels.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return severity
        return None
    
    def _calculate_damage_confidence(
        self, 
        damage_type: Optional[str],
        damage_location: Optional[str],
        damage_extent: Optional[str],
        damage_description: Optional[str]
    ) -> float:
        """Calculate confidence score for damage extraction."""
        fields_found = sum([
            1 if damage_type else 0,
            1 if damage_location else 0,
            1 if damage_extent else 0,
            1 if damage_description else 0
        ])
        return fields_found / 4.0
    
    def _parse_amount_string(self, amount_str: str) -> float:
        """Parse amount string to float."""
        # Remove currency symbols and commas
        cleaned = re.sub(r'[\$€£,]', '', amount_str)
        cleaned = cleaned.strip()
        
        # Convert to Decimal for precision
        decimal_amount = Decimal(cleaned)
        return float(decimal_amount)
    
    def _detect_currency(self, text: str) -> str:
        """Detect currency from text."""
        for currency, patterns in self.currency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return currency
        return "USD"  # Default to USD
    
    def _deduplicate_amounts(self, amounts: List[MonetaryAmount]) -> List[MonetaryAmount]:
        """Remove duplicate amounts."""
        seen = set()
        unique_amounts = []
        
        for amount in amounts:
            amount_key = (amount["amount"], amount["currency"])
            if amount_key not in seen:
                seen.add(amount_key)
                unique_amounts.append(amount)
        
        return sorted(unique_amounts, key=lambda x: x["amount"], reverse=True)
    
    def _extract_dates(self, text: str, extracted_fields: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract dates from text or extracted fields."""
        dates = []
        
        # Check extracted fields first
        if extracted_fields:
            for field_name, field_value in extracted_fields.items():
                if "date" in field_name.lower() and field_value:
                    try:
                        parsed_date = self._parse_date(str(field_value))
                        if parsed_date:
                            dates.append({
                                "date": parsed_date,
                                "context": field_name,
                                "confidence": 0.9
                            })
                    except Exception:
                        pass
        
        # Parse from text if no dates found
        if not dates:
            for pattern in self.date_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        date_str = match.group(0)
                        parsed_date = self._parse_date(date_str)
                        if parsed_date:
                            # Get context
                            context_start = max(0, match.start() - 20)
                            context_end = min(len(text), match.end() + 20)
                            context = text[context_start:context_end].strip()
                            
                            dates.append({
                                "date": parsed_date,
                                "context": context,
                                "confidence": 0.7
                            })
                    except Exception:
                        continue
        
        return dates
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        date_formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y",
            "%Y-%m-%d", "%Y/%m/%d",
            "%B %d, %Y", "%b %d, %Y",
            "%d %B %Y", "%d %b %Y"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_parties(self, text: str, extracted_fields: Optional[Dict[str, Any]] = None) -> List[PartyInfo]:
        """Extract party information from text or extracted fields."""
        parties = []
        
        # Check extracted fields first
        if extracted_fields:
            party_types = ["shipper", "consignee", "carrier", "claimant", "inspector"]
            for party_type in party_types:
                if party_type in extracted_fields:
                    party_info = PartyInfo(
                        party_type=party_type,
                        name=extracted_fields.get(party_type),
                        company=extracted_fields.get(f"{party_type}_company"),
                        email=extracted_fields.get(f"{party_type}_email"),
                        phone=extracted_fields.get(f"{party_type}_phone"),
                        confidence=0.9
                    )
                    parties.append(party_info)
        
        # Extract emails from text
        emails = re.findall(self.email_pattern, text)
        
        # Extract phone numbers from text
        phones = []
        for pattern in self.phone_patterns:
            phones.extend(re.findall(pattern, text))
        
        # If we found contact info but no parties, create generic party entries
        if (emails or phones) and not parties:
            if emails:
                parties.append(PartyInfo(
                    party_type="contact",
                    email=emails[0] if emails else None,
                    phone=phones[0] if phones else None,
                    confidence=0.6
                ))
        
        return parties


# Global document analyzer instance
document_analyzer = DocumentAnalyzer()
