"""Notification service for managing multi-channel communications and MCP integrations."""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import json
import asyncio
from enum import Enum

from ..models.agent_models import NotificationRecord, DeliveryRecord
from ..config.settings import settings

logger = logging.getLogger(__name__)


class MCPServiceType(str, Enum):
    """MCP service types for external integrations."""
    EMAIL_SENDGRID = "sendgrid"
    EMAIL_AWS_SES = "aws_ses"
    SMS_TWILIO = "twilio"
    SMS_AWS_SNS = "aws_sns"
    WEBHOOK_GENERIC = "webhook"
    SLACK_API = "slack"


class NotificationServiceError(Exception):
    """Custom exception for notification service errors."""
    pass


class NotificationService:
    """Service for managing multi-channel notifications and external MCP integrations."""
    
    def __init__(self):
        self.settings = settings
        self.mcp_configs = self._load_mcp_configurations()
        self.delivery_tracking = {}
        self.template_cache = {}
    
    def _load_mcp_configurations(self) -> Dict[str, Any]:
        """Load MCP service configurations from settings."""
        try:
            # In a real implementation, this would load from environment or config files
            return {
                "email": {
                    "service": self.settings.email_service,
                    "api_key": self.settings.sendgrid_api_key,
                    "from_address": self.settings.email_from_address,
                    "from_name": self.settings.email_from_name
                },

            }
        except Exception as e:
            logger.error(f"Failed to load MCP configurations: {e}")
            return {}
    
    async def send_email_notification(
        self,
        recipient: str,
        subject: str,
        body: str,
        priority: str = "medium",
        notification_id: str = None
    ) -> Dict[str, Any]:
        """Send email notification through MCP email service."""
        try:
            email_config = self.mcp_configs.get("email", {})
            service_type = email_config.get("service", "sendgrid")
            
            if service_type == "sendgrid":
                return await self._send_sendgrid_email(
                    recipient, subject, body, priority, notification_id, email_config
                )
            elif service_type == "aws_ses":
                return await self._send_aws_ses_email(
                    recipient, subject, body, priority, notification_id, email_config
                )
            else:
                raise NotificationServiceError(f"Unsupported email service: {service_type}")
                
        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {e}")
            raise NotificationServiceError(f"Email sending failed: {str(e)}")
    
    async def _send_sendgrid_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        priority: str,
        notification_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send email through SendGrid API."""
        try:
            import os
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            
            # Use the API key from settings
            api_key = self.settings.sendgrid_api_key
            
            # Create the email message
            message = Mail(
                from_email=config.get("from_address", "noreply@ltlclaims.com"),
                to_emails=recipient,
                subject=subject,
                html_content=f"<html><body>{body.replace(chr(10), '<br>')}</body></html>"
            )
            
            # Add custom args for tracking
            message.custom_arg = {
                "notification_id": notification_id,
                "priority": priority,
                "source": "ltl_claims_agent"
            }
            
            # Send the email
            sg = SendGridAPIClient(api_key=api_key)
            response = sg.send(message)
            
            logger.info(f"SendGrid email sent successfully: {notification_id}, Status: {response.status_code}")
            
            return {
                "message_id": f"sg_{notification_id}",
                "status": "sent",
                "status_code": response.status_code,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"SendGrid email failed: {e}")
            raise
    
    async def _send_aws_ses_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        priority: str,
        notification_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send email through AWS SES MCP service."""
        try:
            # Simulate MCP call to AWS SES
            # In a real implementation, this would use the MCP AWS SES connector
            
            ses_payload = {
                "Source": config.get("from_address"),
                "Destination": {
                    "ToAddresses": [recipient]
                },
                "Message": {
                    "Subject": {"Data": subject},
                    "Body": {"Text": {"Data": body}}
                },
                "Tags": [
                    {"Name": "notification_id", "Value": notification_id},
                    {"Name": "priority", "Value": priority}
                ]
            }
            
            # Simulate API call
            logger.info(f"Sending email via AWS SES: {notification_id}")
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Simulate successful response
            response = {
                "message_id": f"ses_{notification_id}",
                "status": "sent",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"AWS SES email failed: {e}")
            raise
    

    

    

    
    async def track_delivery_status(self, notification_id: str) -> Dict[str, Any]:
        """Track the delivery status of a notification."""
        try:
            # In a real implementation, this would query the MCP services for delivery status
            # For now, simulate delivery tracking
            
            # Check if we have tracking info
            if notification_id in self.delivery_tracking:
                return self.delivery_tracking[notification_id]
            
            # Simulate delivery status check
            delivery_info = {
                "notification_id": notification_id,
                "status": "delivered",
                "delivery_timestamp": datetime.now(timezone.utc).isoformat(),
                "attempts": 1,
                "last_attempt": datetime.now(timezone.utc).isoformat(),
                "error_message": None,
                "provider_response": "Message delivered successfully"
            }
            
            # Cache the delivery info
            self.delivery_tracking[notification_id] = delivery_info
            
            return delivery_info
            
        except Exception as e:
            logger.error(f"Failed to track delivery for {notification_id}: {e}")
            raise NotificationServiceError(f"Delivery tracking failed: {str(e)}")
    
    async def handle_delivery_failure(
        self,
        notification_id: str,
        error_message: str,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Handle notification delivery failures with retry logic."""
        try:
            max_retries = 3
            
            if retry_count >= max_retries:
                # Mark as permanently failed
                failure_info = {
                    "notification_id": notification_id,
                    "status": "failed",
                    "error_message": error_message,
                    "retry_count": retry_count,
                    "final_failure_time": datetime.now(timezone.utc).isoformat()
                }
                
                # Log permanent failure
                logger.error(f"Notification {notification_id} permanently failed after {retry_count} retries: {error_message}")
                
                return failure_info
            
            # Schedule retry
            retry_delay = 2 ** retry_count  # Exponential backoff
            logger.info(f"Scheduling retry for {notification_id} in {retry_delay} seconds (attempt {retry_count + 1})")
            
            # In a real implementation, this would schedule the retry
            # For now, just return retry info
            retry_info = {
                "notification_id": notification_id,
                "status": "retry_scheduled",
                "retry_count": retry_count + 1,
                "retry_delay": retry_delay,
                "next_attempt": (datetime.now(timezone.utc).timestamp() + retry_delay)
            }
            
            return retry_info
            
        except Exception as e:
            logger.error(f"Failed to handle delivery failure for {notification_id}: {e}")
            raise NotificationServiceError(f"Failure handling failed: {str(e)}")
    
    async def get_notification_templates(self) -> Dict[str, Any]:
        """Get available notification templates."""
        try:
            # Return built-in templates
            templates = {
                "claim_received": {
                    "name": "Claim Received",
                    "description": "Sent when a new claim is received",
                    "channels": ["email"],
                    "variables": ["customer_name", "claim_id", "submission_date", "claim_amount"]
                },
                "claim_approved": {
                    "name": "Claim Approved",
                    "description": "Sent when a claim is approved for payment",
                    "channels": ["email"],
                    "variables": ["customer_name", "claim_id", "approved_amount", "payment_method"]
                },
                "claim_rejected": {
                    "name": "Claim Rejected",
                    "description": "Sent when a claim is rejected",
                    "channels": ["email"],
                    "variables": ["customer_name", "claim_id", "rejection_reason"]
                },
                "escalation_created": {
                    "name": "Escalation Created",
                    "description": "Sent when a claim is escalated for review",
                    "channels": ["email"],
                    "variables": ["claim_id", "escalation_reason", "assigned_reviewer"]
                }
            }
            
            return templates
            
        except Exception as e:
            logger.error(f"Failed to get notification templates: {e}")
            raise NotificationServiceError(f"Template retrieval failed: {str(e)}")
    
    async def validate_notification_config(self) -> Dict[str, Any]:
        """Validate notification service configuration."""
        try:
            validation_results = {
                "email": {"configured": False, "errors": []}
            }
            
            # Validate email configuration
            email_config = self.mcp_configs.get("email", {})
            if email_config.get("api_key") and email_config.get("from_address"):
                validation_results["email"]["configured"] = True
            else:
                validation_results["email"]["errors"].append("Missing API key or from address")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate notification config: {e}")
            raise NotificationServiceError(f"Configuration validation failed: {str(e)}")


# Global notification service instance
notification_service = NotificationService()