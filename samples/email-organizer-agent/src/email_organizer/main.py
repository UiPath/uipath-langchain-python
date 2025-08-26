import logging
import os
import json
import re
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from uipath import UiPath
from uipath_langchain.chat import UiPathChat
from uipath.models import CreateAction
from email_organizer.outlook_client import OutlookClient
from difflib import SequenceMatcher

# Configuration
DEFAULT_MODEL = "gpt-4o-2024-08-06"
DEFAULT_CONFIDENCE = 0.0
USER = 'me'
MAX_EMAILS_TO_ANALYZE = 50
MAX_RULES_TO_CREATE = 5

logger = logging.getLogger(__name__)

uipath = UiPath()

class Email(BaseModel):
    id: str
    subject: str
    sender: str
    preview: str = ""

class Rule(BaseModel):
    id: str = "" 
    rule_name: str
    conditions: Dict
   
    actions: Dict = {}
    target_folder: str
    sequence: int = 1
    isEnabled: bool = True
    rule_type: str = ""

class llmRule(BaseModel):
    rule_name: str = Field(description="The unique identifier for the rule")
    conditions: Dict = Field(default={}, description="Conditions must have this form {'predicate': ['value1', 'value2']}")
    target_folder: str = Field(description="FolderName")
    reasoning: str = Field(description="Why this rule is useful")
    rule_type: str = Field(description="NEW or IMPROVED")

class RuleSuggestions(BaseModel):
    """Container for multiple rule suggestions from LLM"""
    rules: List[llmRule] = Field(description="List of email rule suggestions")

class GraphInput(BaseModel):
    max_emails: int
    max_rules: int
    assignee: Optional[str] = None

class GraphOutput(BaseModel):
    success: bool
    rules_created: int
    message: str

class GraphState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    emails: List[Email] = []
    rules: List[Rule] = []
    suggestions: List[Rule] = []
    folders: Dict[str, str] = {} # folder name to ID mapping
    access_token: str = ""
    human_approved: bool = False
    outlook_client: Optional[OutlookClient] = None
    
    max_emails: int = MAX_EMAILS_TO_ANALYZE  # From GraphInput
    max_rules: int = MAX_RULES_TO_CREATE     # From GraphInput
    assignee: Optional[str] = None           # From GraphInput

async def get_access_token(input_config: GraphInput) -> Command:
    """Get access token for Microsoft Graph API and initialize OutlookClient"""
    try:
        connection_key = os.getenv("OUTLOOK_CONNECTION_KEY")
        logger.info(f"Using Outlook connection key: {connection_key}")

        connection_service = uipath.connections 
        token = await connection_service.retrieve_token_async(connection_key)
        access_token = token.access_token

        if not access_token:
            logger.error("Failed to obtain access token")
            return Command(
                update={
                    "output": GraphOutput(
                        success=False, 
                        rules_created=0, 
                        message="Failed to obtain access token"
                    )
                }
            )
        else:
            logger.info("Successfully obtained access token")
            outlook_client = OutlookClient(access_token=access_token)
            return Command(
                update={
                    "access_token": access_token,
                    "outlook_client": outlook_client,
                    "max_emails": input_config.max_emails,    # Pass from input
                    "max_rules": input_config.max_rules,      # Pass from input  
                    "assignee": input_config.assignee         # Pass from input
                }
            )
    except Exception as e:
        logger.error(f"Error retrieving access token: {e}")
        return Command(
            update={
                "output": GraphOutput(
                    success=False, 
                    rules_created=0, 
                    message=f"Error retrieving access token: {e}"
                )
            }
        )
    
async def fetch_emails(state: GraphState) -> Command:
    """Fetch emails from inbox using OutlookClient"""
    try:
        if not state.outlook_client:
            raise Exception("OutlookClient not initialized")
        
        max_emails = state.max_emails
        logger.info(f"Fetching {max_emails} emails from inbox...")
        
        message_data = await state.outlook_client.get_messages(max_emails, "inbox")
        
        emails = []
        for item in message_data:
            try:
                email = Email(
                    id=item.get("id", ""),
                    subject=item.get("subject", "No Subject"),
                    sender=item.get("from", {}).get("emailAddress", {}).get("address", "Unknown Sender"),
                    preview=item.get("bodyPreview", "")
                )
                emails.append(email)
            except Exception as e:
                logger.warning(f"Skipping malformed email: {e}")
                continue

        logger.info(f"Fetched {len(emails)} emails from inbox")
        
        return Command(
            update={
                "emails": emails
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching emails: {e}")
        return Command(
            update={
                "output": GraphOutput(success=False, rules_created=0, message=f"Error fetching emails: {e}")
            }
        )

async def fetch_folders(state: GraphState) -> Command:
    """Fetch all mail folders """
    try:
        if not state.outlook_client:
            raise Exception("OutlookClient not initialized")
        
        logger.info("Fetching all available folders...")
        folders = await state.outlook_client.get_mail_folders(include_subfolders=True)
        
        logger.info(f"Fetched {len(folders)} folders")
        logger.info(f"All available folders: {list(folders.keys())}")
        
        return Command(
            update={
                "folders": folders
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching folders: {e}")
        return Command(
            update={
                "output": GraphOutput(success=False, rules_created=0, message=f"Error fetching folders: {e}")
            }
        )

async def fetch_rules(state: GraphState) -> Command:
    """Extract just moveToFolder actions from existing rules in Outlook"""
    try:
        if not state.outlook_client:
            raise Exception("OutlookClient not initialized")
        
        logger.info("Fetching existing rules from Outlook...")
        
        rules_data = await state.outlook_client.get_message_rules()
        
        rules = []
        for item in rules_data:
            try:
                # Extract target folder from actions
                target_folder = "Unknown"
                actions = item.get("actions", {})
                if "moveToFolder" in actions:
                    folder_id = actions["moveToFolder"]
                    # Find folder name from ID using state.folders
                    for folder_name, fid in state.folders.items():
                        if fid == folder_id:
                            target_folder = folder_name
                            break
                    if target_folder == "Unknown":
                        target_folder = f"Folder_ID_{folder_id[:8]}..."  # Show partial ID if name not found
                
                rule = Rule(
                    id=item.get("id", ""),
                    rule_name=item.get("displayName", "Unnamed Rule"),
                    conditions=item.get("conditions", {}),
                    actions=item.get("actions", {}),
                    target_folder=target_folder,
                    sequence=item.get("sequence", 1),
                    isEnabled=item.get("isEnabled", True),
                    rule_type="EXISTING"
                )
                rules.append(rule)
            except Exception as e:
                logger.warning(f"Skipping malformed rule: {e}")
                continue

        logger.info(f"Fetched {len(rules)} existing rules from Outlook")
        
        return Command(
            update={
                "rules": rules
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching rules: {e}")
        return Command(
            update={
                "output": GraphOutput(success=False, rules_created=0, message=f"Error fetching rules: {e}")
            }
        )

def _infer_conditions_from_rule(llm_rule: llmRule, emails: List[Email]) -> Dict:
    """Infer rule conditions from rule name, reasoning, and email patterns"""
    conditions = {}

    rule_text = f"{llm_rule.rule_name} {llm_rule.reasoning}".lower()
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails_mentioned = re.findall(email_pattern, llm_rule.reasoning, re.IGNORECASE)
    
    if emails_mentioned:
        conditions['senderContains'] = emails_mentioned
    else:
        # Try to infer from common patterns in rule names
        if 'azure' in rule_text:
            conditions['senderContains'] = ['azure', 'microsoft.com']
        elif 'teams' in rule_text:
            conditions['senderContains'] = ['teams.mail.microsoft']
        elif 'sheerid' in rule_text:
            conditions['senderContains'] = ['sheerid.com']
        elif 'upb' in rule_text or 'elearning' in rule_text:
            conditions['senderContains'] = ['upb.ro', 'curs.upb.ro']
        elif 'grid' in rule_text or 'university' in rule_text:
            conditions['senderContains'] = ['grid.pub.ro']
        else:
            # Fallback: analyze actual emails to find patterns
            target_folder = llm_rule.target_folder.lower()
            sender_patterns = []
            
            for email in emails:
                if any(keyword in email.sender.lower() for keyword in target_folder.split()):
                    sender_patterns.append(email.sender.split('@')[-1] if '@' in email.sender else email.sender)
            
            if sender_patterns:
                # Take most common domain
                from collections import Counter
                most_common = Counter(sender_patterns).most_common(1)
                if most_common:
                    conditions['senderContains'] = [most_common[0][0]]
    
    # If no conditions found, provide a generic fallback
    if not conditions:
        conditions = {'senderContains': [llm_rule.target_folder.lower()]}
    
    return conditions

async def llm_node(state: GraphState) -> Command:
    """Generate rule suggestions using LLM analysis of emails"""
    try:
        content = "Emails:\n"
        emails_to_analyze = state.emails

        for idx, email in enumerate(emails_to_analyze, 1):
            content += f"{idx}. Subject: {email.subject}\n"
            content += f"   From: {email.sender}\n"
            content += f"   Preview: {email.preview}...\n\n"
        
        existing_rules_content = ""
        if state.rules:
            existing_rules_content = "EXISTING RULES:\n"
            for idx, rule in enumerate(state.rules, 1):
                status = "ENABLED" if rule.isEnabled else "DISABLED"
                existing_rules_content += f"{idx}. Rule: {rule.rule_name} ({status})\n"
                existing_rules_content += f"   Conditions: {rule.conditions}\n"
                existing_rules_content += f"   Target Folder: {rule.target_folder}\n\n"
        else:
            existing_rules_content = "EXISTING RULES: None found\n"
            
        max_rules = state.max_rules
        prompt = f"""
You are an email organization expert. Analyze the provided emails and create practical Outlook rules for automatic email organization.

===== EMAIL DATA =====
{content}

===== EXISTING RULES =====
{existing_rules_content}

===== TASK =====
Create up to {max_rules} email rules that will automatically organize incoming emails into appropriate folders.

===== ANALYSIS APPROACH =====
1. **Pattern Recognition**: Look for recurring patterns in senders, subjects, and content
2. **Frequency Analysis**: Identify high-volume email sources that need organization
3. **Category Grouping**: Group similar emails by purpose, source, or content type
4. **Practical Organization**: Focus on rules that provide the most organizational value

===== FOLDER STRATEGY =====
- **Specific**: Company names, services, or clear categories when patterns are strong
- **General**: Broad categories (Work, Personal, News) when patterns are mixed
- **Domain-based**: If uncertain about email categorization but many emails from same domain, use the domain name as folder (e.g., "gmail.com", use gmail)
- **Descriptive**: Names that clearly indicate what emails belong there
- **Concise**: Prefer single words or short phrases

===== RULE CONDITIONS =====
Use these predicates based on email data:
- `senderContains`: Match email addresses or domains
- `bodyOrSubjectContains`: Match keywords in subject or body preview
- `subjectContains`: Match keywords only in subject line


===== RULE TYPES =====
- **NEW**: Create a completely new rule for unhandled email patterns
- **IMPROVED**: Enhance an existing rule with better conditions or broader coverage

===== GUIDELINES =====
- Prefer improving existing rules over creating new ones
- Prioritize rules that organize the most emails
- Avoid overlapping or conflicting rules
- Don't duplicate existing rule functionality
- Focus on clear, actionable patterns
- Keep rule names under 30 characters
- Make folder names intuitive and searchable

===== QUALITY CRITERIA =====
- **High Impact**: Rule should catch many relevant emails
- **Clear Purpose**: Obvious why emails belong in the target folder
- **Maintainable**: Conditions are simple and logical
- **Non-Conflicting**: Doesn't interfere with existing rules

Analyze the email patterns and create the most valuable organizational rules.
        """


        llm = UiPathChat(model=DEFAULT_MODEL)

        structured_llm = llm.with_structured_output(RuleSuggestions)
        response = await structured_llm.ainvoke(prompt)
        print(f"LLM raw response: {response}")  

        suggestions = []
        
        if hasattr(response, 'rules'):
            rules_list = response.rules
        elif isinstance(response, dict) and 'rules' in response:
            rules_list = response['rules']
            rules_list = [llmRule(**rule) if isinstance(rule, dict) else rule for rule in rules_list]
        elif isinstance(response, dict):
            rules_list = [llmRule(**response)]
        elif isinstance(response, list):
            rules_list = [llmRule(**rule) if isinstance(rule, dict) else rule for rule in response]
        else:
            logger.warning(f"Unexpected response format: {type(response)}, content: {response}")
            rules_list = []

        for idx, llm_rule in enumerate(rules_list):
            if isinstance(llm_rule, dict):
                rule_dict = llm_rule.copy()
                if 'conditions' not in rule_dict:
                    rule_dict['conditions'] = {}
                try:
                    llm_rule = llmRule(**rule_dict)
                except Exception as e:
                    logger.warning(f"Failed to create llmRule from dict {rule_dict}: {e}")
                    continue
            
            conditions = llm_rule.conditions
            if not conditions:
                conditions = _infer_conditions_from_rule(llm_rule, state.emails)
            
            rule_suggestion = Rule(
                rule_name=llm_rule.rule_name,
                conditions=conditions,
                actions={"moveToFolder": ""},
                target_folder=llm_rule.target_folder,
                sequence=idx + 1,
                isEnabled=True,
                rule_type=llm_rule.rule_type,
            )
            suggestions.append(rule_suggestion)

        logger.info(f"Generated {len(suggestions)} rule suggestions using structured output")

        return Command(
            update={
                "suggestions": suggestions
            }
        )

    except Exception as e:
        logger.error(f"Error generating rule suggestions: {e}")
        return Command(
            update={
                "output": GraphOutput(success=False, rules_created=0, message=f"Error generating suggestions: {e}")
            }
        )   
 
async def wait_for_human_approval(state: GraphState) -> Command:
    """Wait for human approval before proceeding with rule creation"""

    # Format the suggestions for display
    suggestions_text = "\n" + "="*60 + "\n"
    suggestions_text += "EMAIL RULE SUGGESTIONS\n"
    suggestions_text += "="*60 + "\n"
    
    for idx, suggestion in enumerate(state.suggestions, 1):
        suggestions_text += f"\n{idx}.{suggestion.rule_type} Rule: {suggestion.rule_name}\n"
        suggestions_text += f"  Target Folder: {suggestion.target_folder}\n"
        suggestions_text += "  Conditions:\n"
        
        for condition, values in suggestion.conditions.items():
            if isinstance(values, list):
                suggestions_text += f"   - {condition}: {', '.join(values)}\n"
            else:
                suggestions_text += f"   - {condition}: {values}\n"

        # Add explanation based on rule type
        if suggestion.rule_type == "IMPROVED":
            suggestions_text += "   This will enhance an existing rule with additional conditions\n"
        else:
            suggestions_text += "   This will create a new rule\n"

    # Add summary
    new_rules = len([r for r in state.suggestions if r.rule_type == "NEW"])
    improved_rules = len([r for r in state.suggestions if r.rule_type == "IMPROVED"])
    
    # Get unique target folders that don't exist yet
    suggested_folders = set(r.target_folder for r in state.suggestions if r.rule_type == "NEW")
    existing_folders = set(state.folders.keys())
    new_folders_needed = suggested_folders - existing_folders
    existing_folders_used = suggested_folders & existing_folders


    suggestions_text += f"\n" + "="*60 + "\n"
    suggestions_text += f"SUMMARY:\n"
    suggestions_text += f"• {new_rules} new rules will be created\n"
    suggestions_text += f"• {improved_rules} existing rules will be improved\n"
    suggestions_text += f"• Total suggestions: {len(state.suggestions)}\n"

    # Add folder creation information
    if new_folders_needed:
        suggestions_text += f"\nFOLDERS TO BE CREATED ({len(new_folders_needed)}):\n"
        for folder in sorted(new_folders_needed):
            suggestions_text += f"   • {folder}\n"
    
    if existing_folders_used:
        suggestions_text += f"\nEXISTING FOLDERS TO BE USED ({len(existing_folders_used)}):\n"
        for folder in sorted(existing_folders_used):
            suggestions_text += f"   • {folder}\n"
    

    suggestions_text += "="*60 + "\n"
    suggestions_text += "\n Do you want to proceed with creating these rules?\n"
    suggestions_text += "Select 'true' to create the rules\n"
    suggestions_text += "Select 'false' to cancel\n"

    # Create UiPath action for human approval
    logger.info("Displaying suggestions to user for approval...")
    logger.info(suggestions_text)

    action_data = interrupt(CreateAction(
        app_name="escalation_agent_app", 
        title="Email Rule Suggestions - Approval Required",
        data={
            "AgentOutput": suggestions_text,
            "AgentName": "Email Organization Assistant"
        },
        app_version=1,
        assignee=state.assignee,  # Use assignee from input
        app_folder_path=os.getenv("FOLDER_PATH_PLACEHOLDER")
    ))

    # uncomment this to use regular cli --resume
    # action_data = interrupt("\nAgent output:\n" + suggestions_text + "\n\nDo you approve these email rule suggestions? (Yes/No)")

    logger.info(f"Action data received: {action_data}")
    # Wait for human approval
    human_approved = isinstance(action_data.get("Answer"), bool) and action_data["Answer"] is True

    return Command(
        update={
            "human_approved": human_approved
        }
    )

def conditions_overlap(cond1, cond2):
    """Check if two condition dicts have overlapping values."""
    for key in cond1:
        if key in cond2:
            vals1 = set(cond1[key]) if isinstance(cond1[key], list) else {cond1[key]}
            vals2 = set(cond2[key]) if isinstance(cond2[key], list) else {cond2[key]}
            if vals1 & vals2:
                return True
    return False


async def create_rules(state: GraphState) -> Command:
    """Create folders and rules in Outlook, handling NEW and IMPROVED rule types"""
    try:
        if not state.outlook_client:
            raise Exception("OutlookClient not initialized")
        
        created_folders = {}
        rules_created = 0
        rules_updated = 0
        errors = []
        
        logger.info(f"Starting to process {len(state.suggestions)} rule suggestions")
        
        # Refresh folder list to get current state
        logger.info("Refreshing folder list...")
        current_folders = await state.outlook_client.get_mail_folders()
        state.folders.update(current_folders)
        
        # Create a lookup for existing rules by name
        existing_rules = [rule for rule in state.rules if rule.rule_type == "EXISTING"]

        # Process each rule suggestion
        for idx, suggestion in enumerate(state.suggestions, 1):
            logger.info(f"Processing rule {idx}/{len(state.suggestions)}: {suggestion.rule_name} ({suggestion.rule_type})")
            
            try:
                if suggestion.rule_type == "IMPROVED":
                    # Try to match by name first
                    matched_rule = next((r for r in existing_rules if r.rule_name == suggestion.rule_name), None)
                    # If not found, match by folder and overlapping conditions
                    if not matched_rule:
                        for r in existing_rules:
                            folder_match = r.target_folder == suggestion.target_folder
                            cond_match = conditions_overlap(r.conditions, suggestion.conditions)
                            name_similarity = SequenceMatcher(None, r.rule_name, suggestion.rule_name).ratio()
                            if folder_match and cond_match and name_similarity > 0.5:
                                matched_rule = r
                                break
                    if matched_rule:
                        logger.info(f"Updating existing rule: {matched_rule.rule_name}")
                        folder_id = state.folders.get(suggestion.target_folder, matched_rule.actions.get("moveToFolder", ""))
                        if not folder_id:
                            errors.append(f"No folder ID found for improved rule {suggestion.rule_name}")
                            continue
                        try:
                            await state.outlook_client._delete(f"mailFolders/inbox/messageRules/{matched_rule.id}")
                            logger.info(f"Deleted existing rule: {matched_rule.rule_name}")
                        except Exception as e:
                            logger.warning(f"Could not delete existing rule {matched_rule.rule_name}: {e}")
                        rule_data = {
                            "displayName": suggestion.rule_name,
                            "sequence": matched_rule.sequence,
                            "isEnabled": matched_rule.isEnabled,
                            "conditions": suggestion.conditions,
                            "actions": {
                                "moveToFolder": folder_id,
                                "stopProcessingRules": False
                            }
                        }
                        rule_id = await state.outlook_client.create_message_rule(rule_data)
                        if rule_id:
                            rules_updated += 1
                            logger.info(f"Updated rule: {suggestion.rule_name}")
                        else:
                            errors.append(f"Failed to update rule {suggestion.rule_name}")
                    else:
                        logger.warning(f"Cannot improve rule '{suggestion.rule_name}' - no similar existing rule found")
                        errors.append(f"Cannot improve rule '{suggestion.rule_name}' - no similar existing rule found")
                elif suggestion.rule_type == "NEW":
                    # Handle NEW rule - create folder if needed, then create rule
                    folder_name = suggestion.target_folder
                    folder_id = None
                    
                    # Check if folder exists
                    if folder_name in state.folders:
                        folder_id = state.folders[folder_name]
                        logger.info(f"Using existing folder: {folder_name}")
                    elif folder_name in created_folders:
                        folder_id = created_folders[folder_name]
                        logger.info(f"Using previously created folder: {folder_name}")
                    else:
                        # Create new folder
                        logger.info(f"Creating new folder: {folder_name}")
                        folder_id = await state.outlook_client.create_folder(folder_name)
                        
                        if folder_id:
                            created_folders[folder_name] = folder_id
                            state.folders[folder_name] = folder_id
                            logger.info(f"Created folder: {folder_name}")
                        else:
                            errors.append(f"Failed to create folder {folder_name}")
                            continue
                    
                    if not folder_id:
                        errors.append(f"No folder ID found for {folder_name}")
                        continue
                    
                    # Create the new rule
                    rule_data = {
                        "displayName": suggestion.rule_name,
                        "sequence": suggestion.sequence,
                        "isEnabled": suggestion.isEnabled,
                        "conditions": suggestion.conditions,
                        "actions": {
                            "moveToFolder": folder_id,
                            "stopProcessingRules": False
                        }
                    }
                    
                    logger.info(f"Creating new rule: {suggestion.rule_name} -> {folder_name}")
                    rule_id = await state.outlook_client.create_message_rule(rule_data)
                    
                    if rule_id:
                        rules_created += 1
                        logger.info(f"Created rule: {suggestion.rule_name}")
                    else:
                        errors.append(f"Failed to create rule {suggestion.rule_name}")
                
                else:
                    logger.warning(f"Unknown rule type '{suggestion.rule_type}' for rule {suggestion.rule_name}")
                    errors.append(f"Unknown rule type '{suggestion.rule_type}' for rule {suggestion.rule_name}")
                
                # Rate limiting to avoid API throttling
                import asyncio
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing rule {suggestion.rule_name}: {e}")
                errors.append(f"Error processing rule {suggestion.rule_name}: {str(e)}")
                continue
        
        # Log final summary
        logger.info(f"FINAL SUMMARY:")
        logger.info(f"   • Created {len(created_folders)} new folders")
        logger.info(f"   • Created {rules_created} new rules")
        logger.info(f"   • Updated {rules_updated} existing rules")
        logger.info(f"   • Total rules processed: {rules_created + rules_updated}")
        if errors:
            logger.error(f"   • Errors encountered: {len(errors)}")
            for error in errors:
                logger.error(f"     - {error}")
        
        success = (rules_created + rules_updated) > 0
        total_rules = rules_created + rules_updated
        
        return Command(
            update={
                "output": GraphOutput(
                    success=success,
                    rules_created=total_rules,
                    message=f"Created {len(created_folders)} folders, {rules_created} new rules, updated {rules_updated} existing rules. {len(errors)} errors encountered."
                )
            }
        )
        
    except Exception as e:
        logger.error(f"Error in create_folders_and_rules: {e}")
        return Command(
            update={
                "output": GraphOutput(
                    success=False, 
                    rules_created=0, 
                    message=f"Error creating/updating rules: {e}"
                )
            }
        )

def build_graph() -> StateGraph:
    """Build and compile the email organization graph."""
    builder = StateGraph(GraphState, input=GraphInput, output=GraphOutput)
    
    # Add nodes
    builder.add_node("get_token", get_access_token)
    builder.add_node("fetch_emails", fetch_emails)      
    builder.add_node("fetch_folders", fetch_folders)
    builder.add_node("fetch_rules", fetch_rules)        
    builder.add_node("llm_analysis", llm_node)          
    builder.add_node("wait_for_approval", wait_for_human_approval)
    builder.add_node("create_rules", create_rules)
    
    # Add edges
    builder.add_edge(START, "get_token")
    builder.add_edge("get_token", "fetch_emails")
    builder.add_edge("fetch_emails", "fetch_folders") 
    builder.add_edge("fetch_folders", "fetch_rules")    
    builder.add_edge("fetch_rules", "llm_analysis")     
    builder.add_edge("llm_analysis", "wait_for_approval")    
    def should_create_rules(state: GraphState) -> str:
        return "create_rules" if state.human_approved else "END"
    builder.add_conditional_edges(
        "wait_for_approval", 
        should_create_rules,
        {"create_rules": "create_rules", "END": END}
    )
    builder.add_edge("create_rules", END)
    
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    
    return builder.compile(checkpointer=checkpointer, interrupt_before=["wait_for_approval"])

graph = build_graph()