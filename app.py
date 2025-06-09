import gradio as gr
import requests
import json
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import time
import re
from collections import Counter
import threading
import queue
import uuid
from gradio_consilium_roundtable import consilium_roundtable
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, VisitWebpageTool, Tool
from research_tools import EnhancedResearchAgent
from enhanced_search_functions import ENHANCED_SEARCH_FUNCTIONS

# Load environment variables
load_dotenv()

# API Configuration - These will be updated by UI if needed
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
MODERATOR_MODEL = os.getenv("MODERATOR_MODEL", "mistral")

# Session-based storage for isolated discussions
user_sessions: Dict[str, Dict] = {}

# Model Images
avatar_images = {
    "QwQ-32B": "https://cdn-avatars.huggingface.co/v1/production/uploads/620760a26e3b7210c2ff1943/-s1gyJfvbE1RgO5iBeNOi.png",
    "DeepSeek-R1": "https://logosandtypes.com/wp-content/uploads/2025/02/deepseek.svg",
    "Mistral Large": "https://logosandtypes.com/wp-content/uploads/2025/02/mistral-ai.svg",
    "Meta-Llama-3.3-70B-Instruct": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.46.0/files/dark/meta-color.png",
}

def get_session_id(request: gr.Request = None) -> str:
    """Generate or retrieve session ID"""
    if request and hasattr(request, 'session_hash'):
        return request.session_hash
    return str(uuid.uuid4())

def get_or_create_session_state(session_id: str) -> Dict:
    """Get or create isolated session state"""
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "roundtable_state": {
                "participants": [],
                "messages": [],
                "currentSpeaker": None,
                "thinking": [],
                "showBubbles": []
            },
            "discussion_log": [],
            "final_answer": "",
            "api_keys": {
                "mistral": None,
                "sambanova": None
            }
        }
    return user_sessions[session_id]

def update_session_api_keys(mistral_key, sambanova_key, session_id_state, request: gr.Request = None):
    """Update API keys for THIS SESSION ONLY"""
    session_id = get_session_id(request) if not session_id_state else session_id_state
    session = get_or_create_session_state(session_id)
    
    status_messages = []
    
    # Update keys for THIS SESSION
    if mistral_key.strip():
        session["api_keys"]["mistral"] = mistral_key.strip()
        status_messages.append("✅ Mistral API key saved for this session")
    elif MISTRAL_API_KEY:  # Fall back to env var
        session["api_keys"]["mistral"] = MISTRAL_API_KEY
        status_messages.append("✅ Using Mistral API key from environment")
    else:
        status_messages.append("❌ No Mistral API key available")
        
    if sambanova_key.strip():
        session["api_keys"]["sambanova"] = sambanova_key.strip()
        status_messages.append("✅ SambaNova API key saved for this session")
    elif SAMBANOVA_API_KEY:
        session["api_keys"]["sambanova"] = SAMBANOVA_API_KEY
        status_messages.append("✅ Using SambaNova API key from environment")
    else:
        status_messages.append("❌ No SambaNova API key available")
    
    return " | ".join(status_messages), session_id

class VisualConsensusEngine:
    def __init__(self, moderator_model: str = None, update_callback=None, session_id: str = None):
        self.moderator_model = moderator_model or MODERATOR_MODEL
        self.search_agent = EnhancedResearchAgent()
        self.update_callback = update_callback
        self.session_id = session_id
        
        # Get session-specific keys or fall back to global
        session = get_or_create_session_state(session_id) if session_id else {"api_keys": {}}
        session_keys = session.get("api_keys", {})
        
        mistral_key = session_keys.get("mistral") or MISTRAL_API_KEY
        sambanova_key = session_keys.get("sambanova") or SAMBANOVA_API_KEY
        
        # Research Agent stays visible but is no longer an active participant
        self.models = {
            'mistral': {
                'name': 'Mistral Large',
                'api_key': mistral_key,
                'available': bool(mistral_key)
            },
            'sambanova_deepseek': {
                'name': 'DeepSeek-R1',
                'api_key': sambanova_key,
                'available': bool(sambanova_key)
            },
            'sambanova_llama': {
                'name': 'Meta-Llama-3.3-70B-Instruct',
                'api_key': sambanova_key,
                'available': bool(sambanova_key)
            },
            'sambanova_qwq': {
                'name': 'QwQ-32B',
                'api_key': sambanova_key,
                'available': bool(sambanova_key)
            }
        }
        
        # Store session keys for API calls
        self.session_keys = {
            'mistral': mistral_key,
            'sambanova': sambanova_key
        }
        
        # PROFESSIONAL: Strong, expert role definitions matched to decision protocols
        self.roles = {
            'standard': "Provide expert analysis with clear reasoning and evidence.",
            'expert_advocate': "You are a PASSIONATE EXPERT advocating for your specialized position. Present compelling evidence with conviction.",
            'critical_analyst': "You are a RIGOROUS CRITIC. Identify flaws, risks, and weaknesses in arguments with analytical precision.",
            'strategic_advisor': "You are a STRATEGIC ADVISOR. Focus on practical implementation, real-world constraints, and actionable insights.",
            'research_specialist': "You are a RESEARCH EXPERT with deep domain knowledge. Provide authoritative analysis and evidence-based insights.",
            'innovation_catalyst': "You are an INNOVATION EXPERT. Challenge conventional thinking and propose breakthrough approaches."
        }
        
        # PROFESSIONAL: Different prompt styles based on decision protocol
        self.protocol_styles = {
            'consensus': {
                'intensity': 'collaborative',
                'goal': 'finding common ground',
                'language': 'respectful but rigorous'
            },
            'majority_voting': {
                'intensity': 'competitive',
                'goal': 'winning the argument',
                'language': 'passionate advocacy'
            },
            'weighted_voting': {
                'intensity': 'analytical',
                'goal': 'demonstrating expertise',
                'language': 'authoritative analysis'
            },
            'ranked_choice': {
                'intensity': 'comprehensive',
                'goal': 'exploring all options',
                'language': 'systematic evaluation'
            },
            'unanimity': {
                'intensity': 'diplomatic',
                'goal': 'unanimous agreement',
                'language': 'bridge-building dialogue'
            }
        }
    
    def update_visual_state(self, state_update: Dict[str, Any]):
        """Update the visual roundtable state for this session"""
        if self.update_callback:
            self.update_callback(state_update)
    
    def show_research_activity(self, speaker: str, function: str, query: str):
        """Show research happening in the UI with Research Agent activation"""
        # Get current state properly
        session = get_or_create_session_state(self.session_id)
        current_state = session["roundtable_state"]
        all_messages = list(current_state.get("messages", []))  # Make a copy
        participants = current_state.get("participants", [])
        
        # PRESERVE existing bubbles throughout research
        existing_bubbles = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
        
        # Get function display name
        function_display = {
            'search_web': 'Web Search',
            'search_wikipedia': 'Wikipedia',
            'search_academic': 'Academic Papers',
            'search_technology_trends': 'Technology Trends',
            'search_financial_data': 'Financial Data',
            'multi_source_research': 'Multi-Source Research'
        }.get(function, function.replace('_', ' ').title())
        
        # Step 1: Show expert requesting research
        request_message = {
            "speaker": speaker,
            "text": f"🔍 **Research Request**: {function_display}\n📝 Query: \"{query}\"",
            "type": "research_request"
        }
        all_messages.append(request_message)
        
        self.update_visual_state({
            "participants": participants,
            "messages": all_messages,
            "currentSpeaker": speaker,
            "thinking": [],
            "showBubbles": existing_bubbles + [speaker]
        })
        time.sleep(1.5)
        
        # Step 2: Research Agent starts thinking
        self.update_visual_state({
            "participants": participants,
            "messages": all_messages,
            "currentSpeaker": None,
            "thinking": ["Research Agent"],
            "showBubbles": existing_bubbles + [speaker, "Research Agent"]
        })
        time.sleep(2)
        
        # Step 3: Research Agent working - show detailed activity
        working_message = {
            "speaker": "Research Agent",
            "text": f"🔍 **Conducting Research**: {function_display}\n📊 Analyzing: \"{query}\"\n⏳ Please wait while I gather information...",
            "type": "research_activity"
        }
        all_messages.append(working_message)
        
        self.update_visual_state({
            "participants": participants,
            "messages": all_messages,
            "currentSpeaker": "Research Agent",
            "thinking": [],
            "showBubbles": existing_bubbles + [speaker, "Research Agent"]
        })
        time.sleep(3)  # Longer pause to see research happening
        
        # Step 4: Research completion notification
        completion_message = {
            "speaker": "Research Agent",
            "text": f"✅ **Research Complete**: {function_display}\n📋 Results ready for analysis",
            "type": "research_complete"
        }
        all_messages.append(completion_message)
        
        self.update_visual_state({
            "participants": participants,
            "messages": all_messages,
            "currentSpeaker": "Research Agent",
            "thinking": [],
            "showBubbles": existing_bubbles + [speaker, "Research Agent"]
        })
        time.sleep(1.5)
        
        # Step 5: Expert processing results
        processing_message = {
            "speaker": speaker,
            "text": f"📊 **Processing Research Results**\n🧠 Integrating {function_display} findings into analysis...",
            "type": "research_processing"
        }
        all_messages.append(processing_message)
        
        self.update_visual_state({
            "participants": participants,
            "messages": all_messages,
            "currentSpeaker": speaker,
            "thinking": [],
            "showBubbles": existing_bubbles + [speaker, "Research Agent"]  # Keep Research Agent visible longer
        })
        time.sleep(2)

    def log_research_activity(self, speaker: str, function: str, query: str, result: str, log_function=None):
        """Log research activity to the discussion log"""
        if log_function:
            # Log the research request
            log_function('research_request', 
                        speaker="Research Agent", 
                        content=f"Research requested by {speaker}: {function.replace('_', ' ').title()} - '{query}'",
                        function=function,
                        query=query,
                        requesting_expert=speaker)
            
            # Log the research result (truncated for readability)
            result_preview = result[:300] + "..." if len(result) > 300 else result
            log_function('research_result', 
                        speaker="Research Agent", 
                        content=f"Research completed: {function.replace('_', ' ').title()}\n\n{result_preview}",
                        function=function,
                        query=query,
                        full_result=result,
                        requesting_expert=speaker)
    
    def handle_function_calls(self, completion, original_prompt: str, calling_model: str) -> str:
        """UNIFIED function call handler with enhanced research capabilities"""
        
        # Check if completion is valid
        if not completion or not completion.choices or len(completion.choices) == 0:
            print(f"Invalid completion object for {calling_model}")
            return "Analysis temporarily unavailable - invalid API response"
            
        message = completion.choices[0].message
        
        # If no function calls, return regular response
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            content = message.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                    elif isinstance(part, str):
                        text_parts.append(part)
                return ' '.join(text_parts) if text_parts else "Analysis completed"
            elif isinstance(content, str):
                return content
            else:
                return str(content) if content else "Analysis completed"
        
        # Get the calling model's name for UI display
        calling_model_name = self.models[calling_model]['name']
        
        # Process each function call
        messages = [
            {"role": "user", "content": original_prompt}, 
            {
                "role": "assistant", 
                "content": message.content or "",
                "tool_calls": message.tool_calls
            }
        ]
        
        for tool_call in message.tool_calls:
            try:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Show research activity in UI
                query_param = arguments.get("query") or arguments.get("topic") or arguments.get("technology") or arguments.get("company")
                if query_param:
                    self.show_research_activity(calling_model_name, function_name, query_param)
                
                # Execute the enhanced research functions
                result = self._execute_research_function(function_name, arguments)
                
                # Ensure result is a string
                if not isinstance(result, str):
                    result = str(result)

                # Log the research activity (with access to session log function)
                session = get_or_create_session_state(self.session_id)
                def session_log_function(event_type, speaker="", content="", **kwargs):
                    session["discussion_log"].append({
                        'type': event_type,
                        'speaker': speaker,
                        'content': content,
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        **kwargs
                    })

                if query_param and result:
                    self.log_research_activity(calling_model_name, function_name, query_param, result, session_log_function)
                
                # Add function result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                
            except Exception as e:
                print(f"Error processing tool call: {str(e)}")
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id,
                    "content": f"Research error: {str(e)}"
                })
                continue
        
        # Continue conversation with research results integrated
        try:
            from openai import OpenAI
            
            if calling_model == 'mistral':
                client = OpenAI(
                    base_url="https://api.mistral.ai/v1", 
                    api_key=self.session_keys.get('mistral')
                )
                model_name = 'mistral-large-latest'
            else:
                client = OpenAI(
                    base_url="https://api.sambanova.ai/v1", 
                    api_key=self.session_keys.get('sambanova')
                )
                model_mapping = {
                    'sambanova_deepseek': 'DeepSeek-R1',
                    'sambanova_llama': 'Meta-Llama-3.3-70B-Instruct', 
                    'sambanova_qwq': 'QwQ-32B'
                }
                model_name = model_mapping.get(calling_model, 'Meta-Llama-3.3-70B-Instruct')
            
            final_completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            if final_completion and final_completion.choices and len(final_completion.choices) > 0:
                final_content = final_completion.choices[0].message.content
                
                if isinstance(final_content, list):
                    text_parts = []
                    for part in final_content:
                        if isinstance(part, dict) and 'text' in part:
                            text_parts.append(part['text'])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return ' '.join(text_parts) if text_parts else "Analysis completed with research integration."
                elif isinstance(final_content, str):
                    return final_content
                else:
                    return str(final_content) if final_content else "Analysis completed with research integration."
            else:
                return message.content or "Analysis completed with research integration."
            
        except Exception as e:
            print(f"Error in follow-up completion for {calling_model}: {str(e)}")
            return message.content or "Analysis completed with research integration."

    def _execute_research_function(self, function_name: str, arguments: dict) -> str:
        """Execute research function with enhanced capabilities"""
        try:
            if function_name == "search_web":
                depth = arguments.get("depth", "standard")
                return self.search_agent.search(arguments["query"], depth)
                
            elif function_name == "search_wikipedia":
                return self.search_agent.search_wikipedia(arguments["topic"])
                
            elif function_name == "search_academic":
                source = arguments.get("source", "both")
                if source == "arxiv":
                    return self.search_agent.tools['arxiv'].search(arguments["query"])
                elif source == "scholar":
                    return self.search_agent.tools['scholar'].search(arguments["query"])
                else:  # both
                    arxiv_result = self.search_agent.tools['arxiv'].search(arguments["query"])
                    scholar_result = self.search_agent.tools['scholar'].search(arguments["query"])
                    return f"{arxiv_result}\n\n{scholar_result}"
                    
            elif function_name == "search_technology_trends":
                return self.search_agent.tools['github'].search(arguments["technology"])
                
            elif function_name == "search_financial_data":
                return self.search_agent.tools['sec'].search(arguments["company"])
                
            elif function_name == "multi_source_research":
                return self.search_agent.search(arguments["query"], "deep")
                
            else:
                return f"Unknown research function: {function_name}"
                
        except Exception as e:
            return f"Research function error: {str(e)}"
    
    def call_model(self, model: str, prompt: str, context: str = "") -> Optional[str]:
        """Enhanced model calling with native function calling support"""
        if not self.models[model]['available']:
            print(f"Model {model} not available - missing API key")
            return None
            
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        try:
            if model == 'mistral':
                return self._call_mistral(full_prompt)
            elif model.startswith('sambanova_'):
                return self._call_sambanova(model, full_prompt)
        except Exception as e:
            print(f"Error calling {model}: {str(e)}")
            return None
        
        return None
    
    def _call_sambanova(self, model: str, prompt: str) -> Optional[str]:
        """Enhanced SambaNova API call with native function calling"""
        api_key = self.session_keys.get('sambanova')
        if not api_key:
            print(f"No SambaNova API key available for {model}")
            return None
            
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://api.sambanova.ai/v1", 
                api_key=api_key
            )
            
            model_mapping = {
                'sambanova_deepseek': 'DeepSeek-R1',
                'sambanova_llama': 'Meta-Llama-3.3-70B-Instruct', 
                'sambanova_qwq': 'QwQ-32B'
            }
            
            sambanova_model = model_mapping.get(model, 'Meta-Llama-3.3-70B-Instruct')
            print(f"Calling SambaNova model: {sambanova_model}")
            
            # Check if model supports function calling
            supports_functions = sambanova_model in [
                'DeepSeek-R1-0324',
                'Meta-Llama-3.1-8B-Instruct',
                'Meta-Llama-3.1-405B-Instruct', 
                'Meta-Llama-3.3-70B-Instruct'
            ]
            
            if supports_functions:
                completion = client.chat.completions.create(
                    model=sambanova_model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=ENHANCED_SEARCH_FUNCTIONS,
                    tool_choice="auto",
                    max_tokens=1000,
                    temperature=0.7
                )
            else:
                # QwQ-32B and other models that don't support function calling
                print(f"Model {sambanova_model} doesn't support function calling - using regular completion")
                completion = client.chat.completions.create(
                    model=sambanova_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
            
            # Handle function calls if present (only for models that support it)
            if supports_functions:
                return self.handle_function_calls(completion, prompt, model)
            else:
                # For models without function calling, return response directly
                if completion and completion.choices and len(completion.choices) > 0:
                    return completion.choices[0].message.content
                else:
                    return None
            
        except Exception as e:
            print(f"Error calling SambaNova {model} ({sambanova_model}): {str(e)}")
            # Print more detailed error info
            import traceback
            traceback.print_exc()
            return None
    
    def _call_mistral(self, prompt: str) -> Optional[str]:
        """Enhanced Mistral API call with native function calling"""
        api_key = self.session_keys.get('mistral')
        if not api_key:
            print("No Mistral API key available")
            return None
            
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://api.mistral.ai/v1", 
                api_key=api_key
            )
            
            print("Calling Mistral model: mistral-large-latest")
            
            completion = client.chat.completions.create(
                model='mistral-large-latest',
                messages=[{"role": "user", "content": prompt}],
                tools=ENHANCED_SEARCH_FUNCTIONS,
                tool_choice="auto",
                max_tokens=1000,
                temperature=0.7
            )
            
            # Check if we got a valid response
            if not completion or not completion.choices or len(completion.choices) == 0:
                print("Invalid response structure from Mistral")
                return None
                
            # Handle function calls if present
            return self.handle_function_calls(completion, prompt, 'mistral')
            
        except Exception as e:
            print(f"Error calling Mistral API: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def assign_roles(self, models: List[str], role_assignment: str) -> Dict[str, str]:
        """Assign expert roles for rigorous analysis"""
        
        if role_assignment == "none":
            return {model: "standard" for model in models}
        
        roles_to_assign = []
        if role_assignment == "balanced":
            roles_to_assign = ["expert_advocate", "critical_analyst", "strategic_advisor", "research_specialist"]
        elif role_assignment == "specialized":
            roles_to_assign = ["research_specialist", "strategic_advisor", "innovation_catalyst", "expert_advocate"]
        elif role_assignment == "adversarial":
            roles_to_assign = ["critical_analyst", "innovation_catalyst", "expert_advocate", "strategic_advisor"]
        
        while len(roles_to_assign) < len(models):
            roles_to_assign.append("standard")
        
        model_roles = {}
        for i, model in enumerate(models):
            model_roles[model] = roles_to_assign[i % len(roles_to_assign)]
        
        return model_roles
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        if not response or not isinstance(response, str):
            return 5.0
        
        confidence_match = re.search(r'Confidence:\s*(\d+(?:\.\d+)?)', response)
        if confidence_match:
            try:
                return float(confidence_match.group(1))
            except ValueError:
                pass
        return 5.0
    
    def build_position_summary(self, all_messages: List[Dict], current_model: str, topology: str = "full_mesh") -> str:
        """Build expert position summary for analysis"""
        
        current_model_name = self.models[current_model]['name']
        
        if topology == "full_mesh":
            # Show latest position from each expert
            latest_positions = {}
            for msg in all_messages:
                if msg["speaker"] != current_model_name and msg["speaker"] != "Research Agent":
                    latest_positions[msg["speaker"]] = {
                        'text': msg['text'][:150] + "..." if len(msg['text']) > 150 else msg['text'],
                        'confidence': msg.get('confidence', 5)
                    }
            
            summary = "EXPERT POSITIONS:\n"
            for speaker, pos in latest_positions.items():
                summary += f"• **{speaker}**: {pos['text']} (Confidence: {pos['confidence']}/10)\n"
            
        elif topology == "star":
            # Only show moderator's latest position
            moderator_name = self.models[self.moderator_model]['name']
            summary = "MODERATOR ANALYSIS:\n"
            
            for msg in reversed(all_messages):
                if msg["speaker"] == moderator_name:
                    text = msg['text'][:200] + "..." if len(msg['text']) > 200 else msg['text']
                    summary += f"• **{moderator_name}**: {text}\n"
                    break
            
        elif topology == "ring":
            # Only show previous expert's position
            available_models = [model for model, info in self.models.items() if info['available']]
            current_idx = available_models.index(current_model)
            prev_idx = (current_idx - 1) % len(available_models)
            prev_model_name = self.models[available_models[prev_idx]]['name']
            
            summary = "PREVIOUS EXPERT:\n"
            for msg in reversed(all_messages):
                if msg["speaker"] == prev_model_name:
                    text = msg['text'][:200] + "..." if len(msg['text']) > 200 else msg['text']
                    summary += f"• **{prev_model_name}**: {text}\n"
                    break
        
        return summary
    
    def run_visual_consensus_session(self, question: str, discussion_rounds: int = 3, 
                                   decision_protocol: str = "consensus", role_assignment: str = "balanced",
                                   topology: str = "full_mesh", moderator_model: str = "mistral",
                                   log_function=None):
        """Run expert consensus with protocol-appropriate intensity and Research Agent integration"""
        
        # Get only active models (Research Agent is visual-only now)
        available_models = [model for model, info in self.models.items() if info['available']]
        if not available_models:
            return "❌ No AI models available"
        
        model_roles = self.assign_roles(available_models, role_assignment)
        
        # Visual participants include Research Agent but active participants don't
        visual_participant_names = [self.models[model]['name'] for model in available_models] + ["Research Agent"]
        
        # Get protocol-appropriate style
        protocol_style = self.protocol_styles.get(decision_protocol, self.protocol_styles['consensus'])
        
        # Use session-specific logging
        def log_event(event_type: str, speaker: str = "", content: str = "", **kwargs):
            if log_function:
                log_function(event_type, speaker, content, **kwargs)
        
        # Log the start
        log_event('phase', content=f"🎯 Starting Expert Analysis: {question}")
        log_event('phase', content=f"📊 Configuration: {len(available_models)} experts, {decision_protocol} protocol, {role_assignment} roles, {topology} topology")
        
        # Initialize visual state with Research Agent visible
        self.update_visual_state({
            "participants": visual_participant_names,
            "messages": [],
            "currentSpeaker": None,
            "thinking": [],
            "showBubbles": [],
            "avatarImages": avatar_images
        })
        
        all_messages = []
        
        # Phase 1: Initial expert analysis (Research Agent activates only through function calls)
        log_event('phase', content="📝 Phase 1: Expert Initial Analysis")
        
        for model in available_models:
            # Log and set thinking state - PRESERVE BUBBLES
            log_event('thinking', speaker=self.models[model]['name'])
            
            # Calculate existing bubbles
            existing_bubbles = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
            
            self.update_visual_state({
                "participants": visual_participant_names,
                "messages": all_messages,
                "currentSpeaker": None,
                "thinking": [self.models[model]['name']],
                "showBubbles": existing_bubbles,
                "avatarImages": avatar_images
            })
            
            time.sleep(1)
            
            role = model_roles[model]
            role_context = self.roles[role]
            
            # PROTOCOL-ADAPTED: Prompt intensity based on decision protocol
            if decision_protocol in ['majority_voting', 'ranked_choice']:
                intensity_prompt = "🎯 CRITICAL DECISION"
                action_prompt = "Take a STRONG, CLEAR position and defend it with compelling evidence"
                stakes = "This decision has major consequences - be decisive and convincing"
            elif decision_protocol == 'consensus':
                intensity_prompt = "🤝 COLLABORATIVE ANALYSIS"
                action_prompt = "Provide thorough analysis while remaining open to other perspectives"
                stakes = "Work toward building understanding and finding common ground"
            else:  # weighted_voting, unanimity
                intensity_prompt = "🔬 EXPERT ANALYSIS"
                action_prompt = "Provide authoritative analysis with detailed reasoning"
                stakes = "Your expertise and evidence quality will determine influence"
            
            prompt = f"""{intensity_prompt}: {question}

Your Role: {role_context}

ANALYSIS REQUIREMENTS:
- {action_prompt}
- {stakes}
- Use specific examples, data, and evidence
- If you need current information or research, you can search the web, Wikipedia, academic papers, technology trends, or financial data
- Maximum 200 words of focused analysis
- End with "Position: [YOUR CLEAR STANCE]" and "Confidence: X/10"

Provide your expert analysis:"""

            # Log and set speaking state - PRESERVE BUBBLES
            log_event('speaking', speaker=self.models[model]['name'])
            
            # Calculate existing bubbles
            existing_bubbles = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
            
            self.update_visual_state({
                "participants": visual_participant_names,
                "messages": all_messages,
                "currentSpeaker": self.models[model]['name'],
                "thinking": [],
                "showBubbles": existing_bubbles,
                "avatarImages": avatar_images
            })
            
            time.sleep(2)
            
            # Call model - may trigger function calls and Research Agent activation
            response = self.call_model(model, prompt)
            
            # CRITICAL: Ensure response is a string
            if response and not isinstance(response, str):
                response = str(response)
            
            if response:
                confidence = self._extract_confidence(response)
                message = {
                    "speaker": self.models[model]['name'],
                    "text": response,
                    "confidence": confidence,
                    "role": role
                }
                all_messages.append(message)
                
                # Log the full response
                log_event('message', 
                         speaker=self.models[model]['name'], 
                         content=response,
                         role=role,
                         confidence=confidence)
            else:
                # Handle failed API call gracefully
                log_event('message', 
                         speaker=self.models[model]['name'], 
                         content="Analysis temporarily unavailable - API connection failed",
                         role=role,
                         confidence=0)
                
                message = {
                    "speaker": self.models[model]['name'],
                    "text": "⚠️ Analysis temporarily unavailable - API connection failed. Please check your API keys and try again.",
                    "confidence": 0,
                    "role": role
                }
                all_messages.append(message)
            
            # Update with new message
            responded_speakers = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
            
            self.update_visual_state({
                "participants": visual_participant_names,
                "messages": all_messages,
                "currentSpeaker": None,
                "thinking": [],
                "showBubbles": responded_speakers,
                "avatarImages": avatar_images
            })
            
            time.sleep(2)  # Longer pause to see the response
        
        # Phase 2: Rigorous discussion rounds
        if discussion_rounds > 0:
            log_event('phase', content=f"💬 Phase 2: Expert Discussion ({discussion_rounds} rounds)")
            
            for round_num in range(discussion_rounds):
                log_event('phase', content=f"🔄 Expert Round {round_num + 1}")
                
                for model in available_models:
                    # Log thinking with preserved bubbles
                    log_event('thinking', speaker=self.models[model]['name'])
                    
                    existing_bubbles = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
                    
                    self.update_visual_state({
                        "participants": visual_participant_names,
                        "messages": all_messages,
                        "currentSpeaker": None,
                        "thinking": [self.models[model]['name']],
                        "showBubbles": existing_bubbles,
                        "avatarImages": avatar_images
                    })
                    
                    time.sleep(1)
                    
                    # Build expert position summary
                    position_summary = self.build_position_summary(all_messages, model, topology)
                    
                    role = model_roles[model]
                    role_context = self.roles[role]
                    
                    # PROTOCOL-ADAPTED: Discussion intensity based on protocol
                    if decision_protocol in ['majority_voting', 'ranked_choice']:
                        discussion_style = "DEFEND your position and CHALLENGE weak arguments"
                        discussion_goal = "Prove why your approach is superior"
                    elif decision_protocol == 'consensus':
                        discussion_style = "BUILD on other experts' insights and ADDRESS concerns"
                        discussion_goal = "Work toward a solution everyone can support"
                    else:
                        discussion_style = "REFINE your analysis and RESPOND to other experts"
                        discussion_goal = "Demonstrate the strength of your reasoning"
                    
                    discussion_prompt = f"""🔄 Expert Round {round_num + 1}: {question}

Your Role: {role_context}

{position_summary}

DISCUSSION FOCUS:
- {discussion_style}
- {discussion_goal}
- Address specific points raised by other experts
- Use current data and research if needed
- Maximum 180 words of focused response
- End with "Position: [UNCHANGED/EVOLVED]" and "Confidence: X/10"

Your expert response:"""

                    # Log speaking with preserved bubbles
                    log_event('speaking', speaker=self.models[model]['name'])
                    
                    existing_bubbles = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
                    
                    self.update_visual_state({
                        "participants": visual_participant_names,
                        "messages": all_messages,
                        "currentSpeaker": self.models[model]['name'],
                        "thinking": [],
                        "showBubbles": existing_bubbles,
                        "avatarImages": avatar_images
                    })
                    
                    time.sleep(2)
                    
                    response = self.call_model(model, discussion_prompt)
                    
                    if response:
                        confidence = self._extract_confidence(response)
                        message = {
                            "speaker": self.models[model]['name'],
                            "text": f"Round {round_num + 1}: {response}",
                            "confidence": confidence,
                            "role": model_roles[model]
                        }
                        all_messages.append(message)
                        
                        log_event('message', 
                                 speaker=self.models[model]['name'], 
                                 content=f"Round {round_num + 1}: {response}",
                                 role=model_roles[model],
                                 confidence=confidence)
                    else:
                        # Handle failed API call gracefully
                        log_event('message', 
                                 speaker=self.models[model]['name'], 
                                 content=f"Round {round_num + 1}: Analysis temporarily unavailable - API connection failed",
                                 role=model_roles[model],
                                 confidence=0)
                        
                        message = {
                            "speaker": self.models[model]['name'],
                            "text": f"Round {round_num + 1}: ⚠️ Analysis temporarily unavailable - API connection failed.",
                            "confidence": 0,
                            "role": model_roles[model]
                        }
                        all_messages.append(message)
                    
                    # Update visual state
                    responded_speakers = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
                    
                    self.update_visual_state({
                        "participants": visual_participant_names,
                        "messages": all_messages,
                        "currentSpeaker": None,
                        "thinking": [],
                        "showBubbles": responded_speakers,
                        "avatarImages": avatar_images
                    })
                    
                    time.sleep(1)
        
        # Phase 3: PROTOCOL-SPECIFIC final decision
        if decision_protocol == 'consensus':
            phase_name = "🤝 Phase 3: Building Consensus"
            moderator_title = "Senior Advisor"
        elif decision_protocol in ['majority_voting', 'ranked_choice']:
            phase_name = "⚖️ Phase 3: Final Decision"
            moderator_title = "Lead Analyst"
        else:
            phase_name = "📊 Phase 3: Expert Synthesis"
            moderator_title = "Lead Researcher"
        
        log_event('phase', content=f"{phase_name} - {decision_protocol}")
        log_event('thinking', speaker="All experts", content="Synthesizing final recommendation...")
        
        expert_names = [self.models[model]['name'] for model in available_models]
        
        # Preserve existing bubbles during final thinking
        existing_bubbles = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
        
        self.update_visual_state({
            "participants": visual_participant_names,
            "messages": all_messages,
            "currentSpeaker": None,
            "thinking": expert_names,
            "showBubbles": existing_bubbles,
            "avatarImages": avatar_images
        })
        
        time.sleep(2)
        
        # Generate PROTOCOL-APPROPRIATE final analysis
        moderator = self.moderator_model if self.models[self.moderator_model]['available'] else available_models[0]
        
        # Build expert summary
        final_positions = {}
        confidence_scores = []
        
        for msg in all_messages:
            speaker = msg["speaker"]
            if speaker not in [moderator_title, 'Consilium', 'Research Agent']:
                if speaker not in final_positions:
                    final_positions[speaker] = []
                final_positions[speaker].append(msg)
                if 'confidence' in msg:
                    confidence_scores.append(msg['confidence'])
        
        # Create PROFESSIONAL expert summary
        expert_summary = f"🎯 EXPERT ANALYSIS: {question}\n\nFINAL EXPERT POSITIONS:\n"
        
        for speaker, messages in final_positions.items():
            latest_msg = messages[-1]
            role = latest_msg.get('role', 'standard')
            # Extract the core argument
            core_argument = latest_msg['text'][:200] + "..." if len(latest_msg['text']) > 200 else latest_msg['text']
            confidence = latest_msg.get('confidence', 5)
            
            expert_summary += f"\n📋 **{speaker}** ({role}):\n{core_argument}\nFinal Confidence: {confidence}/10\n"
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 5.0
        
        # PROTOCOL-SPECIFIC synthesis prompt
        if decision_protocol == 'consensus':
            synthesis_goal = "Build a CONSENSUS recommendation that all experts can support"
            synthesis_format = "**CONSENSUS REACHED:** [Yes/Partial/No]\n**RECOMMENDED APPROACH:** [Synthesis]\n**AREAS OF AGREEMENT:** [Common ground]\n**REMAINING CONCERNS:** [Issues to address]"
        elif decision_protocol in ['majority_voting', 'ranked_choice']:
            synthesis_goal = "Determine the STRONGEST position and declare a clear winner"
            synthesis_format = "**DECISION:** [Clear recommendation]\n**WINNING ARGUMENT:** [Most compelling case]\n**KEY EVIDENCE:** [Supporting data]\n**IMPLEMENTATION:** [Next steps]"
        else:
            synthesis_goal = "Synthesize expert insights into actionable recommendations"
            synthesis_format = "**ANALYSIS CONCLUSION:** [Summary]\n**RECOMMENDED APPROACH:** [Best path forward]\n**RISK ASSESSMENT:** [Key considerations]\n**CONFIDENCE LEVEL:** [Overall certainty]"
        
        consensus_prompt = f"""{expert_summary}

📊 SENIOR ANALYSIS REQUIRED:

{synthesis_goal}

SYNTHESIS REQUIREMENTS:
- Analyze the quality and strength of each expert position
- Identify areas where experts align vs disagree  
- Provide a clear, actionable recommendation
- Use additional research if needed to resolve disagreements
- Maximum 300 words of decisive analysis

Average Expert Confidence: {avg_confidence:.1f}/10
Protocol: {decision_protocol}

Format:
{synthesis_format}

Provide your synthesis:"""

        log_event('speaking', speaker=moderator_title, content="Synthesizing expert analysis into final recommendation...")
        
        # Preserve existing bubbles during final speaking
        existing_bubbles = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
        
        self.update_visual_state({
            "participants": visual_participant_names,
            "messages": all_messages,
            "currentSpeaker": "Consilium",
            "thinking": [],
            "showBubbles": existing_bubbles,
            "avatarImages": avatar_images
        })
        
        # Call moderator model - may also trigger function calls
        consensus_result = self.call_model(moderator, consensus_prompt)
        
        if not consensus_result:
            consensus_result = f"""**ANALYSIS INCOMPLETE:** Technical difficulties prevented full synthesis.

**RECOMMENDED APPROACH:** Manual review of expert positions required.

**KEY CONSIDERATIONS:** All expert inputs should be carefully evaluated.

**NEXT STEPS:** Retry analysis or conduct additional expert consultation."""
        
        # Determine result quality based on protocol
        if decision_protocol == 'consensus':
            if "CONSENSUS REACHED: Yes" in consensus_result or avg_confidence >= 7.5:
                visual_summary = "✅ Expert Consensus Achieved"
            elif "Partial" in consensus_result:
                visual_summary = "⚠️ Partial Consensus - Some Expert Disagreement"
            else:
                visual_summary = "🤔 No Consensus - Significant Expert Disagreement"
        elif decision_protocol in ['majority_voting', 'ranked_choice']:
            if any(word in consensus_result.upper() for word in ["DECISION:", "WINNING", "RECOMMEND"]):
                visual_summary = "⚖️ Clear Expert Recommendation"
            else:
                visual_summary = "🤔 Expert Analysis Complete"
        else:
            visual_summary = "📊 Expert Analysis Complete"
        
        final_message = {
            "speaker": moderator_title,
            "text": f"{visual_summary}\n\n{consensus_result}",
            "confidence": avg_confidence,
            "role": "moderator"
        }
        all_messages.append(final_message)
        
        log_event('message', 
                 speaker=moderator_title, 
                 content=consensus_result,
                 confidence=avg_confidence)
        
        responded_speakers = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker") and msg["speaker"] != "Research Agent"))
        
        self.update_visual_state({
            "participants": visual_participant_names,
            "messages": all_messages,
            "currentSpeaker": None,
            "thinking": [],
            "showBubbles": responded_speakers,
            "avatarImages": avatar_images
        })
        
        log_event('phase', content="✅ Expert Analysis Complete")
        
        return consensus_result

def update_session_roundtable_state(session_id: str, new_state: Dict):
    """Update roundtable state for specific session"""
    session = get_or_create_session_state(session_id)
    session["roundtable_state"].update(new_state)
    return json.dumps(session["roundtable_state"])

def run_consensus_discussion_session(question: str, discussion_rounds: int = 3, 
                                   decision_protocol: str = "consensus", role_assignment: str = "balanced",
                                   topology: str = "full_mesh", moderator_model: str = "mistral",
                                   session_id_state: str = None,
                                   request: gr.Request = None):
    """Session-isolated expert consensus discussion"""
    
    # Get unique session
    session_id = get_session_id(request) if not session_id_state else session_id_state
    session = get_or_create_session_state(session_id)
    
    # Reset session state for new discussion
    session["discussion_log"] = []
    session["final_answer"] = ""
    
    def session_visual_update_callback(state_update):
        """Session-specific visual update callback"""
        update_session_roundtable_state(session_id, state_update)
    
    def session_log_event(event_type: str, speaker: str = "", content: str = "", **kwargs):
        """Add event to THIS session's log only"""
        session["discussion_log"].append({
            'type': event_type,
            'speaker': speaker,
            'content': content,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            **kwargs
        })
    
    # Create engine with session-specific callback
    engine = VisualConsensusEngine(moderator_model, session_visual_update_callback, session_id)
    
    # Run consensus with session-specific logging
    result = engine.run_visual_consensus_session(
        question, discussion_rounds, decision_protocol, 
        role_assignment, topology, moderator_model, 
        session_log_event
    )
    
    # Generate session-specific final answer
    available_models = [model for model, info in engine.models.items() if info['available']]
    session["final_answer"] = f"""## 🎯 Expert Analysis Results

{result}

---

### 📊 Analysis Summary
- **Question:** {question}
- **Protocol:** {decision_protocol.replace('_', ' ').title()}
- **Topology:** {topology.replace('_', ' ').title()}
- **Experts:** {len(available_models)} AI specialists
- **Roles:** {role_assignment.title()}
- **Research Integration:** Native function calling with live data
- **Session ID:** {session_id[:3]}...

*Generated by Consilium: Multi-AI Expert Consensus Platform*"""
    
    # Format session-specific discussion log
    formatted_log = format_session_discussion_log(session["discussion_log"])
    
    return ("✅ Expert Analysis Complete - See results below", 
            json.dumps(session["roundtable_state"]), 
            session["final_answer"], 
            formatted_log,
            session_id)

def format_session_discussion_log(discussion_log: list) -> str:
    """Format discussion log for specific session"""
    if not discussion_log:
        return "No discussion log available yet."
    
    formatted_log = "# 🎭 Complete Expert Discussion Log\n\n"
    
    for entry in discussion_log:
        timestamp = entry.get('timestamp', datetime.now().strftime('%H:%M:%S'))
        
        if entry['type'] == 'thinking':
            formatted_log += f"**{timestamp}** 🤔 **{entry['speaker']}** is analyzing...\n\n"
            
        elif entry['type'] == 'speaking':
            formatted_log += f"**{timestamp}** 💬 **{entry['speaker']}** is presenting...\n\n"
            
        elif entry['type'] == 'message':
            formatted_log += f"**{timestamp}** 📋 **{entry['speaker']}** ({entry.get('role', 'standard')}):\n"
            formatted_log += f"> {entry['content']}\n"
            if 'confidence' in entry:
                formatted_log += f"*Confidence: {entry['confidence']}/10*\n\n"
            else:
                formatted_log += "\n"
                
        elif entry['type'] == 'research_request':
            function_name = entry.get('function', 'Unknown')
            query = entry.get('query', 'Unknown query')
            requesting_expert = entry.get('requesting_expert', 'Unknown expert')
            formatted_log += f"**{timestamp}** 🔍 **Research Agent** - Research Request:\n"
            formatted_log += f"> **Function:** {function_name.replace('_', ' ').title()}\n"
            formatted_log += f"> **Query:** \"{query}\"\n"
            formatted_log += f"> **Requested by:** {requesting_expert}\n\n"
            
        elif entry['type'] == 'research_result':
            function_name = entry.get('function', 'Unknown')
            query = entry.get('query', 'Unknown query')
            requesting_expert = entry.get('requesting_expert', 'Unknown expert')
            full_result = entry.get('full_result', entry.get('content', 'No result'))
            formatted_log += f"**{timestamp}** 📊 **Research Agent** - Research Results:\n"
            formatted_log += f"> **Function:** {function_name.replace('_', ' ').title()}\n"
            formatted_log += f"> **Query:** \"{query}\"\n"
            formatted_log += f"> **For Expert:** {requesting_expert}\n\n"
            formatted_log += f"**Research Results:**\n"
            formatted_log += f"```\n{full_result}\n```\n\n"
            
        elif entry['type'] == 'phase':
            formatted_log += f"\n---\n## {entry['content']}\n---\n\n"
    
    return formatted_log

def check_model_status_session(session_id_state: str = None, request: gr.Request = None):
    """Check and display current model availability for specific session"""
    session_id = get_session_id(request) if not session_id_state else session_id_state
    session = get_or_create_session_state(session_id)
    session_keys = session.get("api_keys", {})
    
    # Get session-specific keys or fall back to env vars
    mistral_key = session_keys.get("mistral") or MISTRAL_API_KEY
    sambanova_key = session_keys.get("sambanova") or SAMBANOVA_API_KEY
    
    status_info = "## 🔍 Expert Model Availability\n\n"
    
    models = {
        'Mistral Large': mistral_key,
        'DeepSeek-R1': sambanova_key,
        'Meta-Llama-3.3-70B-Instruct': sambanova_key,
        'QwQ-32B': sambanova_key,
        'Research Agent': True
    }
    
    for model_name, available in models.items():
        if model_name == 'Research Agent':
            status = "✅ Available (Built-in + Native Function Calling)"
        else:
            if available:
                status = f"✅ Available (Key: {available[:3]}...)"
            else:
                status = "❌ Not configured"
        status_info += f"**{model_name}:** {status}\n\n"
    
    return status_info

# Create the professional interface
with gr.Blocks(title="🎭 Consilium: Multi-AI Expert Consensus Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 Consilium: Multi-AI Expert Consensus Platform

    **Watch expert AI models collaborate with live research to solve your most complex decisions**
    
    This MCP server was built for the Gradio Agents and MCP Hackathon 2025. Additionally, I built a custom Gradio component for the roundtable (https://huggingface.co/spaces/azettl/gradio_consilium_roundtable).

    📼 Video UI: https://youtu.be/ciYLqI-Nawc 
    📼 Video MCP: https://youtu.be/r92vFUXNg74
    
    ## Features:
    
    * Visual roundtable of the AI models, including speech bubbles to see the discussion in real time.
    * MCP mode enabled to also use it directly in, for example, Claude Desktop (without the visual table).
    * Includes Mistral (**mistral-large-latest**) via their API and the Models **DeepSeek-R1**, **Meta-Llama-3.3-70B-Instruct** and **QwQ-32B** via the SambaNova API.
    * Research Agent with 6 sources (**Web Search**, **Wikipedia**, **arXiv**, **GitHub**, **SEC EDGAR**, **Google Scholar**) for comprehensive live research.
    * Assign different roles to the models, the protocol they should follow, and decide the communication strategy.
    * Pick one model as the lead analyst (had the best results when picking Mistral).
    * Configure the amount of discussion rounds.
    * After the discussion, the whole conversation and a final answer will be presented.
    """)
    
    # Hidden session state component
    session_state = gr.State()
    
    with gr.Tab("🎭 Expert Consensus Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="🎯 Strategic Decision Question",
                    placeholder="What complex decision would you like expert AI analysis on?",
                    lines=3,
                    value="Should our startup pivot to AI-first product development?"
                )
                
                # Professional question suggestion buttons
                with gr.Row():
                    suggestion_btn1 = gr.Button("🏢 Business Strategy", size="sm")
                    suggestion_btn2 = gr.Button("⚛️ Technology Choice", size="sm") 
                    suggestion_btn3 = gr.Button("🌍 Policy Analysis", size="sm")
                
                with gr.Row():
                    decision_protocol = gr.Dropdown(
                        choices=["consensus", "majority_voting", "weighted_voting", "ranked_choice", "unanimity"],
                        value="consensus",
                        label="⚖️ Decision Protocol",
                        info="How should experts reach a conclusion?"
                    )
                    
                    role_assignment = gr.Dropdown(
                        choices=["balanced", "specialized", "adversarial", "none"],
                        value="balanced",
                        label="🎓 Expert Roles",
                        info="How should expertise be distributed?"
                    )
                
                with gr.Row():
                    topology = gr.Dropdown(
                        choices=["full_mesh", "star", "ring"],
                        value="full_mesh",
                        label="🌐 Communication Structure",
                        info="Full mesh: all collaborate, Star: through moderator, Ring: sequential"
                    )
                    
                    moderator_model = gr.Dropdown(
                        choices=["mistral", "sambanova_deepseek", "sambanova_llama", "sambanova_qwq"],
                        value="mistral",
                        label="👨‍⚖️ Lead Analyst",
                        info="Mistral works best as Lead"
                    )
                
                rounds_input = gr.Slider(
                    minimum=1, maximum=5, value=2, step=1,
                    label="🔄 Discussion Rounds",
                    info="More rounds = deeper analysis"
                )
                
                start_btn = gr.Button("🚀 Start Expert Analysis", variant="primary", size="lg")
                
                status_output = gr.Textbox(label="📊 Analysis Status", interactive=False)
            
            with gr.Column(scale=2):
                # The visual roundtable component
                roundtable = consilium_roundtable(
                    label="AI Expert Roundtable",
                    label_icon="https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
                    value=json.dumps({
                        "participants": [],
                        "messages": [],
                        "currentSpeaker": None,
                        "thinking": [],
                        "showBubbles": [],
                        "avatarImages": avatar_images
                    })
                )
        
        # Final answer section
        with gr.Row():
            final_answer_output = gr.Markdown(
                label="🎯 Expert Analysis Results",
                value="*Expert analysis results will appear here...*"
            )
        
        # Collapsible discussion log
        with gr.Accordion("📋 Complete Expert Discussion Log", open=False):
            discussion_log_output = gr.Markdown(
                value="*Complete expert discussion transcript will appear here...*"
            )
        
        # Professional question handlers
        def set_business_question():
            return "Should our startup pivot to AI-first product development?"
        
        def set_tech_question():
            return "Microservices vs monolith architecture for our scaling platform?"
        
        def set_policy_question():
            return "Should we prioritize geoengineering research over emissions reduction?"
        
        suggestion_btn1.click(set_business_question, outputs=[question_input])
        suggestion_btn2.click(set_tech_question, outputs=[question_input])
        suggestion_btn3.click(set_policy_question, outputs=[question_input])
        
        # Event handlers
        def on_start_discussion(question, rounds, protocol, roles, topology, moderator, session_id_state, request: gr.Request = None):
            # Start discussion immediately
            result = run_consensus_discussion_session(question, rounds, protocol, roles, topology, moderator, session_id_state, request)
            return result
        
        start_btn.click(
            on_start_discussion,
            inputs=[question_input, rounds_input, decision_protocol, role_assignment, topology, moderator_model, session_state],
            outputs=[status_output, roundtable, final_answer_output, discussion_log_output, session_state]
        )
        
        # Auto-refresh the roundtable state every 1 second during discussion for better visibility
        def refresh_roundtable(session_id_state, request: gr.Request = None):
            session_id = get_session_id(request) if not session_id_state else session_id_state
            if session_id in user_sessions:
                return json.dumps(user_sessions[session_id]["roundtable_state"])
            return json.dumps({
                "participants": [],
                "messages": [],
                "currentSpeaker": None,
                "thinking": [],
                "showBubbles": [],
                "avatarImages": avatar_images
            })
        
        gr.Timer(1.0).tick(refresh_roundtable, inputs=[session_state], outputs=[roundtable])
    
    with gr.Tab("🔧 Configuration & Setup"):
        gr.Markdown("## 🔑 API Keys Configuration")
        gr.Markdown("*Enter your API keys below OR set them as environment variables*")
        gr.Markdown("**🔒 Privacy:** Your API keys are stored only for your session and are not shared with other users.")
        
        with gr.Row():
            with gr.Column():
                mistral_key_input = gr.Textbox(
                    label="Mistral API Key",
                    placeholder="Enter your Mistral API key...",
                    type="password",
                    info="Required for Mistral Large expert model with function calling"
                )
                sambanova_key_input = gr.Textbox(
                    label="SambaNova API Key", 
                    placeholder="Enter your SambaNova API key...",
                    type="password",
                    info="Required for DeepSeek, Llama, and QwQ expert models with function calling"
                )
                
            with gr.Column():
                # Add a button to save/update keys
                save_keys_btn = gr.Button("💾 Save API Keys", variant="secondary")
                keys_status = gr.Textbox(
                    label="Keys Status",
                    value="No API keys configured - using environment variables if available",
                    interactive=False
                )
        
        # Connect the save button
        save_keys_btn.click(
            update_session_api_keys,
            inputs=[mistral_key_input, sambanova_key_input, session_state],
            outputs=[keys_status, session_state]
        )
        
        model_status_display = gr.Markdown(check_model_status_session())
        
        # Add refresh button for model status
        refresh_status_btn = gr.Button("🔄 Refresh Expert Status")
        refresh_status_btn.click(
            check_model_status_session,
            inputs=[session_state],
            outputs=[model_status_display]
        )
        
        gr.Markdown("""
        ## 🛠️ Setup Instructions
        
        ### 🚀 Quick Start (Recommended)
        1. **Enter API keys above** (they'll be used only for your session)
        2. **Click "Save API Keys"** 
        3. **Start an expert analysis with live research!**
        
        ### 🔑 Get API Keys:
        - **Mistral:** [console.mistral.ai](https://console.mistral.ai)
        - **SambaNova:** [cloud.sambanova.ai](https://cloud.sambanova.ai)
        
        ## Local Setups
        
        ### 🌐 Environment Variables
        ```bash
        export MISTRAL_API_KEY=your_key_here
        export SAMBANOVA_API_KEY=your_key_here
        export MODERATOR_MODEL=mistral
        ```
        
        ### 📋 Dependencies
        ```bash
        pip install -r requirements.txt
        ```
        ### Start
        ```bash
        python app.py
        ```
        
        ### 🔗 MCP Integration
        Add to your Claude Desktop config:
        ```json
        {
          "mcpServers": {
            "consilium": {
              "command": "npx",
              "args": ["mcp-remote", "http://localhost:7860/gradio_api/mcp/sse"]
            }
          }
        }
        ```
        """)
    
    with gr.Tab("📚 Documentation"):
        gr.Markdown("""
        ## 🔬 **Research Capabilities**

        ### **🌐 Multi-Source Research**
        - **DuckDuckGo Web Search**: Current events, news, real-time information
        - **Wikipedia**: Authoritative background and encyclopedic data  
        - **arXiv**: Academic papers and scientific research preprints
        - **Google Scholar**: Peer-reviewed research and citation analysis
        - **GitHub**: Technology trends, adoption patterns, developer activity
        - **SEC EDGAR**: Public company financial data and regulatory filings

        ### **🎯 Smart Research Routing**
        The system automatically routes queries to the most appropriate sources:
        - **Academic queries** → arXiv + Google Scholar
        - **Technology questions** → GitHub + Web Search
        - **Company research** → SEC filings + Web Search  
        - **Current events** → Web Search + Wikipedia
        - **Deep research** → Multi-source synthesis with quality scoring

        ### **📊 Research Quality Scoring**
        Each research result is scored on:
        - **Recency** (0-1): How current is the information
        - **Authority** (0-1): Source credibility and reliability
        - **Specificity** (0-1): Quantitative data and specific details
        - **Relevance** (0-1): How well it matches the query
        """)
        gr.Markdown("""
        ## 🎓 **Expert Role Assignments**
        
        #### **⚖️ Balanced (Recommended for Most Decisions)**
        - **Expert Advocate**: Passionate defender with compelling evidence
        - **Critical Analyst**: Rigorous critic identifying flaws and risks
        - **Strategic Advisor**: Practical implementer focused on real-world constraints
        - **Research Specialist**: Authoritative knowledge with evidence-based insights
        
        #### **🎯 Specialized (For Technical Decisions)**
        - **Research Specialist**: Deep domain expertise and authoritative analysis
        - **Strategic Advisor**: Implementation-focused practical guidance
        - **Innovation Catalyst**: Breakthrough approaches and unconventional thinking
        - **Expert Advocate**: Passionate championing of specialized viewpoints
        
        #### **⚔️ Adversarial (For Controversial Topics)**
        - **Critical Analyst**: Aggressive identification of weaknesses
        - **Innovation Catalyst**: Deliberately challenging conventional wisdom
        - **Expert Advocate**: Passionate defense of positions
        - **Strategic Advisor**: Hard-nosed practical constraints
        
        ## ⚖️ **Decision Protocols Explained**
        
        ### 🤝 **Consensus** (Collaborative)
        - **Goal**: Find solutions everyone can support
        - **Style**: Respectful but rigorous dialogue
        - **Best for**: Team decisions, long-term strategy
        - **Output**: "Expert Consensus Achieved" or areas of disagreement
        
        ### 🗳️ **Majority Voting** (Competitive)
        - **Goal**: Let the strongest argument win
        - **Style**: Passionate advocacy and strong positions
        - **Best for**: Clear either/or decisions
        - **Output**: "Clear Expert Recommendation" with winning argument
        
        ### 📊 **Weighted Voting** (Expertise-Based)
        - **Goal**: Let expertise and evidence quality determine influence
        - **Style**: Authoritative analysis with detailed reasoning
        - **Best for**: Technical decisions requiring deep knowledge
        - **Output**: Expert synthesis weighted by confidence levels
        
        ### 🏆 **Ranked Choice** (Comprehensive)
        - **Goal**: Explore all options systematically
        - **Style**: Systematic evaluation of alternatives
        - **Best for**: Complex decisions with multiple options
        - **Output**: Ranked recommendations with detailed analysis
        
        ### 🔒 **Unanimity** (Diplomatic)
        - **Goal**: Achieve complete agreement
        - **Style**: Bridge-building and diplomatic dialogue
        - **Best for**: High-stakes decisions requiring buy-in
        - **Output**: Unanimous agreement or identification of blocking issues
        
        ## 🌐 **Communication Structures**
        
        ### 🕸️ **Full Mesh** (Complete Collaboration)
        - Every expert sees all other expert responses
        - Maximum information sharing and cross-pollination
        - Best for comprehensive analysis and complex decisions
        - **Use when:** You want thorough multi-perspective analysis
        
        ### ⭐ **Star** (Hierarchical Analysis)
        - Experts only see the lead analyst's responses
        - Prevents groupthink, maintains independent thinking
        - Good for getting diverse, uninfluenced perspectives
        - **Use when:** You want fresh, independent expert takes
        
        ### 🔄 **Ring** (Sequential Analysis)
        - Each expert only sees the previous expert's response
        - Creates interesting chains of reasoning and idea evolution
        - Can lead to surprising consensus emergence
        - **Use when:** You want to see how ideas build and evolve
        """)

# Launch configuration
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10) 
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        mcp_server=True
    )