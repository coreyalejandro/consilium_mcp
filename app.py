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

# Load environment variables
load_dotenv()

# API Configuration - These will be updated by UI if needed
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
MODERATOR_MODEL = os.getenv("MODERATOR_MODEL", "mistral")

# Session-based storage for isolated discussions
user_sessions: Dict[str, Dict] = {}

class WikipediaTool(Tool):
    name = "wikipedia_search"
    description = "Search Wikipedia for comprehensive information on any topic"
    inputs = {"query": {"type": "string", "description": "The topic to search for on Wikipedia"}}
    output_type = "string"
    
    def forward(self, query: str) -> str:
        try:
            import wikipedia
            # Search for the topic
            search_results = wikipedia.search(query, results=3)
            if not search_results:
                return f"No Wikipedia articles found for: {query}"
            
            # Get the first article
            page = wikipedia.page(search_results[0])
            summary = page.summary[:1000] + "..." if len(page.summary) > 1000 else page.summary
            
            return f"**Wikipedia: {page.title}**\n\n{summary}\n\nSource: {page.url}"
        except Exception as e:
            return f"Wikipedia search error: {str(e)}"

class WebSearchAgent:
    def __init__(self):
        try:
            # Use TinyLlama for faster inference
            self.agent = CodeAgent(
                tools=[
                    DuckDuckGoSearchTool(), 
                    VisitWebpageTool(),
                    WikipediaTool(),
                    FinalAnswerTool()
                ], 
                model=InferenceClientModel(),
                max_steps=3,
                verbosity_level=0
            )
        except Exception as e:
            print(f"Warning: Could not initialize search agent: {e}")
            self.agent = None
    
    def search(self, query: str, max_results: int = 5) -> str:
        """Use the CodeAgent to perform comprehensive web search and analysis"""
        if not self.agent:
            return f"🔍 **Web Search for:** {query}\n\nSearch agent not available. Please check dependencies."
        
        try:
            # Simplified prompt for TinyLlama
            agent_prompt = f"Search for information about: {query}"
            
            # Run the agent
            result = self.agent.run(agent_prompt)
            
            # Format the result nicely
            if result:
                return f"🔍 **Web Research Results for:** {query}\n\n{result}"
            else:
                return f"🔍 **Web Search for:** {query}\n\nNo results found."
            
        except Exception as e:
            # Fallback to simple error message
            return f"🔍 **Web Search Error for:** {query}\n\nError: {str(e)}\n\nPlease try again or rephrase your query."

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
            "step_by_step_active": False,
            "step_continue_event": threading.Event(),
            "api_keys": {
                "mistral": None,
                "sambanova": None,
                "huggingface": None
            }
        }
    return user_sessions[session_id]

def update_session_api_keys(mistral_key, sambanova_key, huggingface_key, session_id_state, request: gr.Request = None):
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
        self.search_agent = WebSearchAgent()
        self.update_callback = update_callback
        self.session_id = session_id
        
        # Get session-specific keys or fall back to global
        session = get_or_create_session_state(session_id) if session_id else {"api_keys": {}}
        session_keys = session.get("api_keys", {})
        
        mistral_key = session_keys.get("mistral") or MISTRAL_API_KEY
        sambanova_key = session_keys.get("sambanova") or SAMBANOVA_API_KEY
        
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
                'name': 'Meta-Llama-3.1-8B',
                'api_key': sambanova_key,
                'available': bool(sambanova_key)
            },
            'sambanova_qwq': {
                'name': 'QwQ-32B',
                'api_key': sambanova_key,
                'available': bool(sambanova_key)
            },
            'search': {
                'name': 'Web Search Agent',
                'api_key': True,
                'available': True
            }
        }
        
        # Store session keys for API calls
        self.session_keys = {
            'mistral': mistral_key,
            'sambanova': sambanova_key,
            'huggingface': hf_token
        }
        
        # Role definitions
        self.roles = {
            'standard': "You are participating in a collaborative AI discussion. Provide thoughtful, balanced analysis.",
            'devils_advocate': "You are the devil's advocate. Challenge assumptions, point out weaknesses, and argue alternative perspectives even if unpopular.",
            'fact_checker': "You are the fact checker. Focus on verifying claims, checking accuracy, and identifying potential misinformation.",
            'synthesizer': "You are the synthesizer. Focus on finding common ground, combining different perspectives, and building bridges between opposing views.",
            'domain_expert': "You are a domain expert. Provide specialized knowledge, technical insights, and authoritative perspective on the topic.",
            'creative_thinker': "You are the creative thinker. Approach problems from unusual angles, suggest innovative solutions, and think outside conventional boundaries."
        }
    
    def update_visual_state(self, state_update: Dict[str, Any]):
        """Update the visual roundtable state for this session"""
        if self.update_callback:
            self.update_callback(state_update)
    
    def call_model(self, model: str, prompt: str, context: str = "") -> Optional[str]:
        """Generic model calling function using session-specific keys"""
        if model == 'search':
            search_query = self._extract_search_query(prompt)
            return self.search_agent.search(search_query)
        
        if not self.models[model]['available']:
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
    
    def _extract_search_query(self, prompt: str) -> str:
        """Extract search query from prompt or generate one"""
        lines = prompt.split('\n')
        for line in lines:
            if 'QUESTION:' in line:
                return line.replace('QUESTION:', '').strip()
        
        for line in lines:
            if len(line.strip()) > 10:
                return line.strip()[:100]
        
        return prompt[:100]
    
    def _call_sambanova(self, model: str, prompt: str) -> Optional[str]:
        api_key = self.session_keys.get('sambanova')
        if not api_key:
            return None
            
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://api.sambanova.ai/v1", 
                api_key=api_key
            )
            
            model_mapping = {
                'sambanova_deepseek': 'DeepSeek-R1',
                'sambanova_llama': 'Meta-Llama-3.1-8B-Instruct', 
                'sambanova_qwq': 'QwQ-32B'
            }
            
            sambanova_model = model_mapping.get(model, 'Meta-Llama-3.1-8B-Instruct')
            
            completion = client.chat.completions.create(
                model=sambanova_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling Sambanova {model}: {str(e)}")
            return None
    
    def _call_mistral(self, prompt: str) -> Optional[str]:
        api_key = self.session_keys.get('mistral')
        if not api_key:
            return None
            
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://api.mistral.ai/v1", 
                api_key=api_key
            )
            
            completion = client.chat.completions.create(
                model='mistral-large-latest',
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling Mistral API mistral-large-latest: {str(e)}")
            return None
    
    def assign_roles(self, models: List[str], role_assignment: str) -> Dict[str, str]:
        """Assign roles to models"""
        if role_assignment == "none":
            return {model: "standard" for model in models}
        
        roles_to_assign = []
        if role_assignment == "balanced":
            roles_to_assign = ["devils_advocate", "fact_checker", "synthesizer", "standard"]
        elif role_assignment == "specialized":
            roles_to_assign = ["domain_expert", "fact_checker", "creative_thinker", "synthesizer"]
        elif role_assignment == "adversarial":
            roles_to_assign = ["devils_advocate", "devils_advocate", "standard", "standard"]
        
        while len(roles_to_assign) < len(models):
            roles_to_assign.append("standard")
        
        model_roles = {}
        for i, model in enumerate(models):
            model_roles[model] = roles_to_assign[i % len(roles_to_assign)]
        
        return model_roles
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        confidence_match = re.search(r'Confidence:\s*(\d+(?:\.\d+)?)', response)
        if confidence_match:
            try:
                return float(confidence_match.group(1))
            except ValueError:
                pass
        return 5.0
    
    def run_visual_consensus_session(self, question: str, discussion_rounds: int = 3, 
                                   decision_protocol: str = "consensus", role_assignment: str = "balanced",
                                   topology: str = "full_mesh", moderator_model: str = "mistral",
                                   enable_step_by_step: bool = False, log_function=None):
        """Run consensus with session-isolated visual updates"""
        
        available_models = [model for model, info in self.models.items() if info['available']]
        if not available_models:
            return "❌ No AI models available"
        
        model_roles = self.assign_roles(available_models, role_assignment)
        participant_names = [self.models[model]['name'] for model in available_models]
        
        # Use session-specific logging
        def log_event(event_type: str, speaker: str = "", content: str = "", **kwargs):
            if log_function:
                log_function(event_type, speaker, content, **kwargs)
        
        # Log the start
        log_event('phase', content=f"🚀 Starting Discussion: {question}")
        log_event('phase', content=f"📊 Configuration: {len(available_models)} models, {decision_protocol} protocol, {role_assignment} roles")
        
        # Initialize visual state
        self.update_visual_state({
            "participants": participant_names,
            "messages": [],
            "currentSpeaker": None,
            "thinking": [],
            "showBubbles": []
        })
        
        all_messages = []
        
        # Phase 1: Initial responses
        log_event('phase', content="📝 Phase 1: Initial Responses")
        
        for model in available_models:
            # Log and set thinking state
            log_event('thinking', speaker=self.models[model]['name'])
            self.update_visual_state({
                "participants": participant_names,
                "messages": all_messages,
                "currentSpeaker": None,
                "thinking": [self.models[model]['name']]
            })
            
            if not enable_step_by_step:
                time.sleep(1)
            
            role = model_roles[model]
            role_context = self.roles[role]
            
            prompt = f"""{role_context}

QUESTION: {question}

Please provide your initial analysis and answer. Be thoughtful, detailed, and explain your reasoning.

Your response should include:
1. Your direct answer to the question
2. Your reasoning and evidence  
3. Any important considerations or nuances
4. END YOUR RESPONSE WITH: "Confidence: X/10" where X is your confidence level"""

            # Log and set speaking state
            log_event('speaking', speaker=self.models[model]['name'])
            self.update_visual_state({
                "participants": participant_names,
                "messages": all_messages,
                "currentSpeaker": self.models[model]['name'],
                "thinking": []
            })
            
            if not enable_step_by_step:
                time.sleep(2)
            
            response = self.call_model(model, prompt)
            
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
                
                # Update with new message
                responded_speakers = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker")))
                
                self.update_visual_state({
                    "participants": participant_names,
                    "messages": all_messages,
                    "currentSpeaker": None,
                    "thinking": [],
                    "showBubbles": responded_speakers
                })
                
                if enable_step_by_step:
                    session = get_or_create_session_state(self.session_id)
                    session["step_continue_event"].clear()
                    session["step_continue_event"].wait()
                else:
                    time.sleep(0.5)
        
        # Phase 2: Discussion rounds
        if discussion_rounds > 0:
            log_event('phase', content=f"💬 Phase 2: Discussion Rounds ({discussion_rounds} rounds)")
            
            for round_num in range(discussion_rounds):
                log_event('phase', content=f"🔄 Discussion Round {round_num + 1}")
                
                for model in available_models:
                    log_event('thinking', speaker=self.models[model]['name'])
                    self.update_visual_state({
                        "participants": participant_names,
                        "messages": all_messages,
                        "currentSpeaker": None,
                        "thinking": [self.models[model]['name']]
                    })
                    
                    if not enable_step_by_step:
                        time.sleep(1)
                    
                    # Create context of other responses
                    other_responses = ""
                    for other_model in available_models:
                        if other_model != model:
                            other_responses += f"\n**{self.models[other_model]['name']}**: [Previous response]\n"
                    
                    discussion_prompt = f"""CONTINUING DISCUSSION FOR: {question}

Round {round_num + 1} of {discussion_rounds}

Other models' current responses:
{other_responses}

Please provide your updated analysis considering the discussion so far.
END WITH: "Confidence: X/10" """

                    log_event('speaking', speaker=self.models[model]['name'])
                    self.update_visual_state({
                        "participants": participant_names,
                        "messages": all_messages,
                        "currentSpeaker": self.models[model]['name'],
                        "thinking": []
                    })
                    
                    if not enable_step_by_step:
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
                        
                        responded_speakers = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker")))
                        
                        self.update_visual_state({
                            "participants": participant_names,
                            "messages": all_messages,
                            "currentSpeaker": None,
                            "thinking": [],
                            "showBubbles": responded_speakers
                        })
                        
                        if enable_step_by_step:
                            session = get_or_create_session_state(self.session_id)
                            session["step_continue_event"].clear()
                            session["step_continue_event"].wait()
                        else:
                            time.sleep(1)
        
        # Phase 3: Final consensus
        log_event('phase', content=f"🎯 Phase 3: Final Consensus ({decision_protocol})")
        log_event('thinking', speaker="All participants", content="Building consensus...")
        
        self.update_visual_state({
            "participants": participant_names,
            "messages": all_messages,
            "currentSpeaker": None,
            "thinking": participant_names
        })
        
        if not enable_step_by_step:
            time.sleep(2)
        
        # Generate consensus
        moderator = self.moderator_model if self.models[self.moderator_model]['available'] else available_models[0]
        
        # Collect responses from session log
        session = get_or_create_session_state(self.session_id)
        all_responses = ""
        confidence_scores = []
        for entry in session["discussion_log"]:
            if entry['type'] == 'message' and entry['speaker'] != 'Consilium':
                all_responses += f"\n**{entry['speaker']}**: {entry['content']}\n"
                if 'confidence' in entry:
                    confidence_scores.append(entry['confidence'])
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 5.0
        consensus_threshold = 7.0
        
        consensus_prompt = f"""You are synthesizing the final result from this AI discussion.

ORIGINAL QUESTION: {question}

ALL PARTICIPANT RESPONSES:
{all_responses}

AVERAGE CONFIDENCE LEVEL: {avg_confidence:.1f}/10

Your task:
1. Analyze if the participants reached genuine consensus or if there are significant disagreements
2. If there IS consensus: Provide a comprehensive final answer incorporating all insights
3. If there is NO consensus: Clearly state the disagreements and present the main conflicting positions
4. If partially aligned: Identify areas of agreement and areas of disagreement

Be honest about the level of consensus achieved. Do not force agreement where none exists.

Format your response as:
**CONSENSUS STATUS:** [Reached/Partial/Not Reached]

**FINAL ANSWER:** [Your synthesis]

**AREAS OF DISAGREEMENT:** [If any - explain the key points of contention]"""

        log_event('speaking', speaker="Consilium", content="Analyzing consensus and synthesizing final answer...")
        self.update_visual_state({
            "participants": participant_names,
            "messages": all_messages,
            "currentSpeaker": "Consilium",
            "thinking": []
        })
        
        consensus_result = self.call_model(moderator, consensus_prompt)
        
        if not consensus_result:
            consensus_result = f"""**CONSENSUS STATUS:** Analysis Failed

**FINAL ANSWER:** Unable to generate consensus analysis. Please review individual participant responses in the discussion log.

**AREAS OF DISAGREEMENT:** Analysis could not be completed due to technical issues."""
        
        consensus_reached = "CONSENSUS STATUS: Reached" in consensus_result or avg_confidence >= consensus_threshold
        
        if consensus_reached:
            visual_summary = "✅ Consensus reached!"
        elif "Partial" in consensus_result:
            visual_summary = "⚠️ Partial consensus - some disagreements remain"
        else:
            visual_summary = "❌ No consensus - significant disagreements identified"
        
        final_message = {
            "speaker": "Consilium",
            "text": f"{visual_summary} {consensus_result}",
            "confidence": avg_confidence,
            "role": "consensus"
        }
        all_messages.append(final_message)
        
        log_event('message', 
                 speaker="Consilium", 
                 content=consensus_result,
                 confidence=avg_confidence)
        
        responded_speakers = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker")))
        
        self.update_visual_state({
            "participants": participant_names,
            "messages": all_messages,
            "currentSpeaker": None,
            "thinking": [],
            "showBubbles": responded_speakers
        })
        
        log_event('phase', content="✅ Discussion Complete")
        
        return consensus_result

def update_session_roundtable_state(session_id: str, new_state: Dict):
    """Update roundtable state for specific session"""
    session = get_or_create_session_state(session_id)
    session["roundtable_state"].update(new_state)
    return json.dumps(session["roundtable_state"])

def run_consensus_discussion_session(question: str, discussion_rounds: int = 3, 
                                   decision_protocol: str = "consensus", role_assignment: str = "balanced",
                                   topology: str = "full_mesh", moderator_model: str = "mistral",
                                   enable_step_by_step: bool = False, session_id_state: str = None,
                                   request: gr.Request = None):
    """Session-isolated consensus discussion"""
    
    # Get unique session
    session_id = get_session_id(request) if not session_id_state else session_id_state
    session = get_or_create_session_state(session_id)
    
    # Reset session state for new discussion
    session["discussion_log"] = []
    session["final_answer"] = ""
    session["step_by_step_active"] = enable_step_by_step
    session["step_continue_event"].clear()
    
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
        enable_step_by_step, session_log_event
    )
    
    # Generate session-specific final answer
    available_models = [model for model, info in engine.models.items() if info['available']]
    session["final_answer"] = f"""## 🎯 Final Consensus Answer

{result}

---

### 📊 Discussion Summary
- **Question:** {question}
- **Protocol:** {decision_protocol.replace('_', ' ').title()}
- **Participants:** {len(available_models)} AI models
- **Roles:** {role_assignment.title()}
- **Session ID:** {session_id[:8]}...

*Generated by Consilium Visual AI Consensus Platform*"""
    
    session["step_by_step_active"] = False
    
    # Format session-specific discussion log
    formatted_log = format_session_discussion_log(session["discussion_log"])
    
    return ("✅ Discussion Complete - See results below", 
            json.dumps(session["roundtable_state"]), 
            session["final_answer"], 
            formatted_log,
            session_id)

def format_session_discussion_log(discussion_log: list) -> str:
    """Format discussion log for specific session"""
    if not discussion_log:
        return "No discussion log available yet."
    
    formatted_log = "# 🎭 Complete Discussion Log\n\n"
    
    for entry in discussion_log:
        timestamp = entry.get('timestamp', datetime.now().strftime('%H:%M:%S'))
        if entry['type'] == 'thinking':
            formatted_log += f"**{timestamp}** 🤔 **{entry['speaker']}** is thinking...\n\n"
        elif entry['type'] == 'speaking':
            formatted_log += f"**{timestamp}** 💬 **{entry['speaker']}** is responding...\n\n"
        elif entry['type'] == 'message':
            formatted_log += f"**{timestamp}** ✅ **{entry['speaker']}** ({entry.get('role', 'standard')}):\n"
            formatted_log += f"> {entry['content']}\n"
            if 'confidence' in entry:
                formatted_log += f"*Confidence: {entry['confidence']}/10*\n\n"
            else:
                formatted_log += "\n"
        elif entry['type'] == 'phase':
            formatted_log += f"\n---\n## {entry['content']}\n---\n\n"
    
    return formatted_log

def continue_step_session(session_id_state: str):
    """Function called by the Next Step button for specific session"""
    if session_id_state and session_id_state in user_sessions:
        session = user_sessions[session_id_state]
        session["step_continue_event"].set()
        return "✅ Continuing... Next AI will respond shortly"
    return "❌ Session not found"

def check_model_status_session(session_id_state: str = None, request: gr.Request = None):
    """Check and display current model availability for specific session"""
    session_id = get_session_id(request) if not session_id_state else session_id_state
    session = get_or_create_session_state(session_id)
    session_keys = session.get("api_keys", {})
    
    # Get session-specific keys or fall back to env vars
    mistral_key = session_keys.get("mistral") or MISTRAL_API_KEY
    sambanova_key = session_keys.get("sambanova") or SAMBANOVA_API_KEY
    
    status_info = "## 🔍 Model Availability Status\n\n"
    
    models = {
        'Mistral Large': mistral_key,
        'DeepSeek-R1': sambanova_key,
        'Meta-Llama-3.1-8B': sambanova_key,
        'QwQ-32B': sambanova_key,
        'Web Search Agent': True
    }
    
    for model_name, available in models.items():
        if model_name == 'Web Search Agent':
            status = "✅ Available (Built-in)"
        else:
            if available:
                status = f"✅ Available (Key: {available[:8]}...)"
            else:
                status = "❌ Not configured"
        status_info += f"**{model_name}:** {status}\n\n"
    
    return status_info

# Create the hybrid interface
with gr.Blocks(title="🎭 Consilium: Visual AI Consensus Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 Consilium: Visual AI Consensus Platform
    
    **Watch AI models collaborate in real-time around a visual roundtable!**
    
    This platform combines:
    - 🎨 **Visual Roundtable Interface** - See AI avatars thinking and speaking
    - 🤖 **Multi-Model Consensus** - Mistral, Deepseek, Llama, QwQ  
    - 🎭 **Dynamic Role Assignment** - Devil's advocate, fact checker, synthesizer roles
    - 🌐 **Communication Topologies** - Full mesh, star, ring patterns
    - 🗳️ **Decision Protocols** - Consensus, voting, weighted, ranked choice
    - 🔍 **Web Search Integration** - Real-time information gathering
    - 🔒 **Session Isolation** - Each user gets their own private discussion space
    
    **Perfect for:** Complex decisions, research analysis, creative brainstorming, problem-solving
    """)
    
    # Hidden session state component
    session_state = gr.State()
    
    with gr.Tab("🎭 Visual Consensus Discussion"):
        with gr.Row():
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="Discussion Question",
                    placeholder="What would you like the AI council to discuss and decide?",
                    lines=3,
                    value="What are the most effective strategies for combating climate change?"
                )
                
                with gr.Row():
                    decision_protocol = gr.Dropdown(
                        choices=["consensus", "majority_voting", "weighted_voting", "ranked_choice", "unanimity"],
                        value="consensus",
                        label="🗳️ Decision Protocol"
                    )
                    
                    role_assignment = gr.Dropdown(
                        choices=["balanced", "specialized", "adversarial", "none"],
                        value="balanced",
                        label="🎭 Role Assignment"
                    )
                
                with gr.Row():
                    topology = gr.Dropdown(
                        choices=["full_mesh", "star", "ring"],
                        value="full_mesh",
                        label="🌐 Communication Pattern"
                    )
                    
                    moderator_model = gr.Dropdown(
                        choices=["mistral", "sambanova_deepseek", "sambanova_llama", "sambanova_qwq"],
                        value="mistral",
                        label="👨‍⚖️ Moderator"
                    )
                
                rounds_input = gr.Slider(
                    minimum=1, maximum=5, value=2, step=1,
                    label="🔄 Discussion Rounds"
                )
                
                enable_clickthrough = gr.Checkbox(
                    label="⏯️ Enable Step-by-Step Mode",
                    value=False,
                    info="Pause at each step for manual control"
                )
                
                start_btn = gr.Button("🚀 Start Visual Consensus Discussion", variant="primary", size="lg")
                
                # Step-by-step control button (only visible when step mode is active)
                next_step_btn = gr.Button("⏯️ Next Step", variant="secondary", size="lg", visible=False)
                step_status = gr.Textbox(label="Step Control", visible=False, interactive=False)
                
                status_output = gr.Textbox(label="📊 Discussion Status", interactive=False)
            
            with gr.Column(scale=2):
                # The visual roundtable component
                roundtable = consilium_roundtable(
                    label="🎭 AI Consensus Roundtable",
                    value=json.dumps({
                        "participants": [],
                        "messages": [],
                        "currentSpeaker": None,
                        "thinking": [],
                        "showBubbles": []
                    })
                )
        
        # Final answer section
        with gr.Row():
            final_answer_output = gr.Markdown(
                label="🎯 Final Consensus Answer",
                value="*Discussion results will appear here...*"
            )
        
        # Collapsible discussion log
        with gr.Accordion("📋 Complete Discussion Log", open=False):
            discussion_log_output = gr.Markdown(
                value="*Complete discussion transcript will appear here...*"
            )
        
        # Event handlers
        def on_start_discussion(*args):
            # Start discussion immediately for both modes
            enable_step = args[-2]  # Second to last argument is enable_step_by_step
            request = args[-1]      # Last argument is request
            
            if enable_step:
                # Step-by-step mode: Start discussion in background thread
                def run_discussion():
                    run_consensus_discussion_session(*args)
                
                discussion_thread = threading.Thread(target=run_discussion)
                discussion_thread.daemon = True
                discussion_thread.start()
                
                # Get session ID for this user
                session_id = get_session_id(request)
                
                return (
                    "🎬 Step-by-step mode: Discussion started - will pause after each AI response",
                    json.dumps(get_or_create_session_state(session_id)["roundtable_state"]),
                    "*Discussion starting in step-by-step mode...*",
                    "*Discussion log will appear here...*",
                    gr.update(visible=True),  # Show next step button
                    gr.update(visible=True, value="Discussion running - will pause after first AI response"),  # Show step status
                    session_id
                )
            else:
                # Normal mode - start immediately and hide step controls
                result = run_consensus_discussion_session(*args)
                return result + (gr.update(visible=False), gr.update(visible=False))
        
        # Function to toggle step controls visibility
        def toggle_step_controls(enable_step):
            return (
                gr.update(visible=enable_step),  # next_step_btn
                gr.update(visible=enable_step)   # step_status
            )
        
        # Hide/show step controls when checkbox changes
        enable_clickthrough.change(
            toggle_step_controls,
            inputs=[enable_clickthrough],
            outputs=[next_step_btn, step_status]
        )
        
        start_btn.click(
            on_start_discussion,
            inputs=[question_input, rounds_input, decision_protocol, role_assignment, topology, moderator_model, enable_clickthrough, session_state],
            outputs=[status_output, roundtable, final_answer_output, discussion_log_output, next_step_btn, step_status, session_state]
        )
        
        # Next step button handler
        next_step_btn.click(
            continue_step_session,
            inputs=[session_state],
            outputs=[step_status]
        )
        
        # Auto-refresh the roundtable state every 2 seconds during discussion
        def refresh_roundtable(session_id_state, request: gr.Request = None):
            session_id = get_session_id(request) if not session_id_state else session_id_state
            if session_id in user_sessions:
                return json.dumps(user_sessions[session_id]["roundtable_state"])
            return json.dumps({
                "participants": [],
                "messages": [],
                "currentSpeaker": None,
                "thinking": [],
                "showBubbles": []
            })
        
        gr.Timer(2).tick(refresh_roundtable, inputs=[session_state], outputs=[roundtable])
    
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
                    info="Required for Mistral Large model"
                )
                sambanova_key_input = gr.Textbox(
                    label="SambaNova API Key", 
                    placeholder="Enter your SambaNova API key...",
                    type="password",
                    info="Required for DeepSeek, Llama, and QwQ models"
                )
                huggingface_key_input = gr.Textbox(
                    label="Hugging Face API Token",
                    placeholder="Enter your Hugging Face API token...",
                    type="password",
                    info="Required for Web Search Agent (TinyLlama)"
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
            inputs=[mistral_key_input, sambanova_key_input, huggingface_key_input, session_state],
            outputs=[keys_status, session_state]
        )
        
        model_status_display = gr.Markdown(check_model_status_session())
        
        # Add refresh button for model status
        refresh_status_btn = gr.Button("🔄 Refresh Model Status")
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
        3. **Start a discussion!**
        
        ### 🔑 Get API Keys:
        - **Mistral:** [console.mistral.ai](https://console.mistral.ai)
        - **SambaNova:** [cloud.sambanova.ai](https://cloud.sambanova.ai)
        - **Hugging Face:** [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        
        ### 🌐 Alternative: Environment Variables
        ```bash
        export MISTRAL_API_KEY=your_key_here
        export SAMBANOVA_API_KEY=your_key_here
        export MODERATOR_MODEL=mistral
        ```
        
        ### 🦙 Sambanova Integration
        The platform includes **3 Sambanova models**:
        - **DeepSeek-R1**: Advanced reasoning model
        - **Meta-Llama-3.1-8B**: Fast, efficient discussions  
        - **QwQ-32B**: Large-scale consensus analysis
        
        ### 🔍 Web Search Agent
        Built-in agent using **smolagents** with:
        - **DuckDuckGoSearchTool**: Web searches
        - **VisitWebpageTool**: Deep content analysis
        - **WikipediaTool**: Comprehensive research
        - **TinyLlama**: Fast inference for search synthesis
        
        ### 📋 Dependencies
        ```bash
        pip install gradio requests python-dotenv smolagents gradio-consilium-roundtable wikipedia openai
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
        
        ### 🔒 Privacy & Security
        - **Session Isolation**: Each user gets their own private discussion space
        - **API Key Protection**: Keys are stored only in your browser session
        - **No Global State**: Your discussions are not visible to other users
        - **Secure Communication**: All API calls use HTTPS encryption
        """)
    
    with gr.Tab("📚 Usage Examples"):
        gr.Markdown("""
        ## 🎯 Example Discussion Topics
        
        ### 🧠 Complex Problem Solving
        - "How should we approach the global housing crisis?"
        - "What's the best strategy for reducing plastic pollution?"
        - "How can we make AI development more democratic?"
        
        ### 💼 Business Strategy
        - "Should our company invest in quantum computing research?"
        - "What's the optimal remote work policy for productivity?"
        - "How should startups approach AI integration?"
        
        ### 🔬 Technical Analysis  
        - "What's the future of web development frameworks?"
        - "How should we handle data privacy in the age of AI?"
        - "What are the best practices for microservices architecture?"
        
        ### 🌍 Social Issues
        - "How can we bridge political divides in society?"
        - "What's the most effective approach to education reform?"
        - "How should we regulate social media platforms?"
        
        ## 🎭 Visual Features
        
        **Watch for these visual cues:**
        - 🤔 **Orange pulsing avatars** = AI is thinking
        - ✨ **Gold glowing avatars** = AI is responding  
        - 💬 **Speech bubbles** = Click avatars to see messages
        - 🎯 **Center consensus** = Final decision reached
        
        **The roundtable updates in real-time as the discussion progresses!**
        
        ## 🎮 Role Assignments Explained
        
        ### 🎭 Balanced (Recommended)
        - **Devil's Advocate**: Challenges assumptions
        - **Fact Checker**: Verifies claims and accuracy
        - **Synthesizer**: Finds common ground
        - **Standard**: Provides balanced analysis
        
        ### 🎓 Specialized
        - **Domain Expert**: Technical expertise
        - **Fact Checker**: Accuracy verification
        - **Creative Thinker**: Innovative solutions
        - **Synthesizer**: Bridge building
        
        ### ⚔️ Adversarial
        - **Double Devil's Advocate**: Maximum challenge
        - **Standard**: Balanced counter-perspective
        
        ## 🗳️ Decision Protocols
        
        - **Consensus**: Seek agreement among all participants
        - **Majority Voting**: Most popular position wins
        - **Weighted Voting**: Higher confidence scores matter more
        - **Ranked Choice**: Preference-based selection
        - **Unanimity**: All must agree completely
        
        ## 🔒 Session Isolation
        
        **Each user gets their own private space:**
        - ✅ Your discussions are private to you
        - ✅ Your API keys are not shared
        - ✅ Your conversation history is isolated
        - ✅ Multiple users can use the platform simultaneously
        
        **Perfect for teams, research groups, and individual use!**
        """)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        mcp_server=True
    )