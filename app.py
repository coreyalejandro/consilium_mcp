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
from gradio_consilium_roundtable import consilium_roundtable
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, VisitWebpageTool, Tool

# Load environment variables
load_dotenv()

# API Configuration - These will be updated by UI if needed
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
MODERATOR_MODEL = os.getenv("MODERATOR_MODEL", "mistral")

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
        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(), 
                VisitWebpageTool(),
                WikipediaTool(),
                FinalAnswerTool()
            ], 
            model=InferenceClientModel(),
            max_steps=5,
            verbosity_level=1
        )
    
    def search(self, query: str, max_results: int = 5) -> str:
        """Use the CodeAgent to perform comprehensive web search and analysis"""
        try:
            # Create a detailed prompt for the agent
            agent_prompt = f"""You are a web research agent. Please research the following query comprehensively:

"{query}"

Your task:
1. Search for relevant information using DuckDuckGo or Wikipedia
2. Visit the most promising web pages to get detailed information
3. Synthesize the findings into a comprehensive, well-formatted response
4. Include sources and links where appropriate
5. Format your response with markdown for better readability

Please provide a thorough analysis based on current, reliable information."""

            # Run the agent
            result = self.agent.run(agent_prompt)
            
            # Format the result nicely
            if result:
                return f"🔍 **Web Research Results for:** {query}\n\n{result}"
            else:
                return f"🔍 **Web Search for:** {query}\n\nNo results found or agent encountered an error."
            
        except Exception as e:
            # Fallback to simple error message
            return f"🔍 **Web Search Error for:** {query}\n\nError: {str(e)}\n\nThe search agent encountered an issue. Please try again or rephrase your query."

class VisualConsensusEngine:
    def __init__(self, moderator_model: str = None, update_callback=None):
        global MISTRAL_API_KEY, SAMBANOVA_API_KEY
        
        self.moderator_model = moderator_model or MODERATOR_MODEL
        self.search_agent = WebSearchAgent()
        self.update_callback = update_callback  # For real-time updates
        
        # Use global API keys (which may be updated from UI)
        self.models = {
            'mistral': {
                'name': 'Mistral Large',
                'api_key': MISTRAL_API_KEY,
                'available': bool(MISTRAL_API_KEY)
            },
            'sambanova_deepseek': {
                'name': 'DeepSeek-R1',
                'api_key': SAMBANOVA_API_KEY,
                'available': bool(SAMBANOVA_API_KEY)
            },
            'sambanova_llama': {
                'name': 'Meta-Llama-3.1-8B',
                'api_key': SAMBANOVA_API_KEY,
                'available': bool(SAMBANOVA_API_KEY)
            },
            'sambanova_qwq': {
                'name': 'QwQ-32B',
                'api_key': SAMBANOVA_API_KEY,
                'available': bool(SAMBANOVA_API_KEY)
            },
            'search': {
                'name': 'Web Search Agent',
                'api_key': True,
                'available': True
            }
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
        """Update the visual roundtable state"""
        if self.update_callback:
            self.update_callback(state_update)
    
    def call_model(self, model: str, prompt: str, context: str = "") -> Optional[str]:
        """Generic model calling function"""
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
        global SAMBANOVA_API_KEY
        if not SAMBANOVA_API_KEY:
            return None
            
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://api.sambanova.ai/v1", 
                api_key=SAMBANOVA_API_KEY
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
        global MISTRAL_API_KEY
        if not MISTRAL_API_KEY:
            return None
            
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://api.mistral.ai/v1", 
                api_key=MISTRAL_API_KEY
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
    
    def run_visual_consensus(self, question: str, discussion_rounds: int = 3, 
                           decision_protocol: str = "consensus", role_assignment: str = "balanced",
                           topology: str = "full_mesh", moderator_model: str = "mistral",
                           enable_step_by_step: bool = False):
        """Run consensus with visual updates"""
        
        available_models = [model for model, info in self.models.items() if info['available']]
        if not available_models:
            return "❌ No AI models available"
        
        model_roles = self.assign_roles(available_models, role_assignment)
        participant_names = [self.models[model]['name'] for model in available_models]
        
        # Log the start
        log_discussion_event('phase', content=f"🚀 Starting Discussion: {question}")
        log_discussion_event('phase', content=f"📊 Configuration: {len(available_models)} models, {decision_protocol} protocol, {role_assignment} roles")
        
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
        log_discussion_event('phase', content="📝 Phase 1: Initial Responses")
        
        for model in available_models:
            # Log and set thinking state
            log_discussion_event('thinking', speaker=self.models[model]['name'])
            self.update_visual_state({
                "participants": participant_names,
                "messages": all_messages,
                "currentSpeaker": None,
                "thinking": [self.models[model]['name']]
            })
            
            # No pause before thinking - let AI think immediately
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
            log_discussion_event('speaking', speaker=self.models[model]['name'])
            self.update_visual_state({
                "participants": participant_names,
                "messages": all_messages,
                "currentSpeaker": self.models[model]['name'],
                "thinking": []
            })
            
            # No pause before speaking - let AI respond immediately
            if not enable_step_by_step:
                time.sleep(2)
            
            response = self.call_model(model, prompt)
            
            if response:
                confidence = self._extract_confidence(response)
                message = {
                    "speaker": self.models[model]['name'],
                    "text": response,  # CHANGE: Don't truncate the response
                    "confidence": confidence,
                    "role": role
                }
                all_messages.append(message)
                
                # Log the full response
                log_discussion_event('message', 
                                   speaker=self.models[model]['name'], 
                                   content=response,
                                   role=role,
                                   confidence=confidence)
                
                # Update with new message - add to showBubbles so bubble stays visible
                responded_speakers = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker")))
                
                self.update_visual_state({
                    "participants": participant_names,
                    "messages": all_messages,
                    "currentSpeaker": None,
                    "thinking": [],
                    "showBubbles": responded_speakers  # Keep bubbles visible for all who responded
                })
                
                # PAUSE AFTER AI RESPONSE - this is when user can read the response
                if enable_step_by_step:
                    step_continue_event.clear()
                    step_continue_event.wait()  # Wait for user to click Next Step
                else:
                    time.sleep(0.5)
        
        # Phase 2: Discussion rounds
        if discussion_rounds > 0:
            log_discussion_event('phase', content=f"💬 Phase 2: Discussion Rounds ({discussion_rounds} rounds)")
            
            for round_num in range(discussion_rounds):
                log_discussion_event('phase', content=f"🔄 Discussion Round {round_num + 1}")
                
                for model in available_models:
                    # Log and set thinking state
                    log_discussion_event('thinking', speaker=self.models[model]['name'])
                    self.update_visual_state({
                        "participants": participant_names,
                        "messages": all_messages,
                        "currentSpeaker": None,
                        "thinking": [self.models[model]['name']]
                    })
                    
                    # No pause before thinking
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

                    # Log and set speaking state
                    log_discussion_event('speaking', speaker=self.models[model]['name'])
                    self.update_visual_state({
                        "participants": participant_names,
                        "messages": all_messages,
                        "currentSpeaker": self.models[model]['name'],
                        "thinking": []
                    })
                    
                    # No pause before speaking
                    if not enable_step_by_step:
                        time.sleep(2)
                    
                    response = self.call_model(model, discussion_prompt)
                    
                    if response:
                        confidence = self._extract_confidence(response)
                        message = {
                            "speaker": self.models[model]['name'],
                            "text": f"Round {round_num + 1}: {response}",  # CHANGE: Don't truncate
                            "confidence": confidence,
                            "role": model_roles[model]
                        }
                        all_messages.append(message)
                        
                        # Log the full response
                        log_discussion_event('message', 
                                           speaker=self.models[model]['name'], 
                                           content=f"Round {round_num + 1}: {response}",
                                           role=model_roles[model],
                                           confidence=confidence)
                        
                        # Update with new message - add to showBubbles so bubble stays visible
                        responded_speakers = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker")))
                        
                        self.update_visual_state({
                            "participants": participant_names,
                            "messages": all_messages,
                            "currentSpeaker": None,
                            "thinking": [],
                            "showBubbles": responded_speakers  # Keep bubbles visible for all who responded
                        })
                        
                        # PAUSE AFTER AI RESPONSE for step-by-step mode
                        if enable_step_by_step:
                            step_continue_event.clear()
                            step_continue_event.wait()
                        else:
                            time.sleep(1)
        
        # Phase 3: Final consensus - ACTUALLY GENERATE THE CONSENSUS
        log_discussion_event('phase', content=f"🎯 Phase 3: Final Consensus ({decision_protocol})")
        log_discussion_event('thinking', speaker="All participants", content="Building consensus...")
        
        self.update_visual_state({
            "participants": participant_names,
            "messages": all_messages,
            "currentSpeaker": None,
            "thinking": participant_names  # Everyone thinking about consensus
        })
        
        # No pause before consensus generation
        if not enable_step_by_step:
            time.sleep(2)
        
        # ACTUALLY GENERATE THE FINAL CONSENSUS ANSWER
        moderator = self.moderator_model if self.models[self.moderator_model]['available'] else available_models[0]
        
        # Collect all the actual responses for synthesis
        all_responses = ""
        confidence_scores = []
        for entry in discussion_log:
            if entry['type'] == 'message' and entry['speaker'] != 'Consilium':
                all_responses += f"\n**{entry['speaker']}**: {entry['content']}\n"
                if 'confidence' in entry:
                    confidence_scores.append(entry['confidence'])
        
        # Calculate average confidence to assess consensus likelihood
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 5.0
        consensus_threshold = 7.0  # If average confidence is below this, flag potential disagreement
        
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

        log_discussion_event('speaking', speaker="Consilium", content="Analyzing consensus and synthesizing final answer...")
        self.update_visual_state({
            "participants": participant_names,
            "messages": all_messages,
            "currentSpeaker": "Consilium",
            "thinking": []
        })
        
        # Generate the actual consensus analysis
        consensus_result = self.call_model(moderator, consensus_prompt)
        
        if not consensus_result:
            consensus_result = f"""**CONSENSUS STATUS:** Analysis Failed

**FINAL ANSWER:** Unable to generate consensus analysis. Please review individual participant responses in the discussion log.

**AREAS OF DISAGREEMENT:** Analysis could not be completed due to technical issues."""
        
        # Check if consensus was actually reached based on the response
        consensus_reached = "CONSENSUS STATUS: Reached" in consensus_result or avg_confidence >= consensus_threshold
        
        # Generate final consensus message for visual
        if consensus_reached:
            visual_summary = "✅ Consensus reached!"
        elif "Partial" in consensus_result:
            visual_summary = "⚠️ Partial consensus - some disagreements remain"
        else:
            visual_summary = "❌ No consensus - significant disagreements identified"
        
        final_message = {
            "speaker": "Consilium",
            "text": f"{visual_summary} {consensus_result}",  # CHANGE: Don't truncate consensus
            "confidence": avg_confidence,
            "role": "consensus"
        }
        all_messages.append(final_message)
        
        log_discussion_event('message', 
                           speaker="Consilium", 
                           content=consensus_result,
                           confidence=avg_confidence)
        
        # Final state - show bubbles for all who responded
        responded_speakers = list(set(msg["speaker"] for msg in all_messages if msg.get("speaker")))
        
        self.update_visual_state({
            "participants": participant_names,
            "messages": all_messages,
            "currentSpeaker": None,
            "thinking": [],
            "showBubbles": responded_speakers
        })
        
        log_discussion_event('phase', content="✅ Discussion Complete")
        
        return consensus_result  # Return the actual analysis, including disagreements

# Global state for the visual component
current_roundtable_state = {
    "participants": [],
    "messages": [],
    "currentSpeaker": None,
    "thinking": [],
    "showBubbles": []
}

def update_roundtable_state(new_state):
    """Update the global roundtable state"""
    global current_roundtable_state
    current_roundtable_state.update(new_state)
    return json.dumps(current_roundtable_state)

# Global variables for step-by-step control
step_pause_queue = queue.Queue()
step_continue_event = threading.Event()

def run_consensus_discussion(question: str, discussion_rounds: int = 3, 
                           decision_protocol: str = "consensus", role_assignment: str = "balanced",
                           topology: str = "full_mesh", moderator_model: str = "mistral",
                           enable_step_by_step: bool = False):
    """Main function that returns both text log and updates visual state"""
    
    global discussion_log, final_answer, step_by_step_active, step_continue_event
    discussion_log = []  # Reset log
    final_answer = ""
    step_by_step_active = enable_step_by_step
    step_continue_event.clear()
    
    def visual_update_callback(state_update):
        """Callback to update visual state during discussion"""
        update_roundtable_state(state_update)
    
    engine = VisualConsensusEngine(moderator_model, visual_update_callback)
    result = engine.run_visual_consensus(
        question, discussion_rounds, decision_protocol, 
        role_assignment, topology, moderator_model, enable_step_by_step
    )
    
    # Generate final answer summary  
    available_models = [model for model, info in engine.models.items() if info['available']]
    final_answer = f"""## 🎯 Final Consensus Answer

{result}

---

### 📊 Discussion Summary
- **Question:** {question}
- **Protocol:** {decision_protocol.replace('_', ' ').title()}
- **Participants:** {len(available_models)} AI models
- **Roles:** {role_assignment.title()}
- **Communication:** {topology.replace('_', ' ').title()}
- **Rounds:** {discussion_rounds}

*Generated by Consilium Visual AI Consensus Platform*"""
    
    step_by_step_active = False  # Reset after discussion
    
    # Return ONLY status for the status field, not the full result
    status_text = "✅ Discussion Complete - See results below"
    return status_text, json.dumps(current_roundtable_state), final_answer, format_discussion_log()

def continue_step():
    """Function called by the Next Step button"""
    global step_continue_event
    step_continue_event.set()
    return "✅ Continuing... Next AI will respond shortly"

# Global variables for step-by-step control
discussion_log = []
final_answer = ""
step_by_step_active = False
current_step_data = {}
step_callback = None

def set_step_callback(callback):
    """Set the callback for step-by-step mode"""
    global step_callback
    step_callback = callback

def wait_for_next_step():
    """Wait for user to click 'Next Step' button in step-by-step mode"""
    global step_by_step_active
    if step_by_step_active and step_callback:
        # Return control to UI - the next step button will continue
        return True
    return False

def format_discussion_log():
    """Format the complete discussion log for display"""
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

def log_discussion_event(event_type: str, speaker: str = "", content: str = "", **kwargs):
    """Add an event to the discussion log"""
    global discussion_log
    discussion_log.append({
        'type': event_type,
        'speaker': speaker,
        'content': content,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        **kwargs
    })

def update_api_keys(mistral_key, sambanova_key):
    """Update API keys from UI input"""
    global MISTRAL_API_KEY, SAMBANOVA_API_KEY
    
    status_messages = []
    
    # Update Mistral key if provided, otherwise keep env var
    if mistral_key.strip():
        MISTRAL_API_KEY = mistral_key.strip()
        status_messages.append("✅ Mistral API key updated")
    elif not MISTRAL_API_KEY:
        status_messages.append("❌ No Mistral API key (env or input)")
    else:
        status_messages.append("✅ Using Mistral API key from environment")
        
    # Update SambaNova key if provided, otherwise keep env var  
    if sambanova_key.strip():
        SAMBANOVA_API_KEY = sambanova_key.strip()
        status_messages.append("✅ SambaNova API key updated")
    elif not SAMBANOVA_API_KEY:
        status_messages.append("❌ No SambaNova API key (env or input)")
    else:
        status_messages.append("✅ Using SambaNova API key from environment")
    
    # Check if we have at least one working key
    if not MISTRAL_API_KEY and not SAMBANOVA_API_KEY:
        return "❌ ERROR: No API keys available! Please provide at least one API key."
    
    return " | ".join(status_messages)

def check_model_status():
    """Check and display current model availability"""
    global MISTRAL_API_KEY, SAMBANOVA_API_KEY
    
    status_info = "## 🔍 Model Availability Status\n\n"
    
    models = {
        'Mistral Large': MISTRAL_API_KEY,
        'DeepSeek-R1': SAMBANOVA_API_KEY,
        'Meta-Llama-3.1-8B': SAMBANOVA_API_KEY,
        'QwQ-32B': SAMBANOVA_API_KEY,
        'Web Search Agent': True
    }
    
    for model_name, available in models.items():
        if model_name == 'Web Search Agent':
            status = "✅ Available (Built-in)"
        else:
            status = "✅ Available" if available else "❌ Not configured"
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
    
    **Perfect for:** Complex decisions, research analysis, creative brainstorming, problem-solving
    """)
    
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
                    value=json.dumps(current_roundtable_state)
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
            enable_step = args[-1]  # Last argument is enable_step_by_step
            
            if enable_step:
                # Step-by-step mode: Start discussion in background thread
                def run_discussion():
                    run_consensus_discussion(*args)
                
                discussion_thread = threading.Thread(target=run_discussion)
                discussion_thread.daemon = True
                discussion_thread.start()
                
                return (
                    "🎬 Step-by-step mode: Discussion started - will pause after each AI response",
                    json.dumps(current_roundtable_state),
                    "*Discussion starting in step-by-step mode...*",
                    "*Discussion log will appear here...*",
                    gr.update(visible=True),  # Show next step button
                    gr.update(visible=True, value="Discussion running - will pause after first AI response")  # Show step status
                )
            else:
                # Normal mode - start immediately and hide step controls
                result = run_consensus_discussion(*args)
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
            inputs=[question_input, rounds_input, decision_protocol, role_assignment, topology, moderator_model, enable_clickthrough],
            outputs=[status_output, roundtable, final_answer_output, discussion_log_output, next_step_btn, step_status]
        )
        
        # Next step button handler
        next_step_btn.click(
            continue_step,
            outputs=[step_status]
        )
        
        # Auto-refresh the roundtable state every 2 seconds during discussion
        gr.Timer(2).tick(lambda: json.dumps(current_roundtable_state), outputs=[roundtable])
    
    with gr.Tab("🔧 Configuration & Setup"):
        gr.Markdown("## 🔑 API Keys Configuration")
        gr.Markdown("*Enter your API keys below OR set them as environment variables*")
        
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
            update_api_keys,
            inputs=[mistral_key_input, sambanova_key_input],
            outputs=[keys_status]
        )
        
        model_status_display = gr.Markdown(check_model_status())
        
        # Add refresh button for model status
        refresh_status_btn = gr.Button("🔄 Refresh Model Status")
        refresh_status_btn.click(
            check_model_status,
            outputs=[model_status_display]
        )
        
        gr.Markdown("""
        ## 🛠️ Setup Instructions
        
        ### 🚀 Quick Start (Recommended)
        1. **Enter API keys above** (they'll be used for this session)
        2. **Click "Save API Keys"** 
        3. **Start a discussion!**
        
        ### 🔑 Get API Keys:
        - **Mistral:** [console.mistral.ai](https://console.mistral.ai)
        - **SambaNova:** [cloud.sambanova.ai](https://cloud.sambanova.ai)
        
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