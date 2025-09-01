"""
Data Processing for Fine-Tuning Pipeline

Processes raw training data into formats suitable for
different fine-tuning approaches.
"""

import json
import jsonlines
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from .quality_scorer import QualityScorer

class DataProcessor:
    """Process training data for fine-tuning"""
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_scorer = QualityScorer(quality_threshold)
        
    def load_training_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load training data from JSONL file"""
        examples = []
        
        with jsonlines.open(file_path) as reader:
            for example in reader:
                examples.append(example)
                
        return examples
    
    def process_for_sft(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process examples for Supervised Fine-Tuning format"""
        sft_examples = []
        
        for example in examples:
            # Create SFT format
            sft_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": self._format_prompt(example)
                    },
                    {
                        "role": "assistant", 
                        "content": example.get('final', '')
                    }
                ]
            }
            sft_examples.append(sft_example)
            
        return sft_examples
    
    def process_for_dpo(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process examples for Direct Preference Optimization format"""
        dpo_examples = []
        
        for example in examples:
            # Create DPO format with preference data
            dpo_example = {
                "prompt": self._format_prompt(example),
                "chosen": example.get('final', ''),
                "rejected": self._generate_rejected_response(example)
            }
            dpo_examples.append(dpo_example)
            
        return dpo_examples
    
    def process_for_instruction_tuning(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process examples for instruction tuning format"""
        instruction_examples = []
        
        for example in examples:
            instruction_example = {
                "instruction": example.get('prompt', ''),
                "input": self._format_context(example),
                "output": example.get('final', ''),
                "context": self._extract_context(example)
            }
            instruction_examples.append(instruction_example)
            
        return instruction_examples
    
    def filter_and_split(self, examples: List[Dict[str, Any]], 
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1) -> Tuple[List, List, List]:
        """Filter examples by quality and split into train/val/test"""
        
        # Filter by quality
        filtered_examples = self.quality_scorer.filter_examples(examples)
        
        # Shuffle
        import random
        random.shuffle(filtered_examples)
        
        # Split
        total = len(filtered_examples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = filtered_examples[:train_end]
        val_data = filtered_examples[train_end:val_end]
        test_data = filtered_examples[val_end:]
        
        return train_data, val_data, test_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], 
                          output_path: str, 
                          format_type: str = "sft") -> None:
        """Save processed data in specified format"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "sft":
            processed_data = self.process_for_sft(data)
        elif format_type == "dpo":
            processed_data = self.process_for_dpo(data)
        elif format_type == "instruction":
            processed_data = self.process_for_instruction_tuning(data)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
            
        with jsonlines.open(output_path, mode='w') as writer:
            for example in processed_data:
                writer.write(example)
    
    def generate_statistics(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about the dataset"""
        stats = {
            "total_examples": len(examples),
            "avg_prompt_length": 0,
            "avg_response_length": 0,
            "tool_usage_stats": {},
            "quality_distribution": {},
            "model_distribution": {}
        }
        
        if not examples:
            return stats
            
        # Calculate averages
        prompt_lengths = [len(ex.get('prompt', '').split()) for ex in examples]
        response_lengths = [len(ex.get('final', '').split()) for ex in examples]
        
        stats["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths)
        stats["avg_response_length"] = sum(response_lengths) / len(response_lengths)
        
        # Tool usage stats
        tool_counts = {}
        for example in examples:
            tools = example.get('tools', [])
            for tool in tools:
                tool_name = tool.get('name', 'unknown')
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        stats["tool_usage_stats"] = tool_counts
        
        # Quality distribution
        quality_scores = []
        for example in examples:
            if 'quality_metrics' in example:
                quality_scores.append(example['quality_metrics']['overall_score'])
        
        if quality_scores:
            stats["quality_distribution"] = {
                "min": min(quality_scores),
                "max": max(quality_scores),
                "mean": sum(quality_scores) / len(quality_scores),
                "count": len(quality_scores)
            }
        
        # Model distribution
        model_counts = {}
        for example in examples:
            model = example.get('calling_model', 'unknown')
            model_counts[model] = model_counts.get(model, 0) + 1
        stats["model_distribution"] = model_counts
        
        return stats
    
    def _format_prompt(self, example: Dict[str, Any]) -> str:
        """Format the prompt for training"""
        prompt = example.get('prompt', '')
        tools = example.get('tools', [])
        
        if not tools:
            return prompt
            
        # Add tool context to prompt
        tool_context = "\n\nResearch Context:\n"
        for i, tool in enumerate(tools, 1):
            tool_name = tool.get('name', f'Tool {i}')
            tool_content = tool.get('content', '')[:500]  # Truncate long content
            tool_context += f"{i}. {tool_name}: {tool_content}\n"
            
        return prompt + tool_context
    
    def _format_context(self, example: Dict[str, Any]) -> str:
        """Format context information"""
        tools = example.get('tools', [])
        context_parts = []
        
        for tool in tools:
            tool_name = tool.get('name', '')
            tool_content = tool.get('content', '')
            if tool_content:
                context_parts.append(f"{tool_name}: {tool_content[:200]}...")
                
        return "\n".join(context_parts) if context_parts else ""
    
    def _extract_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context metadata"""
        return {
            "session_id": example.get('session_id'),
            "calling_model": example.get('calling_model'),
            "timestamp": example.get('ts'),
            "tool_count": len(example.get('tools', [])),
            "metadata": example.get('meta', {})
        }
    
    def _generate_rejected_response(self, example: Dict[str, Any]) -> str:
        """Generate a rejected response for DPO training"""
        # This is a placeholder - in practice, you'd want to generate
        # lower quality responses or use human feedback data
        original_response = example.get('final', '')
        
        # Simple degradation: truncate and add noise
        words = original_response.split()
        if len(words) > 50:
            degraded = " ".join(words[:len(words)//2])
            degraded += " [Response truncated due to length constraints]"
            return degraded
        else:
            return "I don't have enough information to provide a comprehensive answer."
