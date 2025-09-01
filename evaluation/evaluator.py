"""
Model Evaluator for Fine-Tuning Pipeline

Evaluates fine-tuned models on consensus quality and other metrics.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from .metrics import ConsensusMetrics

class ModelEvaluator:
    """Evaluates fine-tuned models"""
    
    def __init__(self):
        self.metrics = ConsensusMetrics()
    
    def evaluate_model(self, model_path: str, test_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a fine-tuned model on test data"""
        
        results = {
            "model_path": model_path,
            "total_examples": len(test_examples),
            "consensus_quality": 0.0,
            "research_depth": 0.0,
            "reasoning_clarity": 0.0,
            "response_completeness": 0.0,
            "tool_usage_effectiveness": 0.0,
            "overall_score": 0.0,
            "detailed_scores": []
        }
        
        if not test_examples:
            return results
            
        # Calculate metrics for each example
        scores = []
        for example in test_examples:
            example_scores = self._evaluate_example(example)
            scores.append(example_scores)
            results["detailed_scores"].append(example_scores)
        
        # Calculate averages
        if scores:
            results["consensus_quality"] = np.mean([s["consensus_quality"] for s in scores])
            results["research_depth"] = np.mean([s["research_depth"] for s in scores])
            results["reasoning_clarity"] = np.mean([s["reasoning_clarity"] for s in scores])
            results["response_completeness"] = np.mean([s["response_completeness"] for s in scores])
            results["tool_usage_effectiveness"] = np.mean([s["tool_usage_effectiveness"] for s in scores])
            results["overall_score"] = np.mean([s["overall_score"] for s in scores])
        
        return results
    
    def _evaluate_example(self, example: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single example"""
        
        # Extract components
        prompt = example.get('prompt', '')
        final_output = example.get('final', '')
        tools = example.get('tools', [])
        metadata = example.get('meta', {})
        
        # Calculate individual metrics
        consensus_quality = self.metrics.calculate_consensus_quality(final_output, metadata)
        research_depth = self.metrics.calculate_research_depth(tools, final_output)
        reasoning_clarity = self.metrics.calculate_reasoning_clarity(final_output)
        response_completeness = self.metrics.calculate_completeness(prompt, final_output)
        tool_usage = self.metrics.calculate_tool_usage(tools, final_output)
        
        # Calculate overall score
        overall_score = np.mean([
            consensus_quality,
            research_depth,
            reasoning_clarity,
            response_completeness,
            tool_usage
        ])
        
        return {
            "consensus_quality": consensus_quality,
            "research_depth": research_depth,
            "reasoning_clarity": reasoning_clarity,
            "response_completeness": response_completeness,
            "tool_usage_effectiveness": tool_usage,
            "overall_score": overall_score
        }
    
    def compare_models(self, model_paths: List[str], test_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple models"""
        
        comparison = {
            "models": {},
            "best_model": None,
            "best_score": 0.0
        }
        
        for model_path in model_paths:
            results = self.evaluate_model(model_path, test_examples)
            comparison["models"][model_path] = results
            
            if results["overall_score"] > comparison["best_score"]:
                comparison["best_score"] = results["overall_score"]
                comparison["best_model"] = model_path
        
        return comparison
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate a detailed evaluation report"""
        
        report = {
            "evaluation_summary": {
                "model_path": results["model_path"],
                "total_examples": results["total_examples"],
                "overall_score": results["overall_score"],
                "timestamp": "2024-01-15T10:30:00Z"
            },
            "metric_breakdown": {
                "consensus_quality": results["consensus_quality"],
                "research_depth": results["research_depth"],
                "reasoning_clarity": results["reasoning_clarity"],
                "response_completeness": results["response_completeness"],
                "tool_usage_effectiveness": results["tool_usage_effectiveness"]
            },
            "detailed_scores": results["detailed_scores"]
        }
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
