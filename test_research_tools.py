#!/usr/bin/env python3
"""
Test Script for Enhanced Research Tools
Run this to verify all research tools are working correctly
"""

import sys
import os
import time
from typing import Dict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from research_tools import EnhancedResearchAgent
    from enhanced_search_functions import get_function_definitions, get_function_names
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all research_tools files are in place!")
    IMPORTS_OK = False


def test_tool_imports():
    """Test that all tools can be imported"""
    print("🔍 Testing Tool Imports...")
    
    if not IMPORTS_OK:
        return False
    
    try:
        from research_tools.web_search import WebSearchTool
        from research_tools.wikipedia_search import WikipediaSearchTool
        from research_tools.arxiv_search import ArxivSearchTool
        from research_tools.github_search import GitHubSearchTool
        from research_tools.sec_search import SECSearchTool
        from research_tools.scholar_search import GoogleScholarTool
        
        print("✅ All tool imports successful")
        return True
    except ImportError as e:
        print(f"❌ Tool import failed: {e}")
        return False


def test_enhanced_research_agent():
    """Test the main research agent"""
    print("\n🤖 Testing Enhanced Research Agent...")
    
    if not IMPORTS_OK:
        return False
    
    try:
        agent = EnhancedResearchAgent()
        print(f"✅ Research agent created with {len(agent.tools)} tools")
        
        # Test tool status
        status = agent.get_tool_status()
        print(f"✅ Tool status check: {len(status)} tools available")
        
        return True
    except Exception as e:
        print(f"❌ Research agent creation failed: {e}")
        return False


def test_function_definitions():
    """Test function definitions"""
    print("\n📋 Testing Function Definitions...")
    
    try:
        functions = get_function_definitions()
        function_names = get_function_names()
        
        print(f"✅ {len(functions)} function definitions loaded")
        print(f"✅ Function names: {', '.join(function_names)}")
        
        # Verify structure
        for func in functions:
            assert "type" in func
            assert "function" in func
            assert "name" in func["function"]
            assert "parameters" in func["function"]
        
        print("✅ All function definitions have correct structure")
        return True
    except Exception as e:
        print(f"❌ Function definition test failed: {e}")
        return False


def test_individual_tools():
    """Test each research tool individually"""
    print("\n🔧 Testing Individual Tools...")
    
    if not IMPORTS_OK:
        return False
    
    results = {}
    
    try:
        agent = EnhancedResearchAgent()
        
        # Quick test queries for each tool
        test_queries = {
            'web': ('AI news 2024', {}),
            'wikipedia': ('artificial intelligence', {}),
            'arxiv': ('machine learning', {}),
            'github': ('python', {}),
            'sec': ('Apple', {}),  # Remove max_results for SEC
            'scholar': ('deep learning', {})
        }
        
        for tool_name, (query, kwargs) in test_queries.items():
            print(f"  Testing {tool_name}...")
            try:
                # Quick test with timeout
                start_time = time.time()
                if tool_name == 'sec':
                    # SEC tool only accepts company_name parameter
                    result = agent.tools[tool_name].search(query)
                else:
                    result = agent.tools[tool_name].search(query, max_results=1)
                duration = time.time() - start_time
                
                if result and len(result) > 50:
                    print(f"    ✅ {tool_name}: '{result}' Working ({duration:.1f}s)")
                    results[tool_name] = "✅ Working"
                else:
                    print(f"    ⚠️ {tool_name}: Limited response")
                    results[tool_name] = "⚠️ Limited"
                    
            except Exception as e:
                print(f"    ❌ {tool_name}: Error - {str(e)[:50]}...")
                results[tool_name] = f"❌ Error"
        
        working_tools = sum(1 for status in results.values() if "✅" in status)
        print(f"\n📊 Tool Test Results: {working_tools}/{len(test_queries)} tools working")
        
        return working_tools > 0
        
    except Exception as e:
        print(f"❌ Individual tool testing failed: {e}")
        return False


def test_smart_routing():
    """Test smart query routing"""
    print("\n🎯 Testing Smart Query Routing...")
    
    if not IMPORTS_OK:
        return False
    
    try:
        agent = EnhancedResearchAgent()
        
        test_cases = [
            ("What is machine learning?", "wikipedia"),  # Definitional
            ("Latest AI research papers", "arxiv"),      # Academic
            ("React vs Vue popularity", "github"),       # Technology
            ("Tesla stock performance", "sec"),          # Financial
            ("Current AI news", "web")                   # Current events
        ]
        
        correct_routes = 0
        for query, expected_tool in test_cases:
            routed_tool = agent._route_query_to_tool(query)
            if routed_tool == expected_tool:
                print(f"  ✅ '{query}' → {routed_tool}")
                correct_routes += 1
            else:
                print(f"  ⚠️ '{query}' → {routed_tool} (expected {expected_tool})")
        
        print(f"\n📊 Routing accuracy: {correct_routes}/{len(test_cases)} correct")
        return correct_routes >= len(test_cases) // 2  # At least 50% correct
        
    except Exception as e:
        print(f"❌ Smart routing test failed: {e}")
        return False


def test_multi_source_research():
    """Test multi-source research synthesis"""
    print("\n🌐 Testing Multi-Source Research...")
    
    if not IMPORTS_OK:
        return False
    
    try:
        agent = EnhancedResearchAgent()
        
        print("  Running deep research test (this may take 10-15 seconds)...")
        result = agent.search("artificial intelligence benefits", research_depth="deep")
        
        if result and len(result) > 200:
            # Check for multi-source indicators
            source_indicators = ["Web Search", "Wikipedia", "arXiv", "Research Sources Used"]
            found_sources = sum(1 for indicator in source_indicators if indicator in result)
            
            if found_sources >= 2:
                print(f"  ✅ Multi-source synthesis working ({found_sources} sources detected)")
                return True
            else:
                print(f"  ⚠️ Limited multi-source synthesis ({found_sources} sources)")
                return False
        else:
            print("  ❌ Multi-source research returned insufficient data")
            return False
            
    except Exception as e:
        print(f"❌ Multi-source research test failed: {e}")
        return False


def test_quality_scoring():
    """Test research quality scoring"""
    print("\n📊 Testing Quality Scoring...")
    
    if not IMPORTS_OK:
        return False
    
    try:
        agent = EnhancedResearchAgent()
        
        # Test quality scoring on a sample text
        sample_text = """
        Recent research from Stanford University published in 2024 shows that 
        artificial intelligence accuracy increased by 23% compared to 2023 data.
        The study, published in Nature, analyzed 1,000 AI models and found 
        significant improvements in neural network architectures.
        """
        
        quality_score = agent.tools['web'].score_research_quality(sample_text, 'web')
        
        print(f"  Sample quality score: {quality_score}")
        
        # Verify scoring structure
        required_metrics = ['recency', 'authority', 'specificity', 'relevance', 'overall']
        for metric in required_metrics:
            if metric not in quality_score:
                print(f"  ❌ Missing metric: {metric}")
                return False
            if not 0 <= quality_score[metric] <= 1:
                print(f"  ❌ Invalid score for {metric}: {quality_score[metric]}")
                return False
        
        print("  ✅ Quality scoring structure correct")
        print(f"  ✅ Overall quality: {quality_score['overall']:.2f}/1.0")
        return True
        
    except Exception as e:
        print(f"❌ Quality scoring test failed: {e}")
        return False


def test_dependency_check():
    """Check for required dependencies"""
    print("\n📦 Testing Dependencies...")
    
    dependencies = {
        'requests': 'HTTP requests',
        'xml.etree.ElementTree': 'XML parsing (built-in)',
        'wikipedia': 'Wikipedia search',
        'scholarly': 'Google Scholar (optional)',
        'smolagents': 'Web search agents'
    }
    
    missing_deps = []
    
    for dep, description in dependencies.items():
        try:
            if dep == 'xml.etree.ElementTree':
                import xml.etree.ElementTree
            else:
                __import__(dep)
            print(f"  ✅ {dep}: {description}")
        except ImportError:
            print(f"  ❌ {dep}: {description} - MISSING")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(dep for dep in missing_deps if dep not in ['xml.etree.ElementTree']))
        return False
    else:
        print("  ✅ All dependencies available")
        return True


def run_full_test_suite():
    """Run the complete test suite"""
    print("🧪 Enhanced Research Tools - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependency Check", test_dependency_check),
        ("Tool Imports", test_tool_imports),
        ("Research Agent", test_enhanced_research_agent),
        ("Function Definitions", test_function_definitions),
        ("Individual Tools", test_individual_tools),
        ("Smart Routing", test_smart_routing),
        ("Quality Scoring", test_quality_scoring),
        ("Multi-Source Research", test_multi_source_research)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print(f"\n{'='*50}")
    print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Research system is ready!")
    elif passed >= total * 0.75:
        print("✅ Most tests passed! Research system should work well.")
    elif passed >= total * 0.5:
        print("⚠️ Some tests failed. Research system has limited functionality.")
    else:
        print("❌ Many tests failed. Please check setup and dependencies.")
    
    return passed, total


if __name__ == "__main__":
    run_full_test_suite()