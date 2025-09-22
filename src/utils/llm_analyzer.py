"""
LLM Analysis utility for Code Quality and Dataset Quality evaluation.
"""

import os
import json
from typing import Dict, Optional, Any
import requests


class LLMAnalyzer:
    """
    Utility class for analyzing code and dataset quality using LLM.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM analyzer."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def analyze_dataset_quality(self, readme_content: str,
                                metadata: Dict[str, Any]) -> float:
        """
        Analyze dataset quality using LLM.
        
        Args:
            readme_content: The README file content
            metadata: Repository metadata dictionary
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        if not self.api_key:
            return self._fallback_dataset_analysis(readme_content, metadata)

        prompt = f"""
Analyze this AI/ML repository for dataset quality (score 0.0-1.0):

README (first 1500 chars):
{readme_content[:1500]}

Metadata:
{json.dumps(metadata, indent=2)[:800]}

Consider:
- Dataset documentation quality
- Data validation/preprocessing mentions
- Dataset diversity and completeness
- Training data transparency
- Bias considerations

Return only a number between 0.0 and 1.0:
"""

        try:
            response = self._call_llm(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"LLM dataset analysis failed: {e}")
            return self._fallback_dataset_analysis(readme_content, metadata)

    def analyze_code_quality(self, code_content: str,
                             readme_content: str) -> float:
        """
        Analyze code quality using LLM.
        
        Args:
            code_content: Sample code content
            readme_content: README content for context
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        if not self.api_key:
            return self._fallback_code_analysis(code_content, readme_content)

        prompt = f"""
Analyze this code for quality and maintainability (score 0.0-1.0):

Code sample (first 1200 chars):
{code_content[:1200]}

README context (first 800 chars):
{readme_content[:800]}

Consider:
- Code structure and organization
- Documentation and comments
- Error handling
- Maintainability
- Best practices

Return only a number between 0.0 and 1.0:
"""

        try:
            response = self._call_llm(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"LLM code analysis failed: {e}")
            return self._fallback_code_analysis(code_content, readme_content)

    def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM service."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an expert in code and "
                 "dataset quality assessment. Respond with only a numeric "
                 "score."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        return result['choices'][0]['message']['content']

    def _fallback_dataset_analysis(self, readme_content: str,
                                   metadata: Dict[str, Any]) -> float:
        """Rule-based fallback for dataset quality analysis."""
        readme_lower = readme_content.lower()
        
        # Check for dataset quality indicators
        quality_indicators = [
            'dataset', 'training data', 'validation', 'preprocessing',
            'data quality', 'bias', 'diversity', 'balanced'
        ]
        
        score = 0.0
        for indicator in quality_indicators:
            if indicator in readme_lower:
                score += 0.1
        
        # Bonus for detailed documentation
        if 'evaluation' in readme_lower:
            score += 0.1
        if 'benchmark' in readme_lower:
            score += 0.1
        
        return min(1.0, score)

    def _fallback_code_analysis(self, code_content: str,
                                readme_content: str) -> float:
        """Rule-based fallback for code quality analysis."""
        if not code_content:
            return 0.0
        
        lines = code_content.split('\n')
        total_lines = len(lines)
        
        if total_lines == 0:
            return 0.0
        
        # Count comments and docstrings
        comment_lines = sum(1 for line in lines if line.strip().startswith(
            ('#', '"""', "'''", '//', '*', '/*')))
        
        # Check for error handling
        error_keywords = ['try', 'except', 'catch', 'error', 'exception']
        error_handling = any(keyword in code_content.lower()
                             for keyword in error_keywords)
        
        # Check for functions/classes
        structure_keywords = ['def ', 'class ', 'function ']
        has_structure = any(keyword in code_content
                            for keyword in structure_keywords)
        
        # Calculate score
        documentation_ratio = comment_lines / total_lines
        score = 0.3  # Base score
        
        score += min(0.3, documentation_ratio * 2)  # Documentation
        score += 0.2 if error_handling else 0  # Error handling
        score += 0.2 if has_structure else 0  # Code structure
        
        return min(1.0, score)
