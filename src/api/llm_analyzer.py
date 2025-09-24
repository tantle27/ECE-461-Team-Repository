"""
LLM Analysis utility for README and metadata evaluation.
"""

import os
import json
from typing import Dict, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class LLMAnalyzer:
    """
    Utility class for analyzing README files and repository metadata
    using Large Language Models to assess dataset quality indicators.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM analyzer.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: LLM model to use for analysis
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry policy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["POST"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def analyze_dataset_quality(
            self, readme_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze README and metadata to evaluate dataset quality indicators.
        
        Args:
            readme_content: The README file content
            metadata: Repository metadata dictionary
            
        Returns:
            Dictionary containing dataset quality analysis results
        """
        if not self.api_key:
            # Fallback to rule-based analysis if no API key
            return self._fallback_analysis(readme_content, metadata)

        prompt = self._create_analysis_prompt(readme_content, metadata)
        
        try:
            response = self._call_llm(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            print(f"LLM analysis failed: {e}, falling back to rule-based analysis")
            return self._fallback_analysis(readme_content, metadata)

    def _create_analysis_prompt(
        self, 
        readme_content: str, 
        metadata: Dict[str, Any]
    ) -> str:
        """Create a structured prompt for dataset quality analysis."""
        prompt = f"""
Analyze the following repository's README and metadata to evaluate dataset quality 
indicators for an AI/ML model repository.

README Content:
{readme_content[:2000]}  # Truncate for API limits

Metadata:
{json.dumps(metadata, indent=2)[:1000]}

Please evaluate and return a JSON response with the following boolean/numeric fields:

1. has_data_validation: Does the repository mention data validation, cleaning, or 
   preprocessing steps?
2. data_diversity_score: Score 0.0-1.0 based on mentions of data diversity, 
   multiple sources, balanced datasets
3. data_completeness: Score 0.0-1.0 based on mentions of complete datasets, 
   missing data handling, data coverage
4. has_dataset_card: Does it mention dataset cards, data sheets, or dataset documentation?
5. data_fields_documented: Are the data fields, schema, or format well documented?
6. example_usage: Are there clear examples of how to use the dataset?
7. data_source_credibility: Score 0.0-1.0 based on credibility of data sources mentioned
8. bias_considerations: Score 0.0-1.0 based on discussion of potential biases or limitations

Return only valid JSON in this exact format:
{{
    "has_data_validation": boolean,
    "data_diversity_score": float,
    "data_completeness": float,
    "has_dataset_card": boolean,
    "data_fields_documented": boolean,
    "example_usage": boolean,
    "data_source_credibility": float,
    "bias_considerations": float,
    "confidence_score": float
}}
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM service."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": (
                        "You are an expert in dataset quality assessment for AI/ML repositories. "
                        "Always respond with valid JSON only."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        response = self.session.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        try:
            # Clean response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            data = json.loads(response)
            
            # Validate required fields
            required_fields = [
                'has_data_validation', 'data_diversity_score', 'data_completeness',
                'has_dataset_card', 'data_fields_documented', 'example_usage',
                'data_source_credibility', 'bias_considerations', 'confidence_score'
            ]
            
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure scores are within valid range
            score_fields = ['data_diversity_score', 'data_completeness', 
                          'data_source_credibility', 'bias_considerations', 'confidence_score']
            for field in score_fields:
                data[field] = max(0.0, min(1.0, float(data[field])))
            
            return data
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            # Return default fallback
            return self._get_default_analysis()

    def _fallback_analysis(self, readme_content: str, 
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback analysis when LLM is unavailable.
        """
        readme_lower = readme_content.lower()
        
        # Rule-based detection
        has_validation = any(term in readme_lower for term in [
            'validation', 'preprocessing', 'cleaning', 'quality check'
        ])
        
        has_dataset_card = any(term in readme_lower for term in [
            'dataset card', 'data sheet', 'dataset documentation'
        ])
        
        data_fields_documented = any(term in readme_lower for term in [
            'schema', 'fields', 'columns', 'format', 'structure'
        ])
        
        example_usage = any(term in readme_lower for term in [
            'example', 'usage', 'how to use', 'tutorial'
        ])
        
        # Score based on keyword density
        diversity_keywords = ['diverse', 'balanced', 'multiple sources', 'variety']
        diversity_score = min(1.0, sum(readme_lower.count(kw) for kw in diversity_keywords) * 0.2)
        
        completeness_keywords = ['complete', 'comprehensive', 'full coverage']
        # Calculate score for completeness keywords
        completeness_score = min(
            1.0, 
            sum(readme_lower.count(kw) for kw in completeness_keywords) * 0.3
        )
        
        credibility_keywords = ['official', 'verified', 'peer-reviewed', 'published']
        # Calculate score for credibility keywords
        credibility_score = min(
            1.0, 
            sum(readme_lower.count(kw) for kw in credibility_keywords) * 0.25
        )
        
        bias_keywords = ['bias', 'limitation', 'fairness', 'ethical']
        bias_score = min(1.0, sum(readme_lower.count(kw) for kw in bias_keywords) * 0.3)
        
        return {
            'has_data_validation': has_validation,
            'data_diversity_score': diversity_score,
            'data_completeness': completeness_score,
            'has_dataset_card': has_dataset_card,
            'data_fields_documented': data_fields_documented,
            'example_usage': example_usage,
            'data_source_credibility': credibility_score,
            'bias_considerations': bias_score,
            'confidence_score': 0.6  # Lower confidence for rule-based
        }

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when all else fails."""
        return {
            'has_data_validation': False,
            'data_diversity_score': 0.0,
            'data_completeness': 0.0,
            'has_dataset_card': False,
            'data_fields_documented': False,
            'example_usage': False,
            'data_source_credibility': 0.0,
            'bias_considerations': 0.0,
            'confidence_score': 0.0
        }

    def analyze_implementation_assistance(self, code_content: str, 
                                        repo_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze code and repository for implementation assistance indicators.
        
        Args:
            code_content: Sample code content from the repository
            repo_context: Repository context information
            
        Returns:
            Dictionary containing implementation assistance analysis
        """
        if not self.api_key:
            return self._fallback_implementation_analysis(code_content, repo_context)

        prompt = f"""
Analyze the following code and repository context to evaluate implementation assistance quality:

Code Sample:
{code_content[:1500]}

Repository Context:
{json.dumps(repo_context, indent=2)[:800]}

Evaluate and return JSON with these fields:
{{
    "has_code_examples": boolean,
    "documentation_quality": float (0.0-1.0),
    "api_documentation": boolean,
    "tutorial_availability": boolean,
    "setup_instructions": boolean,
    "dependency_management": boolean,
    "error_handling_examples": boolean,
    "performance_guidance": boolean,
    "implementation_score": float (0.0-1.0)
}}
"""

        try:
            response = self._call_llm(prompt)
            return self._parse_implementation_response(response)
        except Exception as e:
            print(f"Implementation analysis failed: {e}")
            return self._fallback_implementation_analysis(code_content, repo_context)

    def _parse_implementation_response(self, response: str) -> Dict[str, Any]:
        """Parse implementation analysis response."""
        try:
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            data = json.loads(response)
            
            # Ensure numeric fields are in valid range
            numeric_fields = ['documentation_quality', 'implementation_score']
            for field in numeric_fields:
                if field in data:
                    data[field] = max(0.0, min(1.0, float(data[field])))
            
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing implementation response: {e}")
            return self._get_default_implementation_analysis()

    def _fallback_implementation_analysis(self, code_content: str, 
                                        repo_context: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based implementation analysis fallback."""
        code_lower = code_content.lower()
        
        has_examples = any(term in code_lower for term in [
            'example', 'demo', 'sample', 'tutorial'
        ])
        
        has_api_docs = any(term in code_lower for term in [
            'api', 'endpoint', 'function', 'method'
        ])
        
        has_setup = any(term in code_lower for term in [
            'install', 'setup', 'requirements', 'dependencies'
        ])
        
        has_error_handling = any(term in code_lower for term in [
            'try', 'except', 'error', 'exception'
        ])
        
        # Calculate documentation quality based on comments and docstrings
        lines = code_content.split('\n')
        # Count lines with comments or docstrings
        comment_markers = ('#', '"""', "'''", '//')
        comment_lines = sum(1 for line in lines if line.strip().startswith(comment_markers))
        doc_quality = min(1.0, comment_lines / max(1, len(lines)) * 3)
        
        implementation_score = (
            (0.2 if has_examples else 0) +
            (0.2 * doc_quality) +
            (0.15 if has_api_docs else 0) +
            (0.15 if has_setup else 0) +
            (0.1 if has_error_handling else 0) +
            0.2  # Base score
        )
        
        return {
            'has_code_examples': has_examples,
            'documentation_quality': doc_quality,
            'api_documentation': has_api_docs,
            'tutorial_availability': has_examples,
            'setup_instructions': has_setup,
            'dependency_management': has_setup,
            'error_handling_examples': has_error_handling,
            'performance_guidance': False,
            'implementation_score': min(1.0, implementation_score)
        }

    def _get_default_implementation_analysis(self) -> Dict[str, Any]:
        """Return default implementation analysis."""
        return {
            'has_code_examples': False,
            'documentation_quality': 0.0,
            'api_documentation': False,
            'tutorial_availability': False,
            'setup_instructions': False,
            'dependency_management': False,
            'error_handling_examples': False,
            'performance_guidance': False,
            'implementation_score': 0.0
        }
