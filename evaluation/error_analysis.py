import difflib
from typing import List, Tuple, Dict

class ErrorAnalyzer:
    def __init__(self):
        pass
    
    def analyze_errors(self, original: str, cpg_output: str, llm_output: str) -> Dict:
        """
        Analyze errors in both paraphrase outputs.
        """
        analysis = {
            'cpg': self._analyze_single(original, cpg_output, 'CPG'),
            'llm': self._analyze_single(original, llm_output, 'LLM'),
            'comparison': self._compare_errors(cpg_output, llm_output)
        }
        
        return analysis
    
    def _analyze_single(self, original: str, paraphrase: str, system_name: str) -> Dict:
        """
        Analyze errors for a single system.
        """
        orig_words = original.lower().split()
        para_words = paraphrase.lower().split()
        
        # Find common words
        common_words = set(orig_words) & set(para_words)
        unique_to_orig = set(orig_words) - set(para_words)
        unique_to_para = set(para_words) - set(orig_words)
        
        # Calculate lexical overlap
        overlap_ratio = len(common_words) / max(len(set(orig_words)), 1)
        
        # Check for repetition
        repetition_score = self._calculate_repetition(paraphrase)
        
        # Check sentence structure changes
        structure_change = self._analyze_structure_change(original, paraphrase)
        
        return {
            'system': system_name,
            'lexical_overlap': overlap_ratio,
            'unique_words_lost': len(unique_to_orig),
            'new_words_added': len(unique_to_para),
            'repetition_score': repetition_score,
            'structure_change': structure_change,
            'key_issues': self._identify_key_issues(original, paraphrase)
        }
    
    def _calculate_repetition(self, text: str) -> float:
        """Calculate repetition score (lower is better)."""
        words = text.split()
        if len(words) < 2:
            return 0
        
        # Count consecutive repeated words
        repeats = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repeats += 1
        
        return repeats / len(words)
    
    def _analyze_structure_change(self, original: str, paraphrase: str) -> str:
        """Analyze how sentence structure changed."""
        orig_sentences = original.split('.')
        para_sentences = paraphrase.split('.')
        
        if abs(len(orig_sentences) - len(para_sentences)) > 2:
            return "major_restructuring"
        elif abs(len(orig_sentences) - len(para_sentences)) > 0:
            return "moderate_restructuring"
        else:
            return "minimal_restructuring"
    
    def _identify_key_issues(self, original: str, paraphrase: str) -> List[str]:
        """Identify specific issues with the paraphrase."""
        issues = []
        
        # Check for meaning drift indicators
        orig_keywords = self._extract_keywords(original)
        para_keywords = self._extract_keywords(paraphrase)
        
        missing_keywords = orig_keywords - para_keywords
        if len(missing_keywords) > 2:
            issues.append(f"Missing key concepts: {list(missing_keywords)[:3]}")
        
        # Check for simplification
        orig_avg_word_len = sum(len(w) for w in original.split()) / len(original.split())
        para_avg_word_len = sum(len(w) for w in paraphrase.split()) / len(paraphrase.split())
        
        if para_avg_word_len < orig_avg_word_len * 0.8:
            issues.append("Over-simplification of vocabulary")
        
        # Check for hallucinations
        if self._detect_hallucinations(original, paraphrase):
            issues.append("Possible hallucinated content")
        
        return issues
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> set:
        """Extract top keywords from text."""
        from collections import Counter
        import re
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Get most common words
        common_words = Counter(words).most_common(top_n)
        return {word for word, count in common_words}
    
    def _detect_hallucinations(self, original: str, paraphrase: str) -> bool:
        """Simple hallucination detection."""
        # Check for new named entities or specific facts
        # This is a simplified version - in practice, use NER
        orig_uppercase = sum(1 for c in original if c.isupper())
        para_uppercase = sum(1 for c in paraphrase if c.isupper())
        
        # If paraphrase has significantly more uppercase (possible new entities)
        if para_uppercase > orig_uppercase * 1.5:
            return True
        
        return False
    
    def _compare_errors(self, cpg_output: str, llm_output: str) -> Dict:
        """Compare errors between CPG and LLM outputs."""
        comparison = {
            'common_errors': [],
            'cpg_specific': [],
            'llm_specific': []
        }
        
        # Compare at word level
        cpg_words = set(cpg_output.lower().split())
        llm_words = set(llm_output.lower().split())
        
        # Check for shared uncommon patterns
        if self._has_repetition(cpg_output) and self._has_repetition(llm_output):
            comparison['common_errors'].append("Both outputs contain repetition")
        
        # System-specific issues
        if len(cpg_words) < len(llm_words) * 0.7:
            comparison['cpg_specific'].append("CPG output has significantly less lexical diversity")
        
        if self._has_short_sentences(llm_output):
            comparison['llm_specific'].append("LLM output contains very short sentences")
        
        return comparison
    
    def _has_repetition(self, text: str) -> bool:
        """Check if text has obvious repetition."""
        words = text.split()
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
        return False
    
    def _has_short_sentences(self, text: str, threshold: int = 3) -> bool:
        """Check if text has many very short sentences."""
        sentences = text.split('.')
        short_sentences = sum(1 for s in sentences if len(s.split()) <= threshold)
        return short_sentences > len(sentences) * 0.3
    
    def print_analysis(self, analysis: Dict):
        """Print formatted error analysis."""
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        for system in ['cpg', 'llm']:
            sys_data = analysis[system]
            print(f"\n{sys_data['system']} Analysis:")
            print(f"  Lexical Overlap: {sys_data['lexical_overlap']:.2%}")
            print(f"  Unique Words Lost: {sys_data['unique_words_lost']}")
            print(f"  New Words Added: {sys_data['new_words_added']}")
            print(f"  Repetition Score: {sys_data['repetition_score']:.3f}")
            print(f"  Structure Change: {sys_data['structure_change']}")
            
            if sys_data['key_issues']:
                print(f"  Key Issues:")
                for issue in sys_data['key_issues']:
                    print(f"    • {issue}")
        
        print("\nComparison:")
        comp = analysis['comparison']
        if comp['common_errors']:
            print("  Common Errors:")
            for error in comp['common_errors']:
                print(f"    • {error}")
        
        if comp['cpg_specific']:
            print("  CPG-Specific Issues:")
            for issue in comp['cpg_specific']:
                print(f"    • {issue}")
        
        if comp['llm_specific']:
            print("  LLM-Specific Issues:")
            for issue in comp['llm_specific']:
                print(f"    • {issue}")