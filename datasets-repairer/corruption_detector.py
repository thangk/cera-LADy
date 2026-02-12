"""
Corruption Detection Utilities
=============================

Advanced corruption pattern detection and analysis for LLM-generated XML datasets.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class CorruptionReport:
    """Detailed corruption analysis report"""
    file_path: str
    is_valid_xml: bool
    total_issues: int
    corruption_patterns: Dict[str, int]
    severity_level: str  # 'minimal', 'moderate', 'severe', 'critical'
    repairability: str   # 'easy', 'moderate', 'difficult', 'requires_regeneration'
    sample_errors: List[str]

class CorruptionDetector:
    """
    Advanced corruption detection for LLM-generated XML datasets.
    
    Identifies and categorizes various corruption patterns that can occur
    when LLMs generate XML data, including structural issues, malformed
    tags, and content problems.
    """
    
    def __init__(self):
        self.severity_thresholds = {
            'minimal': 5,      # 1-5 issues
            'moderate': 20,    # 6-20 issues  
            'severe': 50,      # 21-50 issues
            'critical': float('inf')  # 50+ issues
        }
    
    def analyze_file(self, file_path: str) -> CorruptionReport:
        """Comprehensive corruption analysis of an XML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, FileNotFoundError) as e:
            return CorruptionReport(
                file_path=file_path,
                is_valid_xml=False,
                total_issues=1,
                corruption_patterns={'file_read_error': 1},
                severity_level='critical',
                repairability='requires_regeneration',
                sample_errors=[f"File read error: {e}"]
            )
        
        # Check if file is valid XML
        is_valid_xml = self._is_valid_xml(file_path)
        
        # Detect corruption patterns
        corruption_patterns = self._detect_patterns(content)
        total_issues = sum(corruption_patterns.values())
        
        # Determine severity
        severity_level = self._assess_severity(total_issues)
        
        # Assess repairability
        repairability = self._assess_repairability(corruption_patterns, total_issues)
        
        # Get sample errors
        sample_errors = self._get_sample_errors(content, corruption_patterns)
        
        return CorruptionReport(
            file_path=file_path,
            is_valid_xml=is_valid_xml,
            total_issues=total_issues,
            corruption_patterns=corruption_patterns,
            severity_level=severity_level,
            repairability=repairability,
            sample_errors=sample_errors
        )
    
    def _is_valid_xml(self, file_path: str) -> bool:
        """Check if file is valid XML"""
        try:
            ET.parse(file_path)
            return True
        except ET.ParseError:
            return False
    
    def _detect_patterns(self, content: str) -> Dict[str, int]:
        """Detect all corruption patterns in content"""
        patterns = {}
        
        # Split closing tags
        patterns['split_closing_tags'] = len(re.findall(r'</\w+$', content, re.MULTILINE))
        
        # Malformed opinion tags - specific malformed patterns only
        # NOTE: Valid tags are </Opinion> and </Opinions> - do NOT count these as malformed
        malformed_opinion_patterns = [
            r'</O-pinions>',          # Hyphenated O-pinions
            r'</Opinces>',            # Typo: Opinces 
            r'</Opinionss>',          # Double s: Opinionss
            r'</Opin>',               # Truncated: Opin
            r'</Opini>',              # Truncated: Opini
            r'</Opinio>',             # Truncated: Opinio
        ]
        # Explicitly exclude valid </Opinion> and </Opinions> from detection
        patterns['malformed_opinions'] = sum(
            len(re.findall(pattern, content)) for pattern in malformed_opinion_patterns
        )
        
        # Missing attribute spaces
        patterns['missing_spaces'] = len(re.findall(r'="[^"]*"[a-zA-Z]', content))
        
        # Orphaned closing tags - only detect tags that appear without proper opening context
        # NOTE: This should only detect truly orphaned tags, not valid closing tags
        orphaned_patterns = []  # Removed overly broad patterns that flag valid XML structure
        patterns['orphaned_tags'] = sum(
            len(re.findall(pattern, content, re.MULTILINE)) for pattern in orphaned_patterns
        )
        
        # Unmatched sentence tags
        opening_sentences = len(re.findall(r'<sentence[^>]*>', content))
        closing_sentences = len(re.findall(r'</sentence>', content))
        patterns['unmatched_sentences'] = abs(opening_sentences - closing_sentences)
        
        # Invalid XML characters
        patterns['invalid_characters'] = len(re.findall(
            r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFDCF\uFDF0-\uFFFD]', content
        ))
        
        # Malformed attributes
        patterns['malformed_attributes'] = len(re.findall(r' a category=', content))
        
        # Incomplete tags (tags that don't close properly)
        patterns['incomplete_tags'] = len(re.findall(r'<[^>]*[^/>]$', content, re.MULTILINE))
        
        # Nested structural issues
        patterns['structural_nesting'] = self._detect_nesting_issues(content)
        
        # Encoding issues
        patterns['encoding_issues'] = self._detect_encoding_issues(content)
        
        return patterns
    
    def _detect_nesting_issues(self, content: str) -> int:
        """Detect XML nesting and structural issues"""
        issues = 0
        
        # Check for common nesting problems
        lines = content.splitlines()
        in_reviews = False
        in_review = False
        in_sentences = False
        in_sentence = False
        
        for line in lines:
            stripped = line.strip()
            
            # Track nesting state
            if '<Reviews>' in stripped:
                in_reviews = True
            elif '</Reviews>' in stripped:
                in_reviews = False
            elif '<Review' in stripped:
                in_review = True
            elif '</Review>' in stripped:
                in_review = False
            elif '<sentences>' in stripped:
                in_sentences = True
            elif '</sentences>' in stripped:
                in_sentences = False
            elif '<sentence' in stripped:
                in_sentence = True
            elif '</sentence>' in stripped:
                in_sentence = False
            
            # Check for content outside proper structure
            if '<Opinion' in stripped and not (in_reviews and in_review and in_sentences and in_sentence):
                issues += 1
            if '<Opinions>' in stripped and not (in_reviews and in_review and in_sentences and in_sentence):
                issues += 1
        
        return issues
    
    def _detect_encoding_issues(self, content: str) -> int:
        """Detect potential encoding-related issues"""
        issues = 0
        
        # Check for common encoding artifacts
        encoding_artifacts = [
            '�',  # Replacement character
            '\x00',  # Null character
            '\ufeff',  # BOM
        ]
        
        for artifact in encoding_artifacts:
            issues += content.count(artifact)
        
        return issues
    
    def _assess_severity(self, total_issues: int) -> str:
        """Assess corruption severity level"""
        for level, threshold in self.severity_thresholds.items():
            if total_issues <= threshold:
                return level
        return 'critical'
    
    def _assess_repairability(self, corruption_patterns: Dict[str, int], total_issues: int) -> str:
        """Assess how difficult the file would be to repair"""
        # Critical patterns that are hard to repair
        critical_patterns = ['structural_nesting', 'encoding_issues', 'unmatched_sentences']
        critical_issues = sum(corruption_patterns.get(pattern, 0) for pattern in critical_patterns)
        
        if total_issues == 0:
            return 'no_repair_needed'
        elif critical_issues > 10:
            return 'requires_regeneration'
        elif critical_issues > 5 or total_issues > 50:
            return 'difficult'
        elif total_issues > 10:
            return 'moderate'
        else:
            return 'easy'
    
    def _get_sample_errors(self, content: str, corruption_patterns: Dict[str, int]) -> List[str]:
        """Get sample error instances for each corruption type found"""
        samples = []
        
        # Sample split closing tags
        if corruption_patterns.get('split_closing_tags', 0) > 0:
            matches = re.finditer(r'</\w+$', content, re.MULTILINE)
            for match in list(matches)[:2]:  # First 2 samples
                line_num = content[:match.start()].count('\n') + 1
                samples.append(f"Line {line_num}: Split closing tag '{match.group()}'")
        
        # Sample malformed opinions
        if corruption_patterns.get('malformed_opinions', 0) > 0:
            malformed_patterns = [r'</O-pinions>', r'</Opinces>', r'</Opinionss>']
            for pattern in malformed_patterns:
                matches = re.finditer(pattern, content)
                for match in list(matches)[:1]:  # One sample per pattern
                    line_num = content[:match.start()].count('\n') + 1
                    samples.append(f"Line {line_num}: Malformed tag '{match.group()}'")
        
        # Sample missing spaces
        if corruption_patterns.get('missing_spaces', 0) > 0:
            matches = re.finditer(r'="[^"]*"[a-zA-Z]', content)
            for match in list(matches)[:2]:
                line_num = content[:match.start()].count('\n') + 1
                samples.append(f"Line {line_num}: Missing space in attributes '{match.group()}'")
        
        # Limit to 5 samples total
        return samples[:5]
    
    def batch_analyze(self, file_paths: List[str]) -> Dict[str, CorruptionReport]:
        """Analyze multiple files and return detailed reports"""
        reports = {}
        for file_path in file_paths:
            reports[file_path] = self.analyze_file(file_path)
        return reports
    
    def get_summary_stats(self, reports: Dict[str, CorruptionReport]) -> Dict[str, any]:
        """Generate summary statistics from multiple corruption reports"""
        if not reports:
            return {}
        
        total_files = len(reports)
        valid_files = sum(1 for r in reports.values() if r.is_valid_xml)
        
        severity_counts = {}
        repairability_counts = {}
        total_issues = 0
        
        for report in reports.values():
            # Count severities
            severity_counts[report.severity_level] = severity_counts.get(report.severity_level, 0) + 1
            
            # Count repairabilities
            repairability_counts[report.repairability] = repairability_counts.get(report.repairability, 0) + 1
            
            # Sum total issues
            total_issues += report.total_issues
        
        return {
            'total_files': total_files,
            'valid_xml_files': valid_files,
            'corrupted_files': total_files - valid_files,
            'total_issues': total_issues,
            'average_issues_per_file': total_issues / total_files if total_files > 0 else 0,
            'severity_distribution': severity_counts,
            'repairability_distribution': repairability_counts
        }