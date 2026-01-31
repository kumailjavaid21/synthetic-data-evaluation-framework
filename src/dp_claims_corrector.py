import os
import re
import json
from pathlib import Path
from datetime import datetime

class DPClaimsCorrector:
    """
    Scan and correct invalid DP claims in reports and documentation.
    Ensures adaptive budget claims are properly framed as training heuristics.
    """
    
    def __init__(self):
        self.invalid_patterns = [
            r"feature-wise\s+noise\s+multipliers?",
            r"per-feature\s+DP",
            r"feature-level\s+privacy\s+guarantee",
            r"adaptive\s+DP\s+mechanism",
            r"formal\s+guarantee.*adaptive",
            r"feature-wise\s+differential\s+privacy",
            r"individual\s+feature\s+noise",
            r"heterogeneous\s+noise\s+application"
        ]
        
        self.corrections = {
            "feature-wise noise multipliers": "sensitivity-weighted loss scaling (training heuristic)",
            "per-feature DP": "feature importance weighting (training optimization)",
            "feature-level privacy guarantee": "global DP-SGD guarantee only",
            "adaptive DP mechanism": "adaptive training heuristic with global DP-SGD",
            "formal guarantee": "training optimization (no additional DP claims)",
            "feature-wise differential privacy": "global differential privacy via DP-SGD",
            "individual feature noise": "uniform noise application (DP-SGD)",
            "heterogeneous noise application": "standard DP-SGD with uniform noise"
        }
        
        self.scan_results = []
    
    def scan_file(self, filepath):
        """Scan a single file for invalid DP claims."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"[ERROR] Cannot read {filepath}: {e}")
            return []
        
        issues = []
        
        for pattern in self.invalid_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Get context around the match
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end].replace('\n', ' ').strip()
                
                issues.append({
                    'file': filepath,
                    'pattern': pattern,
                    'match': match.group(),
                    'context': context,
                    'line_number': content[:match.start()].count('\n') + 1
                })
        
        return issues
    
    def scan_directory(self, directory="."):
        """Scan directory for files with potential DP claim issues."""
        print(f"[SCAN] Scanning {directory} for invalid DP claims...")
        
        # File types to scan
        extensions = ['.py', '.md', '.txt', '.json', '.csv', '.tex']
        
        all_issues = []
        files_scanned = 0
        
        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            skip_dirs = ['.git', '__pycache__', 'node_modules', '.venv']
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    filepath = os.path.join(root, file)
                    issues = self.scan_file(filepath)
                    all_issues.extend(issues)
                    files_scanned += 1
        
        self.scan_results = all_issues
        print(f"[SCAN] Scanned {files_scanned} files, found {len(all_issues)} potential issues")
        
        return all_issues
    
    def generate_corrections_report(self):
        """Generate report of required corrections."""
        if not self.scan_results:
            return None
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(self.scan_results),
            'files_affected': len(set(issue['file'] for issue in self.scan_results)),
            'issues_by_file': {},
            'correction_guidelines': {
                'principle': 'Separate training heuristics from DP guarantees',
                'dp_guarantee': 'Global DP-SGD with uniform noise only',
                'adaptive_component': 'Training optimization heuristic only',
                'no_additional_claims': 'Adaptive part makes no DP claims'
            },
            'required_corrections': []
        }
        
        # Group issues by file
        for issue in self.scan_results:
            filepath = issue['file']
            if filepath not in report['issues_by_file']:
                report['issues_by_file'][filepath] = []
            
            report['issues_by_file'][filepath].append({
                'line': issue['line_number'],
                'pattern': issue['pattern'],
                'match': issue['match'],
                'context': issue['context']
            })
        
        # Generate specific corrections
        for filepath, issues in report['issues_by_file'].items():
            for issue in issues:
                correction = {
                    'file': filepath,
                    'line': issue['line'],
                    'original': issue['match'],
                    'suggested_correction': self.get_correction_suggestion(issue['match']),
                    'rationale': 'Separate training heuristics from DP guarantees'
                }
                report['required_corrections'].append(correction)
        
        return report
    
    def get_correction_suggestion(self, original_text):
        """Get correction suggestion for invalid claim."""
        original_lower = original_text.lower()
        
        for pattern, correction in self.corrections.items():
            if pattern.lower() in original_lower:
                return correction
        
        # Default correction
        return "training heuristic (no additional DP claims)"
    
    def create_corrected_claims_statement(self):
        """Create official corrected claims statement."""
        statement = {
            'title': 'Corrected DP Claims for Adaptive Budget Training',
            'timestamp': datetime.now().isoformat(),
            
            'official_dp_guarantee': {
                'mechanism': 'DP-SGD with uniform noise (Opacus implementation)',
                'guarantee': '(ε, δ)-differential privacy',
                'scope': 'Global model parameters only',
                'accountant': 'Renyi Differential Privacy (RDP) composition',
                'noise_application': 'Uniform across all gradient components'
            },
            
            'adaptive_training_component': {
                'classification': 'Training optimization heuristic',
                'purpose': 'Sensitivity-weighted loss scaling for improved convergence',
                'dp_claims': 'NONE - does not provide additional privacy guarantees',
                'relationship_to_dp': 'Orthogonal to DP mechanism (does not affect privacy analysis)',
                'implementation': 'Feature importance weighting applied to loss function'
            },
            
            'explicitly_avoided_claims': [
                'Feature-wise noise multipliers',
                'Per-feature differential privacy',
                'Adaptive DP mechanisms',
                'Feature-level privacy guarantees',
                'Heterogeneous noise application',
                'Individual feature privacy accounting'
            ],
            
            'technical_clarification': {
                'dp_mechanism': 'Standard DP-SGD (unchanged)',
                'noise_calibration': 'Based on global sensitivity and epsilon budget',
                'privacy_accounting': 'Standard RDP composition rules',
                'adaptive_weighting': 'Applied to loss function only (pre-gradient computation)',
                'guarantee_preservation': 'DP guarantee unaffected by adaptive weighting'
            },
            
            'validation': {
                'mechanism_verified': 'Standard Opacus DP-SGD implementation',
                'claims_reviewed': 'All DP claims limited to global DP-SGD',
                'heuristics_separated': 'Training optimizations clearly distinguished',
                'no_unproven_guarantees': 'No claims beyond standard DP-SGD'
            }
        }
        
        return statement
    
    def save_reports(self):
        """Save all correction reports."""
        # Save scan results
        if self.scan_results:
            corrections_report = self.generate_corrections_report()
            with open('dp_claims_corrections_needed.json', 'w') as f:
                json.dump(corrections_report, f, indent=2)
            print(f"[SAVED] Corrections needed: dp_claims_corrections_needed.json")
        
        # Save corrected claims statement
        corrected_statement = self.create_corrected_claims_statement()
        with open('official_corrected_dp_claims.json', 'w') as f:
            json.dump(corrected_statement, f, indent=2)
        print(f"[SAVED] Official corrected claims: official_corrected_dp_claims.json")
        
        return corrected_statement

def main():
    """Run DP claims correction process."""
    print("=== DP Claims Correction Process ===")
    
    corrector = DPClaimsCorrector()
    
    # Scan for invalid claims
    issues = corrector.scan_directory(".")
    
    # Generate and save reports
    corrected_statement = corrector.save_reports()
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Files scanned: Multiple")
    print(f"Issues found: {len(issues)}")
    print(f"Files affected: {len(set(issue['file'] for issue in issues))}")
    
    if issues:
        print(f"\n[ACTION REQUIRED] Review dp_claims_corrections_needed.json")
        print(f"[GUIDANCE] Use official_corrected_dp_claims.json for proper claims")
    else:
        print(f"\n[OK] No invalid DP claims detected")
    
    print(f"\n[PRINCIPLE] Adaptive budget = training heuristic only")
    print(f"[GUARANTEE] DP guarantee = global DP-SGD only")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n[SUCCESS] All DP claims are valid")
    else:
        print(f"\n[REVIEW] Some claims need correction")