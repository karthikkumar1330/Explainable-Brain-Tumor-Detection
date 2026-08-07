import re
from collections import defaultdict

def analyze_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple regex to strip comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    # Regex to find CSS blocks: selectors { rules }
    # Note: this is a simple regex that doesn't handle nested media queries perfectly, but works for top-level selectors.
    matches = re.findall(r'([^{]+)\{([^}]+)\}', content)
    
    selectors_seen = defaultdict(list)
    for index, (selector_str, rules_str) in enumerate(matches):
        # Clean selectors
        selectors = [s.strip() for s in selector_str.split(',')]
        rules = rules_str.strip()
        for sel in selectors:
            if sel:
                selectors_seen[sel].append((index, rules))
                
    duplicates = {k: v for k, v in selectors_seen.items() if len(v) > 1}
    
    print(f"--- CSS ANALYSIS FOR {file_path} ---")
    print(f"Total unique selectors: {len(selectors_seen)}")
    print(f"Selectors declared multiple times: {len(duplicates)}")
    
    if duplicates:
        print("\nDetails of duplicate selectors:")
        for sel, occurrences in duplicates.items():
            print(f"\nSelector: '{sel}' ({len(occurrences)} times)")
            for idx, rules in occurrences:
                # print a snippet of rules
                rules_snippet = rules.replace('\n', ' ').strip()
                if len(rules_snippet) > 80:
                    rules_snippet = rules_snippet[:77] + "..."
                print(f"  - Block {idx}: {rules_snippet}")

if __name__ == "__main__":
    analyze_css(r"d:\BrainTumorProject\UNeXt-pytorch\ui_system\design_system.css")
