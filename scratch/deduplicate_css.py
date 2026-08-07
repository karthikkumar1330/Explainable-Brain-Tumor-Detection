import re

def parse_css_properties(rules_str):
    # Parse properties from a CSS declaration block
    properties = {}
    # Split by semicolon, but ignore semicolons inside parentheses (e.g. gradients)
    # A simple way is to match property: value
    # We can split by ; and then clean up
    raw_rules = rules_str.split(';')
    for rule in raw_rules:
        rule = rule.strip()
        if not rule:
            continue
        if ':' in rule:
            prop, val = rule.split(':', 1)
            properties[prop.strip()] = val.strip()
    return properties

def format_css_properties(properties):
    return ";\n    ".join(f"{p}: {v}" for p, v in properties.items()) + ";"

def deduplicate_css(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # State machine to extract top-level blocks
    blocks = []
    i = 0
    n = len(content)
    
    while i < n:
        # Skip whitespaces
        if content[i].isspace():
            i += 1
            continue
            
        # Parse comment
        if content[i:i+2] == "/*":
            end_idx = content.find("*/", i+2)
            if end_idx == -1:
                blocks.append(('comment', content[i:]))
                break
            else:
                blocks.append(('comment', content[i:end_idx+2]))
                i = end_idx + 2
                continue
                
        # Parse block (find next '{' and its matching '}')
        brace_level = 0
        j = i
        start_brace = -1
        while j < n:
            if content[j] == '{':
                if brace_level == 0:
                    start_brace = j
                brace_level += 1
            elif content[j] == '}':
                brace_level -= 1
                if brace_level == 0:
                    break
            j += 1
            
        if start_brace != -1:
            selector = content[i:start_brace].strip()
            block_content = content[start_brace+1:j].strip()
            
            # Determine if it's a special block (media query, keyframes)
            if selector.startswith('@media') or selector.startswith('@keyframes') or selector.startswith('@font-face') or selector.startswith('@import'):
                blocks.append(('special', selector, content[start_brace:j+1]))
            else:
                blocks.append(('rule', selector, block_content))
            i = j + 1
        else:
            # Trailing text or invalid CSS
            trailing = content[i:].strip()
            if trailing:
                blocks.append(('trailing', trailing))
            break

    # Now let's analyze standard rule blocks
    # Group them by selector (exact match)
    selector_indices = {}
    for idx, block in enumerate(blocks):
        if block[0] == 'rule':
            selector = block[1]
            if selector not in selector_indices:
                selector_indices[selector] = []
            selector_indices[selector].append(idx)
            
    # For any selector declared multiple times:
    # 1. Merge their properties (later overrides earlier)
    # 2. Keep the last block index, and set earlier block indices to None (removed)
    for selector, indices in selector_indices.items():
        if len(indices) > 1:
            # Merge properties
            merged_properties = {}
            for idx in indices:
                block_properties = parse_css_properties(blocks[idx][2])
                merged_properties.update(block_properties)
                
            # Update the last block content
            last_idx = indices[-1]
            new_rules_str = format_css_properties(merged_properties)
            blocks[last_idx] = ('rule', selector, new_rules_str)
            
            # Nullify earlier blocks
            for idx in indices[:-1]:
                blocks[idx] = None

    # Reconstruct the CSS
    new_css_parts = []
    for block in blocks:
        if block is None:
            continue
        if block[0] == 'comment':
            new_css_parts.append(block[1])
        elif block[0] == 'rule':
            selector, rules = block[1], block[2]
            new_css_parts.append(f"{selector} {{\n    {rules}\n}}")
        elif block[0] == 'special':
            selector, body = block[1], block[2]
            # Special blocks might contain raw css, write selector + body
            new_css_parts.append(f"{selector} {body}")
        elif block[0] == 'trailing':
            new_css_parts.append(block[1])
            
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(new_css_parts))
        
    print("Deduplication completed!")

if __name__ == "__main__":
    deduplicate_css(
        r"d:\BrainTumorProject\UNeXt-pytorch\ui_system\design_system.css",
        r"d:\BrainTumorProject\UNeXt-pytorch\ui_system\design_system.css"
    )
