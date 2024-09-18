from __future__ import annotations

import re
import warnings


def load_as_score(text):
    """
    validate and returns given text as score
    """

    pattern = r"^[\d.]+$"
    pattern = r"^[\d.]+"#$"
    if not re.match(pattern, text):
        warnings.warn("Invalid score")
        score = 0.0
    else:
        score = eval(text)

    return score

import re
import warnings

def parse_regex(text, pattern): 

    matches = re.findall(pattern, text)
    
    if not matches: 
        raise ValueError(f"Invalid text for {text} pattern: {pattern}")
    else: 
        #print("TEST")
        #print(matches[0])
        return matches[0]

def load_as_score(text):
    """
    Validate and return the first occurrence of a pattern (digits and periods) in the given text as score.
    If no valid pattern is found, warn and return 0.0.
    """

    pattern = r"^[\d.]+"
    matches = re.findall(pattern, text)

    if not matches:
        warnings.warn("Invalid score")
        return 0.0
    else:
        # Assuming you want to take the first match and evaluate it as a score
        try:
            score = eval(matches[0])
        except Exception:
            warnings.warn("Invalid score format")
            print(matches)
            return 0.0

    return score
