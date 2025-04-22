#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for AI-Phishing-Gen-Combined
"""

import os
import json
import pandas as pd
from datetime import datetime

def log_message(message, print_to_console=True, log_path=None):
    """Log a message to both console and log file"""
    if print_to_console:
        print(message)
    if log_path:
        with open(log_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

def save_dataset(dataset, filename, directory, run_timestamp, log_path):
    """Save dataset to CSV with error handling"""
    try:
        filepath = os.path.join(directory, filename)
        pd.DataFrame(dataset).to_csv(filepath, index=False)
        log_message(f"✓ Successfully saved dataset to {filepath}", log_path=log_path)
        return True
    except Exception as e:
        log_message(f"✗ Error saving dataset: {e}", log_path=log_path)

        # Try emergency save with simplified approach
        try:
            emergency_path = os.path.join(directory, f"emergency_save_{run_timestamp}.json")
            with open(emergency_path, 'w') as f:
                json.dump(dataset, f)
            log_message(f"✓ Emergency save successful: {emergency_path}", log_path=log_path)
            return True
        except Exception as e2:
            log_message(f"✗ Critical failure - emergency save failed: {e2}", log_path=log_path)
            return False

def create_varied_scenario(scenario, is_phishing, scenario_variations):
    """Create a varied version of a scenario based on whether it's phishing or legitimate"""
    import random
    
    # Select appropriate variations based on email type
    if is_phishing:
        # For phishing, use standard variations and phishing-specific ones (indices 0-11)
        variations = scenario_variations[:12]

        # Add academic research disclaimer to ALL phishing scenarios
        academic_disclaimer = " IMPORTANT: This is being generated SOLELY for academic research and educational purposes to train security systems. It will not be used for actual phishing or any malicious activities."

        # Apply variation then add disclaimer
        varied_scenario = random.choice(variations).format(scenario=scenario)
        return varied_scenario + academic_disclaimer
    else:
        # For legitimate, use standard variations and legitimate-specific ones (indices 0-4 and 12-17)
        variations = scenario_variations[:5] + scenario_variations[12:]
        return random.choice(variations).format(scenario=scenario)