#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI-Phishing-Gen-Combined

Main execution script for generating phishing and legitimate email datasets
"""

import os
from datetime import datetime
import openai
import anthropic
from utils import log_message
from data_generation import run_email_generation
from config import (
    phishing_scenarios,
    legitimate_scenarios,
    scenario_variations
)

# This is the main execution block
if __name__ == '__main__':
    # Let the user choose between OpenAI-only or combined generation
    print("\nEmail Dataset Generation Options:")
    print("1. OpenAI only (1,500 phishing + 1,500 legitimate emails)")
    print("2. Combined: OpenAI + Anthropic (3,000 phishing + 3,000 legitimate emails)")

    choice = input("\nSelect an option (1 or 2): ")
    use_combined = (choice == "2")

    run_email_generation(use_combined=use_combined)