#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Email dataset generation functions for AI-Phishing-Gen-Combined
"""

import random
import time
import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from utils import log_message, save_dataset, create_varied_scenario
from api_clients import (
    generate_single_email_openai,
    generate_single_email_anthropic
)
from data_processing import add_dataset_features, extract_email_metadata

def generate_email_dataset(
    num_samples=200,
    model="gpt-4.1-mini",
    scenario_type="phishing",
    batch_size=10,
    save_interval=10,
    max_retries=5,  # Increased retries
    phishing_scenarios=None,
    legitimate_scenarios=None,
    scenario_variations=None,
    project_dir=None,
    run_timestamp=None,
    log_path=None,
    client=None
):
    """Generate a dataset of emails with improved handling and recovery"""
    # Setup scenarios based on type
    if scenario_type == "phishing":
        scenarios = phishing_scenarios
        label = "phishing"
        is_phishing = True
    else:
        # Legitimate email scenarios
        scenarios = legitimate_scenarios
        label = "legitimate"
        is_phishing = False

    # Initialize empty dataset
    dataset = []

    log_message(f"\n{'-'*50}", log_path=log_path)
    log_message(f"STARTING {scenario_type.upper()} EMAIL GENERATION", log_path=log_path)
    log_message(f"Target: {num_samples} samples", log_path=log_path)
    log_message(f"Model: {model}", log_path=log_path)
    log_message(f"{'-'*50}", log_path=log_path)

    # Create a set to track already used scenarios to ensure diversity
    used_scenarios = set()

    # Try smaller batches if failures occur
    adaptive_batch_size = batch_size
    consecutive_failures = 0

    # Main generation loop
    with tqdm(total=num_samples, desc=f"Generating {scenario_type} emails") as pbar:
        samples_generated = 0

        while samples_generated < num_samples:
            # Adjust batch size based on failure rate
            if consecutive_failures > 3:
                old_batch_size = adaptive_batch_size
                adaptive_batch_size = max(5, adaptive_batch_size // 2)  # Reduce batch size but not below 5
                log_message(f"Too many failures, reducing batch size from {old_batch_size} to {adaptive_batch_size}", log_path=log_path)
                consecutive_failures = 0

            # Calculate number of samples for this batch
            current_batch_size = min(adaptive_batch_size, num_samples - samples_generated)

            log_message(f"\nStarting batch of {current_batch_size} samples", log_path=log_path)

            batch_success = 0
            batch_start_time = time.time()

            # Generate each sample in the batch
            for i in range(current_batch_size):
                overall_index = samples_generated + 1

                # Select and vary scenario - try to get unused scenarios first
                available_scenarios = [s for s in scenarios if s not in used_scenarios]

                # If we've used all scenarios, reset the tracking
                if not available_scenarios:
                    log_message("All scenarios have been used - resetting diversity tracking", print_to_console=False, log_path=log_path)
                    used_scenarios.clear()
                    available_scenarios = scenarios

                base_scenario = random.choice(available_scenarios)
                used_scenarios.add(base_scenario)

                scenario = create_varied_scenario(base_scenario, is_phishing, scenario_variations)
                log_message(f"Sample {overall_index}/{num_samples}: {scenario}", print_to_console=False, log_path=log_path)

                # Generate the email
                success, result_text = generate_single_email_openai(
                    scenario=scenario,
                    is_phishing=is_phishing,
                    model=model,
                    max_retries=max_retries,
                    client=client,
                    log_path=log_path
                )

                # Add to dataset based on result
                if success:
                    batch_success += 1
                    consecutive_failures = 0  # Reset failure counter

                    # Extract metadata from the generated text
                    metadata = extract_email_metadata(result_text)

                    dataset.append({
                        "text": result_text,
                        "source": model,
                        "label": label,
                        "type": "ai_generated",
                        "scenario": scenario,
                        "base_scenario": base_scenario,
                        "subject": metadata["subject"],
                        "sender": metadata["sender"],
                        "recipient": metadata["recipient"],
                        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    samples_generated += 1
                    pbar.update(1)
                else:
                    # Track failures
                    consecutive_failures += 1

                    # Use a placeholder for failed generations
                    log_message(f"  ✗ Failed to generate sample after {max_retries} attempts", log_path=log_path)
                    dataset.append({
                        "text": f"[GENERATION FAILED] {scenario}",
                        "source": "failed",
                        "label": label,
                        "type": "ai_generated",
                        "scenario": scenario,
                        "base_scenario": base_scenario,
                        "subject": "",
                        "sender": "",
                        "recipient": "",
                        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": "Generation failed after retries"
                    })
                    samples_generated += 1
                    pbar.update(1)

                # Save at intervals
                if samples_generated % save_interval == 0 or samples_generated == num_samples:
                    save_dataset(
                        dataset,
                        f"fresh_{scenario_type}_emails_progress_{run_timestamp}.csv",
                        project_dir,
                        run_timestamp,
                        log_path
                    )

            # Log batch results
            batch_time = time.time() - batch_start_time
            log_message(f"Batch completed: {batch_success}/{current_batch_size} successful in {batch_time:.2f}s", log_path=log_path)

            # If success rate is good, increase batch size back up (but not beyond original)
            if batch_success > 0.8 * current_batch_size and adaptive_batch_size < batch_size:
                adaptive_batch_size = min(batch_size, adaptive_batch_size * 2)
                log_message(f"Good success rate, increasing batch size to {adaptive_batch_size}", log_path=log_path)

            # Pause between batches (longer pause if we had failures)
            if samples_generated < num_samples:
                pause_time = 10 + random.randint(0, 5)
                if consecutive_failures > 0:
                    pause_time = 20 + random.randint(0, 10)  # Longer pause if failures
                log_message(f"Pausing for {pause_time}s between batches...", log_path=log_path)
                time.sleep(pause_time)

    # Final save
    save_dataset(dataset, f"fresh_{scenario_type}_emails_{run_timestamp}.csv", project_dir, run_timestamp, log_path)

    log_message(f"\n✓ Dataset generation complete: {len(dataset)} samples", log_path=log_path)
    return pd.DataFrame(dataset)