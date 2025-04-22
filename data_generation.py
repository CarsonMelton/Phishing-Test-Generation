#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
High-level data generation functions for AI-Phishing-Gen-Combined
"""

import os
import pandas as pd
from datetime import datetime
import random
from openai import OpenAI
import anthropic

from utils import log_message, save_dataset
from data_processing import add_dataset_features
from generators import generate_email_dataset, generate_email_dataset_combined
from config import (
    phishing_scenarios,
    legitimate_scenarios,
    scenario_variations,
    DEFAULT_CONFIG
)

def generate_fresh_datasets(
    phishing_samples=1500,
    legitimate_samples=1500,
    model="gpt-4.1-mini",
    batch_size=10,  # Reduced default batch size
    max_retries=5,  # Increased retries
    phishing_scenarios=None,
    legitimate_scenarios=None,
    scenario_variations=None,
    project_dir=None,
    run_timestamp=None,
    log_path=None,
    client=None
):
    """Generate completely new datasets of phishing and legitimate emails with improved resilience"""
    log_message("\n" + "="*50, log_path=log_path)
    log_message("STARTING FRESH DATASET GENERATION WITH IMPROVED HANDLING", log_path=log_path)
    log_message(f"Target: {phishing_samples} phishing + {legitimate_samples} legitimate emails", log_path=log_path)
    log_message(f"Model: {model}", log_path=log_path)
    log_message(f"Starting batch size: {batch_size} (will adapt based on success rate)", log_path=log_path)
    log_message("="*50, log_path=log_path)

    # Generate phishing emails
    try:
        log_message("\nGenerating fresh phishing emails...", log_path=log_path)
        phishing_df = generate_email_dataset(
            num_samples=phishing_samples,
            model=model,
            scenario_type="phishing",
            batch_size=batch_size,
            save_interval=20,
            max_retries=max_retries,
            phishing_scenarios=phishing_scenarios,
            legitimate_scenarios=legitimate_scenarios,
            scenario_variations=scenario_variations,
            project_dir=project_dir,
            run_timestamp=run_timestamp,
            log_path=log_path,
            client=client
        )

        log_message(f"✓ Phishing dataset complete: {len(phishing_df)} samples", log_path=log_path)

        # Count failed generations
        failed_count = len(phishing_df[phishing_df['source'] == 'failed'])
        success_rate = ((len(phishing_df) - failed_count) / len(phishing_df)) * 100
        log_message(f"  - Successful generations: {len(phishing_df) - failed_count} ({success_rate:.1f}%)", log_path=log_path)

    except Exception as e:
        log_message(f"✗ Error in phishing generation: {e}", log_path=log_path)
        phishing_df = pd.DataFrame()
        log_message("✗ Could not generate phishing samples", log_path=log_path)

    # Generate legitimate emails
    try:
        log_message("\nGenerating fresh legitimate emails...", log_path=log_path)
        legitimate_df = generate_email_dataset(
            num_samples=legitimate_samples,
            model=model,
            scenario_type="legitimate",
            batch_size=batch_size,
            save_interval=20,
            max_retries=max_retries,
            phishing_scenarios=phishing_scenarios,
            legitimate_scenarios=legitimate_scenarios,
            scenario_variations=scenario_variations,
            project_dir=project_dir,
            run_timestamp=run_timestamp,
            log_path=log_path,
            client=client
        )

        log_message(f"✓ Legitimate dataset complete: {len(legitimate_df)} samples", log_path=log_path)

        # Count failed generations
        failed_count = len(legitimate_df[legitimate_df['source'] == 'failed'])
        success_rate = ((len(legitimate_df) - failed_count) / len(legitimate_df)) * 100
        log_message(f"  - Successful generations: {len(legitimate_df) - failed_count} ({success_rate:.1f}%)", log_path=log_path)

    except Exception as e:
        log_message(f"✗ Error in legitimate generation: {e}", log_path=log_path)
        legitimate_df = pd.DataFrame()
        log_message("✗ Could not generate legitimate samples", log_path=log_path)

    # Combine datasets
    combined_samples = len(phishing_df) + len(legitimate_df)
    if combined_samples > 0:
        log_message(f"\nCombining datasets: {combined_samples} total samples", log_path=log_path)
        combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)

        # Add features
        combined_df = add_dataset_features(combined_df, log_path)

        # Save final dataset
        final_filename = f"fresh_workplace_email_dataset_{run_timestamp}.csv"
        save_dataset(combined_df, final_filename, project_dir, run_timestamp, log_path)

        # Display stats
        log_message("\nFinal Dataset Statistics:", log_path=log_path)
        log_message(f"- Total samples: {len(combined_df)}", log_path=log_path)
        log_message(f"- Phishing samples: {len(phishing_df)}", log_path=log_path)
        log_message(f"- Legitimate samples: {len(legitimate_df)}", log_path=log_path)

        # Count successful vs failed generations
        if 'source' in combined_df.columns:
            source_counts = combined_df['source'].value_counts()
            log_message("\nGeneration Success Rate:", log_path=log_path)
            for source, count in source_counts.items():
                log_message(f"- {source}: {count} samples ({count/len(combined_df)*100:.1f}%)", log_path=log_path)

        # Show feature statistics
        log_message("\nDataset Features Summary:", log_path=log_path)
        for feature in ['word_count', 'character_count', 'urgency_word_count', 'threat_word_count']:
            if feature in combined_df.columns:
                by_type = combined_df.groupby('label')[feature].mean()
                log_message(f"- Average {feature}: {by_type.to_dict()}", log_path=log_path)

        return combined_df
    else:
        log_message("✗ No samples were successfully generated", log_path=log_path)
        return None

def generate_fresh_datasets_combined(
    phishing_samples=3000,  # Total across both providers
    legitimate_samples=3000,  # Total across both providers
    openai_model="gpt-4.1-mini",
    anthropic_model="claude-3-haiku-20240307",
    batch_size=10,
    max_retries=5,
    phishing_scenarios=None,
    legitimate_scenarios=None,
    scenario_variations=None,
    project_dir=None,
    run_timestamp=None,
    log_path=None,
    openai_client=None,
    anthropic_client=None
):
    """Generate completely new datasets with both OpenAI and Anthropic"""
    log_message("\n" + "="*50, log_path=log_path)
    log_message("STARTING FRESH COMBINED DATASET GENERATION", log_path=log_path)
    log_message(f"Target: {phishing_samples} phishing + {legitimate_samples} legitimate emails", log_path=log_path)
    log_message(f"Models: OpenAI {openai_model}, Anthropic {anthropic_model}", log_path=log_path)
    log_message(f"Starting batch size: {batch_size} (will adapt based on success rate)", log_path=log_path)
    log_message("="*50, log_path=log_path)

    # Generate phishing emails
    try:
        log_message("\nGenerating fresh phishing emails (combined providers)...", log_path=log_path)
        phishing_df = generate_email_dataset_combined(
            num_samples=phishing_samples,
            openai_model=openai_model,
            anthropic_model=anthropic_model,
            scenario_type="phishing",
            batch_size=batch_size,
            save_interval=20,
            max_retries=max_retries,
            phishing_scenarios=phishing_scenarios,
            legitimate_scenarios=legitimate_scenarios,
            scenario_variations=scenario_variations,
            project_dir=project_dir,
            run_timestamp=run_timestamp,
            log_path=log_path,
            openai_client=openai_client,
            anthropic_client=anthropic_client
        )

        log_message(f"✓ Phishing dataset complete: {len(phishing_df)} samples", log_path=log_path)

        # Count failed generations
        failed_count = len(phishing_df[phishing_df['source'].str.contains('failed')])
        success_rate = ((len(phishing_df) - failed_count) / len(phishing_df)) * 100
        log_message(f"  - Successful generations: {len(phishing_df) - failed_count} ({success_rate:.1f}%)", log_path=log_path)

        # Count by provider
        openai_count = len(phishing_df[phishing_df['source'].str.contains('OpenAI')])
        anthropic_count = len(phishing_df[phishing_df['source'].str.contains('Anthropic')])
        log_message(f"  - OpenAI generations: {openai_count}", log_path=log_path)
        log_message(f"  - Anthropic generations: {anthropic_count}", log_path=log_path)

    except Exception as e:
        log_message(f"✗ Error in phishing generation: {e}", log_path=log_path)
        phishing_df = pd.DataFrame()
        log_message("✗ Could not generate phishing samples", log_path=log_path)

    # Generate legitimate emails
    try:
        log_message("\nGenerating fresh legitimate emails (combined providers)...", log_path=log_path)
        legitimate_df = generate_email_dataset_combined(
            num_samples=legitimate_samples,
            openai_model=openai_model,
            anthropic_model=anthropic_model,
            scenario_type="legitimate",
            batch_size=batch_size,
            save_interval=20,
            max_retries=max_retries,
            phishing_scenarios=phishing_scenarios,
            legitimate_scenarios=legitimate_scenarios,
            scenario_variations=scenario_variations,
            project_dir=project_dir,
            run_timestamp=run_timestamp,
            log_path=log_path,
            openai_client=openai_client,
            anthropic_client=anthropic_client
        )

        log_message(f"✓ Legitimate dataset complete: {len(legitimate_df)} samples", log_path=log_path)

        # Count failed generations
        failed_count = len(legitimate_df[legitimate_df['source'].str.contains('failed')])
        success_rate = ((len(legitimate_df) - failed_count) / len(legitimate_df)) * 100
        log_message(f"  - Successful generations: {len(legitimate_df) - failed_count} ({success_rate:.1f}%)", log_path=log_path)

        # Count by provider
        openai_count = len(legitimate_df[legitimate_df['source'].str.contains('OpenAI')])
        anthropic_count = len(legitimate_df[legitimate_df['source'].str.contains('Anthropic')])
        log_message(f"  - OpenAI generations: {openai_count}", log_path=log_path)
        log_message(f"  - Anthropic generations: {anthropic_count}", log_path=log_path)

    except Exception as e:
        log_message(f"✗ Error in legitimate generation: {e}", log_path=log_path)
        legitimate_df = pd.DataFrame()
        log_message("✗ Could not generate legitimate samples", log_path=log_path)

    # Combine datasets
    combined_samples = len(phishing_df) + len(legitimate_df)
    if combined_samples > 0:
        log_message(f"\nCombining datasets: {combined_samples} total samples", log_path=log_path)
        combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)

        # Add features
        combined_df = add_dataset_features(combined_df, log_path)

        # Save final dataset
        final_filename = f"fresh_workplace_email_dataset_combined_{run_timestamp}.csv"
        save_dataset(combined_df, final_filename, project_dir, run_timestamp, log_path)

        # Display stats
        log_message("\nFinal Dataset Statistics:", log_path=log_path)
        log_message(f"- Total samples: {len(combined_df)}", log_path=log_path)
        log_message(f"- Phishing samples: {len(phishing_df)}", log_path=log_path)
        log_message(f"- Legitimate samples: {len(legitimate_df)}", log_path=log_path)

        # Count by provider
        if 'source' in combined_df.columns:
            provider_counts = {
                'OpenAI': len(combined_df[combined_df['source'].str.contains('OpenAI')]),
                'Anthropic': len(combined_df[combined_df['source'].str.contains('Anthropic')]),
                'Failed': len(combined_df[combined_df['source'].str.contains('failed')])
            }

            log_message("\nGeneration by Provider:", log_path=log_path)
            for provider, count in provider_counts.items():
                log_message(f"- {provider}: {count} samples ({count/len(combined_df)*100:.1f}%)", log_path=log_path)

        # Show feature statistics
        log_message("\nDataset Features Summary:", log_path=log_path)
        for feature in ['word_count', 'character_count', 'urgency_word_count', 'threat_word_count']:
            if feature in combined_df.columns:
                by_type = combined_df.groupby('label')[feature].mean()
                log_message(f"- Average {feature}: {by_type.to_dict()}", log_path=log_path)

        return combined_df
    else:
        log_message("✗ No samples were successfully generated", log_path=log_path)
        return None

def run_email_generation(use_combined=True):
    """Run the email generation process with OpenAI only or combined with Anthropic"""
    # Set up project directory
    project_dir = os.path.join(os.getcwd(), 'phishing_project')
    os.makedirs(project_dir, exist_ok=True)
    print(f"✓ Project directory confirmed: {project_dir}")

    # Create timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create log file
    mode_str = "combined" if use_combined else "openai_only"
    log_path = f"{project_dir}/{mode_str}_generation_log_{run_timestamp}.txt"
    with open(log_path, "w") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing {mode_str} dataset generation\n")
    print(f"✓ Log file created at: {log_path}")

    # Set up OpenAI API client
    try:
        # Get API key from environment variable or user input
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            print("Please enter your OpenAI API key:")
            openai_api_key = input()
            os.environ['OPENAI_API_KEY'] = openai_api_key  # Set for future use

        openai_client = OpenAI(api_key=openai_api_key)

        # Test API with a simple request
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("✓ Successfully connected to OpenAI API!")
    except Exception as e:
        print(f"✗ Error setting up OpenAI client: {e}")
        print("Please enter your OpenAI API key again:")
        openai_api_key = input()
        openai_client = OpenAI(api_key=openai_api_key)

    # Set up Anthropic API client if using combined approach
    anthropic_client = None
    if use_combined:
        try:
            anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not anthropic_api_key:
                print("Please enter your Anthropic API key:")
                anthropic_api_key = input()
                os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key  # Set for future use

            anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

            # Test API with a simple request
            response = anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            print("✓ Successfully connected to Anthropic API!")
        except Exception as e:
            print(f"✗ Error setting up Anthropic client: {e}")
            print("Please enter your Anthropic API key again:")
            anthropic_api_key = input()
            anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    print("\n" + "="*50)
    if use_combined:
        print("COMBINED WORKPLACE EMAIL DATASET GENERATOR (OpenAI + Anthropic)")
        PHISHING_SAMPLES = DEFAULT_CONFIG["combined"]["phishing_samples"]
        LEGITIMATE_SAMPLES = DEFAULT_CONFIG["combined"]["legitimate_samples"]
        print(f"OpenAI Model: {DEFAULT_CONFIG['combined']['openai_model']}")
        print(f"Anthropic Model: {DEFAULT_CONFIG['combined']['anthropic_model']}")
    else:
        print("WORKPLACE EMAIL DATASET GENERATOR (OpenAI only)")
        PHISHING_SAMPLES = DEFAULT_CONFIG["openai"]["phishing_samples"]
        LEGITIMATE_SAMPLES = DEFAULT_CONFIG["openai"]["legitimate_samples"]
        print(f"Model: {DEFAULT_CONFIG['openai']['model']}")

    print("="*50)

    BATCH_SIZE = 10  # Initial batch size for stability

    print(f"Dataset composition: {PHISHING_SAMPLES} phishing emails, {LEGITIMATE_SAMPLES} legitimate emails")
    if use_combined:
        print(f"(50/50 split between OpenAI and Anthropic for each type)")
    print(f"Focus: Workplace environment email scenarios")
    print(f"IMPORTANT: All phishing emails include academic research disclaimer")

    if use_combined:
        # Parameters for combined generation
        generation_params = {
            "phishing_samples": PHISHING_SAMPLES,
            "legitimate_samples": LEGITIMATE_SAMPLES,
            "openai_model": DEFAULT_CONFIG["combined"]["openai_model"],
            "anthropic_model": DEFAULT_CONFIG["combined"]["anthropic_model"],
            "batch_size": BATCH_SIZE,
            "max_retries": DEFAULT_CONFIG["combined"]["max_retries"],
            "phishing_scenarios": phishing_scenarios,
            "legitimate_scenarios": legitimate_scenarios,
            "scenario_variations": scenario_variations,
            "project_dir": project_dir,
            "run_timestamp": run_timestamp,
            "log_path": log_path,
            "openai_client": openai_client,
            "anthropic_client": anthropic_client
        }

        proceed = input("Do you want to proceed with generating FRESH datasets using both OpenAI and Anthropic? (y/n): ")
        if proceed.lower() != 'y':
            print("Generation cancelled by user")
            return None

        # Generate fresh datasets using both providers
        dataset = generate_fresh_datasets_combined(**generation_params)
    else:
        # Parameters for OpenAI-only generation
        generation_params = {
            "phishing_samples": PHISHING_SAMPLES,
            "legitimate_samples": LEGITIMATE_SAMPLES,
            "model": DEFAULT_CONFIG["openai"]["model"],
            "batch_size": BATCH_SIZE,
            "max_retries": DEFAULT_CONFIG["openai"]["max_retries"],
            "phishing_scenarios": phishing_scenarios,
            "legitimate_scenarios": legitimate_scenarios,
            "scenario_variations": scenario_variations,
            "project_dir": project_dir,
            "run_timestamp": run_timestamp,
            "log_path": log_path,
            "client": openai_client
        }

        proceed = input("Do you want to proceed with generating FRESH datasets using OpenAI only? (y/n): ")
        if proceed.lower() != 'y':
            print("Generation cancelled by user")
            return None

        # Generate fresh datasets using OpenAI only
        dataset = generate_fresh_datasets(**generation_params)

    # Completion message
    if dataset is not None and len(dataset) > 0:
        print("\n" + "="*50)
        if use_combined:
            print("COMBINED DATASET GENERATION SUCCESSFUL")
        else:
            print("DATASET GENERATION SUCCESSFUL")
        print(f"Final dataset saved with {len(dataset)} samples")
        print("="*50)

        if use_combined:
            # Display sample emails from both providers
            print("\nSample phishing email from OpenAI:")
            try:
                phishing_sample_openai = dataset[(dataset['label'] == 'phishing') &
                                                (dataset['source'].str.contains('OpenAI'))].iloc[0]
                print("-" * 50)
                print(phishing_sample_openai['text'][:500] + "..." if len(phishing_sample_openai['text']) > 500
                    else phishing_sample_openai['text'])
                print("-" * 50)

                print("\nSample phishing email from Anthropic:")
                phishing_sample_anthropic = dataset[(dataset['label'] == 'phishing') &
                                                (dataset['source'].str.contains('Anthropic'))].iloc[0]
                print("-" * 50)
                print(phishing_sample_anthropic['text'][:500] + "..." if len(phishing_sample_anthropic['text']) > 500
                    else phishing_sample_anthropic['text'])
                print("-" * 50)

                print("\nSample legitimate email from OpenAI:")
                legitimate_sample_openai = dataset[(dataset['label'] == 'legitimate') &
                                                (dataset['source'].str.contains('OpenAI'))].iloc[0]
                print("-" * 50)
                print(legitimate_sample_openai['text'][:500] + "..." if len(legitimate_sample_openai['text']) > 500
                    else legitimate_sample_openai['text'])
                print("-" * 50)

                print("\nSample legitimate email from Anthropic:")
                legitimate_sample_anthropic = dataset[(dataset['label'] == 'legitimate') &
                                                    (dataset['source'].str.contains('Anthropic'))].iloc[0]
                print("-" * 50)
                print(legitimate_sample_anthropic['text'][:500] + "..." if len(legitimate_sample_anthropic['text']) > 500
                    else legitimate_sample_anthropic['text'])
                print("-" * 50)
            except:
                print("Could not display sample emails.")
        else:
            # Display sample emails from OpenAI only
            print("\nSample phishing email:")
            try:
                phishing_sample = dataset[dataset['label'] == 'phishing'].iloc[0]
                print("-" * 50)
                print(phishing_sample['text'][:500] + "..." if len(phishing_sample['text']) > 500 else phishing_sample['text'])
                print("-" * 50)

                print("\nSample legitimate email:")
                legitimate_sample = dataset[dataset['label'] == 'legitimate'].iloc[0]
                print("-" * 50)
                print(legitimate_sample['text'][:500] + "..." if len(legitimate_sample['text']) > 500 else legitimate_sample['text'])
                print("-" * 50)
            except:
                print("Could not display sample emails.")
    else:
        print("\n" + "="*50)
        print("DATASET GENERATION ENCOUNTERED PROBLEMS")
        print("Please check the log file for details")
        print("="*50)

    return dataset