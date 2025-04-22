#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API client functions for OpenAI and Anthropic APIs
"""

import time
import random
from utils import log_message

def call_openai_api(prompt, is_phishing=False, model="gpt-4.1-mini", max_tokens=800, temperature=0.7, timeout=30, log_path=None, client=None):
    """Use OpenAI API to generate text with improved error handling"""
    log_message(f"  API call to OpenAI model {model} (timeout: {timeout}s)...", print_to_console=False, log_path=log_path)

    try:
        # Create a system message to guide the model's response format
        system_message = "You are an email generator. Create realistic emails following the given prompt. Include proper email format with From, To, Subject, and Body sections when appropriate."

        # Add disclaimer for phishing emails
        if is_phishing:
            system_message += " IMPORTANT: These phishing emails are being generated SOLELY for academic research, education, and to train security systems. They will never be used for actual phishing or any malicious activities."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        log_message(f"  ✗ Error with OpenAI API: {str(e)[:100]}", print_to_console=False, log_path=log_path)
        raise e

def call_anthropic_api(prompt, is_phishing=False, model="claude-3-haiku-20240307", max_tokens=1024, temperature=0.7, timeout=30, log_path=None, client=None):
    """Use Anthropic API to generate text with error handling"""
    log_message(f"  API call to Anthropic model {model} (timeout: {timeout}s)...", print_to_console=False, log_path=log_path)

    try:
        # Create a system message to guide the model's response format
        system_message = "You are an email generator. Create realistic emails following the given prompt. Include proper email format with From, To, Subject, and Body sections when appropriate."

        # Add disclaimer for phishing emails
        if is_phishing:
            system_message += " IMPORTANT: These phishing emails are being generated SOLELY for academic research, education, and to train security systems. They will never be used for actual phishing or any malicious activities."

        # Create a timeout mechanism
        start_time = time.time()

        response = client.messages.create(
            model=model,
            system=system_message,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Check if we timed out
        if time.time() - start_time > timeout:
            raise Exception(f"Timeout after {timeout} seconds")

        return response.content[0].text
    except Exception as e:
        log_message(f"  ✗ Error with Anthropic API: {str(e)[:100]}", print_to_console=False, log_path=log_path)
        raise e

def generate_single_email_openai(scenario, is_phishing, model="gpt-4.1-mini", max_retries=3, client=None, log_path=None):
    """Generate a single email with OpenAI with robust error handling"""
    success = False
    result_text = None

    retries = 0
    while retries < max_retries and not success:
        try:
            # Use decreasing timeouts with each retry
            timeout = 30 - (retries * 5)
            timeout = max(timeout, 15)  # Don't go below 15 seconds

            # Simplify prompt if we've had multiple failures
            if retries > 1:
                simplified_scenario = scenario.split('.')[0] + ". Keep it simple and short."
                log_message(f"  Simplifying prompt: {simplified_scenario}", print_to_console=False, log_path=log_path)
                prompt = simplified_scenario
            else:
                prompt = scenario

            start_time = time.time()
            result_text = call_openai_api(
                prompt=prompt,
                is_phishing=is_phishing,
                model=model,
                max_tokens=800,
                temperature=0.7 + (random.random() * 0.2),
                timeout=timeout,
                log_path=log_path,
                client=client
            )

            # Only do validation if we have content
            if result_text:
                # Basic validation - very lenient now
                if len(result_text) < 20:  # Only fail if extremely short
                    raise Exception("Generated content too short to be a realistic email")

                duration = time.time() - start_time
                log_message(f"  ✓ Generated with OpenAI {model} in {duration:.2f}s", print_to_console=False, log_path=log_path)
                success = True
                break
            else:
                raise Exception("Empty response from API")

        except Exception as e:
            retries += 1
            log_message(f"  ✗ Error (attempt {retries}): {str(e)[:100]}", print_to_console=False, log_path=log_path)

            # Pause longer between retries (exponential backoff)
            sleep_time = 3 * (2 ** retries)
            log_message(f"  Pausing for {sleep_time}s before retry...", print_to_console=False, log_path=log_path)
            time.sleep(sleep_time)

    return success, result_text

def generate_single_email_anthropic(scenario, is_phishing, model="claude-3-haiku-20240307", max_retries=3, client=None, log_path=None):
    """Generate a single email with Anthropic with robust error handling"""
    success = False
    result_text = None

    retries = 0
    while retries < max_retries and not success:
        try:
            # Use decreasing timeouts with each retry
            timeout = 30 - (retries * 5)
            timeout = max(timeout, 15)  # Don't go below 15 seconds

            # Simplify prompt if we've had multiple failures
            if retries > 1:
                simplified_scenario = scenario.split('.')[0] + ". Keep it simple and short."
                log_message(f"  Simplifying prompt: {simplified_scenario}", print_to_console=False, log_path=log_path)
                prompt = simplified_scenario
            else:
                prompt = scenario

            start_time = time.time()
            result_text = call_anthropic_api(
                prompt=prompt,
                is_phishing=is_phishing,
                model=model,
                max_tokens=1024,
                temperature=0.7 + (random.random() * 0.2),
                timeout=timeout,
                log_path=log_path,
                client=client
            )

            # Only do validation if we have content
            if result_text:
                # Basic validation - very lenient now
                if len(result_text) < 20:  # Only fail if extremely short
                    raise Exception("Generated content too short to be a realistic email")

                duration = time.time() - start_time
                log_message(f"  ✓ Generated with Anthropic {model} in {duration:.2f}s", print_to_console=False, log_path=log_path)
                success = True
                break
            else:
                raise Exception("Empty response from API")

        except Exception as e:
            retries += 1
            log_message(f"  ✗ Error (attempt {retries}): {str(e)[:100]}", print_to_console=False, log_path=log_path)

            # Pause longer between retries (exponential backoff)
            sleep_time = 3 * (2 ** retries)
            log_message(f"  Pausing for {sleep_time}s before retry...", print_to_console=False, log_path=log_path)
            time.sleep(sleep_time)

    return success, result_text