#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data processing functions for feature extraction and dataset analysis
"""

import re
from utils import log_message

def add_dataset_features(df, log_path=None):
    """Add useful features to the email dataset"""
    log_message("Adding features to dataset...", log_path=log_path)

    try:
        # Basic text features
        df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
        df["character_count"] = df["text"].apply(lambda x: len(str(x)))
        df["line_count"] = df["text"].apply(lambda x: len(str(x).split('\n')))

        # Email structure features
        df["has_greeting"] = df["text"].apply(
            lambda x: bool(re.search(r'(dear|hello|hi|greetings|good morning|good afternoon|good evening)',
                                    str(x).lower()))
        )

        df["has_signature"] = df["text"].apply(
            lambda x: bool(re.search(r'(sincerely|regards|thank you|thanks|best|cheers|yours truly|team)',
                                    str(x).lower()))
        )

        # Content indicators
        urgency_words = ["urgent", "immediately", "alert", "warning", "attention", "now",
                         "critical", "important", "emergency", "limited time", "deadline",
                         "urgent action", "risk", "expir", "suspend", "terminat"]
        df["contains_urgency"] = df["text"].apply(
            lambda x: any(word in str(x).lower() for word in urgency_words)
        )

        df["urgency_word_count"] = df["text"].apply(
            lambda x: sum(1 for word in urgency_words if word in str(x).lower())
        )

        link_indicators = ["click", "link", "http", "https", "www", ".com", ".net", ".org", "url",
                          "website", "portal", "login", "sign in", "access", "account", "verify"]
        df["contains_link_text"] = df["text"].apply(
            lambda x: any(indicator in str(x).lower() for indicator in link_indicators)
        )

        df["link_indicator_count"] = df["text"].apply(
            lambda x: sum(1 for indicator in link_indicators if indicator in str(x).lower())
        )

        # Threat and security indicators
        security_words = ["password", "account", "security", "verify", "confirm", "authenticate",
                         "credentials", "login", "sign in", "access", "identity", "verification"]
        df["security_word_count"] = df["text"].apply(
            lambda x: sum(1 for word in security_words if word in str(x).lower())
        )

        threat_words = ["suspicious", "unauthorized", "fraud", "breach", "hack", "compromised",
                       "risk", "threat", "urgent", "warning", "alert", "unusual", "suspicious activity",
                       "unusual activity", "security breach", "violation", "compromised"]
        df["threat_word_count"] = df["text"].apply(
            lambda x: sum(1 for word in threat_words if word in str(x).lower())
        )

        # Financial indicators
        financial_words = ["payment", "credit card", "bank", "account number", "transaction",
                          "transfer", "money", "fund", "balance", "statement", "invoice", "bill",
                          "charge", "$", "dollar", "refund", "credit", "debit"]
        df["financial_word_count"] = df["text"].apply(
            lambda x: sum(1 for word in financial_words if word in str(x).lower())
        )

        # Time pressure indicators
        time_pressure_words = ["today", "now", "immediately", "urgent", "deadline", "soon",
                              "limited time", "expires", "expiring", "within 24 hours", "within 48 hours",
                              "by tomorrow", "as soon as possible", "asap"]
        df["time_pressure_word_count"] = df["text"].apply(
            lambda x: sum(1 for word in time_pressure_words if word in str(x).lower())
        )

        # Sentiment indicators (simplified)
        negative_words = ["suspicious", "unauthorized", "fraud", "breach", "hack", "compromised",
                         "risk", "threat", "urgent", "warning", "alert", "unusual", "error", "problem",
                         "issue", "concern", "attention", "immediate", "suspend", "terminate"]
        df["negative_sentiment_score"] = df["text"].apply(
            lambda x: sum(1 for word in negative_words if word in str(x).lower())
        )

        positive_words = ["thank", "please", "appreciate", "value", "happy", "glad", "congratulations",
                         "opportunity", "benefit", "bonus", "reward", "gift", "free", "discount", "offer"]
        df["positive_sentiment_score"] = df["text"].apply(
            lambda x: sum(1 for word in positive_words if word in str(x).lower())
        )

        log_message(f"✓ Added {len(df.columns) - 9} features to dataset", log_path=log_path)
        return df

    except Exception as e:
        log_message(f"✗ Error adding features: {e}", log_path=log_path)
        return df

def extract_email_metadata(result_text):
    """Extract metadata from generated email text"""
    subject_line = ""
    sender = ""
    recipient = ""

    # Try to extract subject line using common patterns
    subject_match = re.search(r'Subject:(.+?)(?:\n|$)', result_text, re.IGNORECASE)
    if subject_match:
        subject_line = subject_match.group(1).strip()

    # Try to extract sender using common patterns
    from_match = re.search(r'From:(.+?)(?:\n|$)', result_text, re.IGNORECASE)
    if from_match:
        sender = from_match.group(1).strip()

    # Try to extract recipient using common patterns
    to_match = re.search(r'To:(.+?)(?:\n|$)', result_text, re.IGNORECASE)
    if to_match:
        recipient = to_match.group(1).strip()
        
    return {
        "subject": subject_line,
        "sender": sender,
        "recipient": recipient
    }