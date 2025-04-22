#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration data for AI-Phishing-Gen-Combined
"""

# Define workplace phishing scenarios
phishing_scenarios = [
    # IT and system related
    "Write a phishing email pretending to be from IT support requiring immediate password reset",
    "Create a phishing email claiming to be from IT security about malware detection on your workstation",
    "Write a phishing email as the IT department about an urgent software update needed",
    "Create a phishing email about VPN access expiring and needing immediate renewal",
    "Write a phishing email from the IT helpdesk about suspicious login attempts on your account",
    "Create a phishing email about an urgent security patch required for remote access",
    "Write a phishing email about unauthorized access to company system requiring immediate password change",
    "Create a phishing email about a security audit requiring verification of login credentials",
    "Write a phishing email about outdated antivirus requiring immediate system access",
    "Create a phishing email about required multi-factor authentication setup",

    # HR related
    "Write a phishing email claiming to be from HR about an urgent document review",
    "Create a phishing email from HR about benefits enrollment deadline",
    "Write a phishing email pretending to be from the payroll department about a payment issue",
    "Create a phishing email about an urgent update to employee information required for tax purposes",
    "Write a phishing email as HR requesting verification of personal information",
    "Create a phishing email about changes to employee benefits requiring immediate action",
    "Write a phishing email about payroll discrepancy requiring bank account verification",
    "Create a phishing email about upcoming performance review requiring form completion",
    "Write a phishing email about employee satisfaction survey with suspicious link",
    "Create a phishing email about retirement fund account verification",

    # Executive and management
    "Write a phishing email from the CEO requesting a confidential task",
    "Create a phishing email from the CFO requesting verification of financial information",
    "Write a phishing email pretending to be from the CEO requesting gift cards for a client",
    "Create a phishing email from a senior executive requesting access to confidential files",
    "Write a phishing email as the CEO asking for urgent help with a sensitive matter",
    "Create a phishing email from a board member requesting confidential company information",
    "Write a phishing email from the CFO about urgent wire transfer requiring verification",
    "Create a phishing email from a department head requesting confidential employee information",
    "Write a phishing email from an executive about upcoming restructuring requiring login",
    "Create a phishing email from the COO about emergency protocol verification",

    # Collaboration and services
    "Write a phishing email about a shared document requiring immediate attention",
    "Create a phishing email about a voicemail message waiting in the system",
    "Write a phishing email about a failed email delivery requiring authentication",
    "Create a phishing email about expiring access to cloud storage with company documents",
    "Write a phishing email about an invitation to edit an important document requiring sign-in",
    "Create a phishing email requesting credentials for a mandatory training platform",
    "Write a phishing email about calendar invitation containing suspicious link",
    "Create a phishing email about video conference access requiring login verification",
    "Write a phishing email about company chat platform security update requiring password reset",
    "Create a phishing email about shared file access problem requiring credential verification",

    # Company operations
    "Write a phishing email claiming to be from facilities management about an office issue",
    "Create a phishing email about unauthorized access to the building requiring credential verification",
    "Write a phishing email about an important company survey with a suspicious link",
    "Create a phishing email about changes to parking access requiring account verification",
    "Write a phishing email pretending to be from the procurement department about an invoice",
    "Create a phishing email about a security incident in the building requiring information verification",
    "Write a phishing email about office access card renewal requiring personal information",
    "Create a phishing email about expired building credentials requiring immediate update",
    "Write a phishing email about company equipment inventory check requiring login",
    "Create a phishing email about workplace safety incident report requiring credentials"
]

# Define legitimate workplace email scenarios
legitimate_scenarios = [
    # Business communication
    "Write a professional email announcing a company meeting",
    "Create an email from a manager providing feedback on a recent project",
    "Write a friendly email to a colleague about a project update",
    "Create an email requesting time off for vacation",
    "Write an email scheduling a team lunch",
    "Create an email sharing quarterly business results",
    "Write an email introducing a new team member",
    "Create an email about project milestone achievements",
    "Write an email requesting status updates from team members",
    "Create an email documenting meeting minutes",

    # IT and support
    "Create an email about scheduled system maintenance",
    "Write an email announcing new IT security procedures",
    "Create an email about password policy updates",
    "Write an email about new software roll-out and training",
    "Create an email about updates to the company VPN service",
    "Write an email announcing secure file sharing policy updates",
    "Create an email about IT service desk hours during holidays",
    "Write an email about new employee technology onboarding",
    "Create an email about scheduled server maintenance",
    "Write an email about printer troubleshooting procedures",

    # HR and company policy
    "Create an email about an upcoming performance review process",
    "Write an email about holiday schedule and office closures",
    "Create an email about healthcare benefits enrollment period",
    "Write an email about workplace safety procedures",
    "Create an email about company policy updates",
    "Write an email about a workplace wellness initiative",
    "Create an email about changes to the employee handbook",
    "Write an email about employee development opportunities",
    "Create an email about work-from-home policy updates",
    "Write an email about employee recognition program nominations",

    # Executive and company-wide
    "Create an email from the CEO sharing quarterly company updates",
    "Write an email announcing strategic company changes",
    "Create an email about an employee recognition program",
    "Write an email about company achievement and milestones",
    "Create an email announcing merger or acquisition news",
    "Write an email sharing company goals for the upcoming quarter",
    "Create an email about company charitable initiatives",
    "Write an email about leadership team changes",
    "Create an email about company anniversary celebration",
    "Write an email about industry award recognition",

    # Events and training
    "Create an email about upcoming training opportunities",
    "Write an email inviting staff to a company event",
    "Create an email about professional development resources",
    "Write an email about an upcoming team building activity",
    "Create an email with conference registration information",
    "Write an email about a lunch and learn session",
    "Create an email about mandatory compliance training",
    "Write an email about workshop registration details",
    "Create an email about continuing education opportunities",
    "Write an email about a successful company event recap"
]

# Define more detailed variations to make the scenarios diverse
scenario_variations = [
    # Standard variations
    "{scenario}. Make it sound urgent.",
    "{scenario}. Include a call to action.",
    "{scenario}. Keep it short and direct.",
    "{scenario}. Make it personalized.",
    "{scenario}. Mention a deadline to respond.",

    # Phishing-specific variations (will only be applied to phishing scenarios)
    "{scenario}. Include a fake URL that looks legitimate.",
    "{scenario}. Mention account suspension if no action is taken.",
    "{scenario}. Include an attachment reference.",
    "{scenario}. Try to create fear or urgency without being too obvious.",
    "{scenario}. Mention potential consequences if no action is taken.",
    "{scenario}. Make it appear to come from the security department.",
    "{scenario}. Mention suspicious activity on their account.",

    # Legitimate-specific variations (will only be applied to legitimate scenarios)
    "{scenario}. Include a proper email signature.",
    "{scenario}. Include friendly but professional language.",
    "{scenario}. Make it appear automated like a system notification.",
    "{scenario}. Include a FAQ section at the bottom.",
    "{scenario}. Include a disclaimer about company policy.",
    "{scenario}. Include contact information for questions."
]

# Default values for dataset generation
DEFAULT_CONFIG = {
    # OpenAI-only mode
    "openai": {
        "phishing_samples": 1500,
        "legitimate_samples": 1500,
        "model": "gpt-4.1-mini",
        "batch_size": 10,
        "max_retries": 5
    },
    # Combined mode
    "combined": {
        "phishing_samples": 3000,  # 1500 OpenAI + 1500 Anthropic
        "legitimate_samples": 3000,  # 1500 OpenAI + 1500 Anthropic
        "openai_model": "gpt-4.1-mini",
        "anthropic_model": "claude-3-haiku-20240307",
        "batch_size": 10,
        "max_retries": 5
    }
}