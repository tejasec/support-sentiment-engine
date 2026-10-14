"""
Synthetic Customer Support Ticket Corpus Generator.
Creates realistic multi-class, sentiment-annotated, and entity-rich support queries.
"""

import json
import random
from typing import Dict, List, Optional
import pandas as pd


NAMES = [
    "Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince", "Evan Wright",
    "Fiona Gallagher", "George Clark", "Hannah Abbott", "Ian Malcolm", "Julia Roberts"
]

PRODUCTS = [
    "Pro Cloud Subscription", "Enterprise Data Engine", "Smart Home Hub v2",
    "Ultra Wireless Earbuds", "AI Analytics Dashboard", "Omni POS Terminal"
]

TEMPLATES = {
    "Billing & Invoicing": [
        ("Disputed charge on invoice #{order_id}", "Hello, I was charged twice for order {order_id} on {date}. Please refund the excess amount immediately.", "High", "Negative"),
        ("Invoice request for {product}", "Hi support, could you email me the tax invoice for order {order_id} dated {date}? Thanks, {name}.", "Low", "Neutral"),
        ("Unexpected subscription fee", "Why did my credit card get billed $49 on {date}? I canceled my {product} last week. Need help!", "Medium", "Negative"),
        ("Billing receipt received", "Thank you for sending the updated receipt for {order_id}. All details look perfect!", "Low", "Positive")
    ],
    "Technical Support": [
        ("System crash when opening {product}", "Urgent: The application keeps throwing Error 503 since {date}. Our whole team is blocked. Order {order_id}.", "Critical", "Negative"),
        ("Feature configuration issue", "Having trouble configuring the webhook for {product}. Documentation doesn't cover this edge case.", "Medium", "Neutral"),
        ("Bug resolved nicely", "Just wanted to say the technical fix deployed on {date} resolved my issue with {product}. Great job!", "Low", "Positive"),
        ("Memory leak in dashboard", "We observed severe lag and high CPU usage in {product} after the latest patch on {date}.", "High", "Negative")
    ],
    "Account Access": [
        ("Locked out of my account", "I cannot log in with email {email}. The reset password link expired on {date}. Please unlock my account.", "High", "Negative"),
        ("2FA reset inquiry", "I lost my 2FA authentication device. Can you help verify my identity for account linked to {email}?", "High", "Neutral"),
        ("Update profile email address", "Hi team, please update my primary email from {email} to my corporate address. Order {order_id}.", "Low", "Neutral"),
        ("Password reset worked", "Thanks for quickly resolving my access issue for {product} today!", "Low", "Positive")
    ],
    "Product & Feature Inquiries": [
        ("Does {product} support SSO?", "Hello, we are evaluating {product} for 500 seats. Does it integrate with Okta or Azure AD?", "Medium", "Neutral"),
        ("API rate limits documentation", "Could you share the maximum requests per minute allowed on the {product} enterprise plan?", "Low", "Neutral"),
        ("Amazing new release!", "I tried the new feature launched on {date} for {product}. Truly fantastic improvement!", "Low", "Positive"),
        ("Feature request for reporting", "Is there an export to Parquet or CSV option planned for {product}? Currently only PDF is available.", "Low", "Neutral")
    ],
    "Shipping & Delivery": [
        ("Delayed shipment for {order_id}", "Where is my package? Tracking shows order {order_id} stuck in transit since {date}. Very frustrated!", "High", "Negative"),
        ("Change delivery address", "Can I update the delivery address for order {order_id}? Ordered {product} on {date}.", "Medium", "Neutral"),
        ("Package received in good shape", "My order {order_id} arrived earlier than expected on {date}. Excellent delivery service!", "Low", "Positive"),
        ("Damaged box upon arrival", "The courier delivered order {order_id} but the box of {product} was torn open. Please replace it.", "High", "Negative")
    ],
    "Cancellation & Returns": [
        ("Return request for order {order_id}", "I would like to initiate a return for {product} purchased on {date} under order {order_id}.", "Medium", "Neutral"),
        ("Cancel service immediately", "Please cancel my {product} subscription immediately. The service did not meet my expectations.", "High", "Negative"),
        ("Refund confirmed", "I received the confirmation refund for {order_id}. Appreciate the fast turnaround!", "Low", "Positive"),
        ("Return policy clarification", "What is the return window for {product}? Is it 30 days or 60 days from {date}?", "Low", "Neutral")
    ]
}

SPAM_TEMPLATES = [
    ("CLAIM YOUR $10,000 PRIZE NOW!!!", "Congratulations! You won a brand new gift card. Click http://spam-link.xyz to claim before midnight! Free rewards!", "Low", "Positive"),
    ("Urgent: Verify your crypto wallet immediately", "Security alert: Verify your secret recovery phrase at http://phishing-scam.net/login immediately or funds will be lost.", "High", "Negative"),
    ("Cheap luxury watches discount 90%", "Buy replica Swiss watches today only! Limited stock at http://watches-sale-cheap.biz! Best quality guaranteed.", "Low", "Neutral")
]


def generate_support_tickets(
    n_samples: int = 1000,
    random_state: int = 42,
    spam_ratio: float = 0.08
) -> pd.DataFrame:
    """
    Generate synthetic customer support dataset with category, urgency,
    sentiment, and entity metadata.
    """
    random.seed(random_state)
    records: List[Dict[str, object]] = []

    categories = list(TEMPLATES.keys())

    for idx in range(1, n_samples + 1):
        ticket_id = f"TCK-{10000 + idx}"
        customer_name = random.choice(NAMES)
        email = f"{customer_name.lower().replace(' ', '.')}@example.com"
        product = random.choice(PRODUCTS)
        order_num = f"ORD-{random.randint(20000, 99999)}"
        date_str = f"2026-0{random.randint(1, 9)}-{random.randint(10, 28)}"

        is_spam = 1 if (random.random() < spam_ratio) else 0

        if is_spam:
            subject_tpl, body_tpl, urgency, sentiment = random.choice(SPAM_TEMPLATES)
            category = "Spam / Promotion"
            subject = subject_tpl
            body = body_tpl
            entities = {"spam_url": True}
        else:
            category = random.choice(categories)
            subject_tpl, body_tpl, urgency, sentiment = random.choice(TEMPLATES[category])
            subject = subject_tpl.format(order_id=order_num, product=product, date=date_str, name=customer_name, email=email)
            body = body_tpl.format(order_id=order_num, product=product, date=date_str, name=customer_name, email=email)
            entities = {
                "order_id": order_num,
                "product": product,
                "date": date_str,
                "customer_name": customer_name
            }

        records.append({
            "ticket_id": ticket_id,
            "customer_name": customer_name,
            "email": email,
            "subject": subject,
            "body": body,
            "text": f"{subject} {body}",
            "category": category,
            "urgency": urgency,
            "sentiment": sentiment,
            "is_spam": is_spam,
            "entities_json": json.dumps(entities)
        })

    return pd.DataFrame(records)
