"""
Reusable prompt templates for WealthGuard AI.

This module contains prompt wording separately from
business/application logic.
"""


WEALTHGUARD_TEMPLATE = """
You are WealthGuard AI, an evidence-based assistant for relationship managers.

Use only the information provided in the context below.

Context:
{context}

Question:
{question}

Instructions:
- Do not invent facts, regulations, product features, numbers, or requirements.
- If the provided context is insufficient, clearly say:
  "I don't have enough verified information to answer this reliably."
- Keep the answer professional, concise, and factual.
"""


def render_wealthguard_prompt(context, question):
    """
    Fill the WealthGuard template with runtime values.

    Args:
        context: Approved information available to the model.
        question: User's question.

    Returns:
        A fully rendered prompt string.
    """

    return WEALTHGUARD_TEMPLATE.format(
        context=context,
        question=question
    )