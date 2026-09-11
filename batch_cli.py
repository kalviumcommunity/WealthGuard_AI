import os

from dotenv import load_dotenv
from openai import OpenAI

from prompts.templates import render_wealthguard_prompt


load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")

if not BASE_URL:
    raise RuntimeError("OPENAI_BASE_URL is missing from .env")

if not MODEL:
    raise RuntimeError("OPENAI_MODEL is missing from .env")


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


def ask_wealthguard(context, question):
    user_prompt = render_wealthguard_prompt(
        context=context,
        question=question,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        max_tokens=100,
    )

    return response.choices[0].message.content


def main():
    context = """
Product: Secure Growth Plan
Annual management fee: 1.2%
Minimum investment: ₹50,000
Lock-in period: 3 years
Early withdrawal: Subject to applicable terms and conditions
Capital protection: Not guaranteed
"""

    question = (
        "What are the key points a relationship manager "
        "should explain to a customer?"
    )

    print("=" * 70)
    print("WEALTHGUARD AI - BATCH/CLI FEATURE")
    print("=" * 70)

    answer = ask_wealthguard(
        context=context,
        question=question,
    )

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()