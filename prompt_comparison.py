import os
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIError


# Load environment variables
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


# Create OpenAI-compatible client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


USER_QUESTION = (
    "What should a relationship manager consider when explaining "
    "an investment product to a customer?"
)


# Prompt Variation 1: Vague
VAGUE_SYSTEM_PROMPT = """
You are a wealth management assistant.
Answer the user's question.
"""


# Prompt Variation 2: Clear and constrained
CONSTRAINED_SYSTEM_PROMPT = """
You are WealthGuard AI, an evidence-based assistant for relationship managers.

Scope:
- Provide general informational guidance based on approved wealth-management knowledge.
- Explain concepts clearly without making personalized financial recommendations.
- Do not invent facts, regulations, product features, tax treatment, or eligibility requirements.
- If the required information is unavailable or cannot be verified, say:
  "I don't have enough verified information to answer this reliably."

Constraints:
- Keep the answer under 120 words.
- Use a professional and neutral tone.
- Structure the response as 3-5 concise bullet points.
- Clearly distinguish general information from customer-specific advice.
"""


def get_response(system_prompt, user_question):
    messages = [
        {
            "role": "system",
            "content": system_prompt.strip(),
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return response.choices[0].message.content


def main():
    print("=" * 70)
    print("WEALTHGUARD AI - PROMPT COMPARISON")
    print("=" * 70)

    print("\nUSER QUESTION:")
    print(USER_QUESTION)

    print("\n" + "=" * 70)
    print("PROMPT VARIATION 1 - VAGUE")
    print("=" * 70)

    print("\nSYSTEM PROMPT:")
    print(VAGUE_SYSTEM_PROMPT.strip())

    try:
        vague_response = get_response(
            VAGUE_SYSTEM_PROMPT,
            USER_QUESTION,
        )

        print("\nMODEL OUTPUT:")
        print(vague_response)

    except AuthenticationError:
        print("\nERROR: Authentication failed (401). Check your API key.")

    except RateLimitError:
        print("\nERROR: Rate limit exceeded (429). Try again later.")

    except APIConnectionError:
        print("\nERROR: Could not connect to the API endpoint.")

    except APIError as error:
        print(f"\nERROR: API error: {error}")

    print("\n" + "=" * 70)
    print("PROMPT VARIATION 2 - CLEAR AND CONSTRAINED")
    print("=" * 70)

    print("\nSYSTEM PROMPT:")
    print(CONSTRAINED_SYSTEM_PROMPT.strip())

    try:
        constrained_response = get_response(
            CONSTRAINED_SYSTEM_PROMPT,
            USER_QUESTION,
        )

        print("\nMODEL OUTPUT:")
        print(constrained_response)

    except AuthenticationError:
        print("\nERROR: Authentication failed (401). Check your API key.")

    except RateLimitError:
        print("\nERROR: Rate limit exceeded (429). Try again later.")

    except APIConnectionError:
        print("\nERROR: Could not connect to the API endpoint.")

    except APIError as error:
        print(f"\nERROR: API error: {error}")

    print("\n" + "=" * 70)
    print("CHOSEN PROMPT")
    print("=" * 70)

    print(
        """
Variation 2 - Clear and constrained

Why it works better:
- Clearly defines the assistant's role and scope.
- Prevents unsupported or invented financial information.
- Sets a professional and neutral tone.
- Limits response length for easier review.
- Requires a consistent bullet-point format.
- Explicitly defines what to say when verified information is unavailable.
- Keeps customer-specific financial decisions with qualified professionals.
"""
    )


if __name__ == "__main__":
    main()