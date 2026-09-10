import os

from dotenv import load_dotenv
from openai import OpenAI
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIError


# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

# Use gpt-4o for parameter experiments
MODEL = "gpt-4o"

if not API_KEY or not BASE_URL:
    raise ValueError(
        "Missing OPENAI_API_KEY or OPENAI_BASE_URL in .env"
    )


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


SYSTEM_PROMPT = """
You are WealthGuard AI, an evidence-based assistant for relationship managers.

Answer only using the information provided in the user prompt.

Do not invent facts, regulations, product features, or numbers.

If the information is insufficient, clearly say that the information is insufficient.

Keep the answer professional, concise, and factual.
"""


USER_PROMPT = """
According to the approved product information below:

Product: Secure Growth Plan

Annual management fee: 1.2%

Minimum investment: ₹50,000

Lock-in period: 3 years

Early withdrawal: Subject to applicable terms and conditions

Capital protection: Not guaranteed

Question:

What are the key points a relationship manager should explain to a customer?
"""


def run_experiment(
    name,
    temperature,
    max_tokens=None,
    top_p=None,
    stop=None
):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Temperature: {temperature}")

    if max_tokens is not None:
        print(f"Max tokens: {max_tokens}")

    if top_p is not None:
        print(f"Top P: {top_p}")

    if stop is not None:
        print(f"Stop: {stop}")

    try:
        parameters = {
            "model": MODEL,
            "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": USER_PROMPT
            }
        ],
        "temperature": temperature,
        "max_tokens": 100,
}

        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens

        if top_p is not None:
            parameters["top_p"] = top_p

        if stop is not None:
            parameters["stop"] = stop

        response = client.chat.completions.create(**parameters)

        output = response.choices[0].message.content

        print("\nOutput:")
        print(output)

        print("\nUsage:")
        print(f"Input tokens: {response.usage.prompt_tokens}")
        print(f"Output tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")

        return output

    except AuthenticationError:
        print("Authentication error. Check your API key.")

    except RateLimitError as e:
        print(f"Rate limit/quota error: {e}")

    except APIConnectionError:
        print("Connection error. Check your internet connection.")

    except APIError as e:
        print(f"API error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    print("=" * 70)
    print("WEALTHGUARD AI - PARAMETER EXPERIMENTS")
    print("=" * 70)

    # ---------------------------------------------------------
    # TASK 1: TEMPERATURE
    # ---------------------------------------------------------

    print("\n\nTASK 1 - TEMPERATURE EXPERIMENT")

    run_experiment(
        "Temperature = 0.0 (Stable / Deterministic)",
        temperature=0.0
    )

    run_experiment(
        "Temperature = 0.7 (More Varied)",
        temperature=0.7
    )

    run_experiment(
        "Temperature = 1.2 (Creative / More Variable)",
        temperature=1.2
    )

    # ---------------------------------------------------------
    # TASK 2: MAX TOKENS
    # ---------------------------------------------------------

    print("\n\nTASK 2 - MAX TOKENS EXPERIMENT")

    run_experiment(
        "max_tokens = 30 (Short Response Limit)",
        temperature=0.0,
        max_tokens=30
    )

    run_experiment(
        "max_tokens = 100 (Longer Response Limit)",
        temperature=0.0,
        max_tokens=100
    )

    # ---------------------------------------------------------
    # TASK 3: TOP P
    # ---------------------------------------------------------

    print("\n\nTASK 3 - TOP P EXPERIMENT")

    run_experiment(
        "top_p = 0.2 (Narrower Token Selection)",
        temperature=0.0,
        top_p=0.2
    )

    run_experiment(
        "top_p = 1.0 (Full Probability Distribution)",
        temperature=0.0,
        top_p=1.0
    )

    # ---------------------------------------------------------
    # TASK 4: RECOMMENDED SETTINGS
    # ---------------------------------------------------------

    print("\n\nTASK 4 - RECOMMENDED SETTINGS")
    print("=" * 70)

    print("""
Recommended settings for a grounded WealthGuard AI task:

Temperature:
Use a low temperature such as 0.0-0.2.
This reduces unnecessary variation and makes factual answers
more stable and predictable.

max_tokens:
Use a sensible limit such as 100-200 tokens for concise
relationship-manager answers. The limit should be large enough
to provide complete answers without allowing unnecessary rambling.

top_p:
Use the default 1.0 unless there is a specific reason to tune it.
Temperature and top_p should generally not be tuned aggressively
at the same time.

stop:
Use stop sequences only when a specific response boundary is
needed. It is optional for normal factual answers.

Overall recommendation:
temperature = 0.0-0.2
max_tokens = 100-200
top_p = 1.0
stop = None unless required
""")


if __name__ == "__main__":
    main()