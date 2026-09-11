import json
import logging
import os

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIError,
)

from prompts.templates import render_wealthguard_prompt


# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL")

# Validate configuration
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")

if not BASE_URL:
    raise RuntimeError("OPENAI_BASE_URL is missing from .env")

if not MODEL:
    raise RuntimeError("OPENAI_MODEL is missing from .env")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# Create OpenAI-compatible client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


def main():
    context = """
WealthGuard AI is an evidence-based assistant for relationship managers.
It helps users find and understand information from approved
wealth-management knowledge sources.
"""

    question = "What is the purpose of WealthGuard AI?"

    user_prompt = render_wealthguard_prompt(
        context=context,
        question=question,
    )

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    logger.info("Sending chat completion request")
    logger.info(
        "Request messages: %s",
        json.dumps(messages, indent=2),
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        # Print model response
        reply = response.choices[0].message.content

        print("\nModel response:")
        print(reply)

        # Log response
        logger.info(
            "Response payload: %s",
            response.model_dump_json(indent=2),
        )

        # Log token usage
        if response.usage:
            logger.info(
                "Token usage: prompt=%s, completion=%s, total=%s",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
            )
        else:
            logger.info("Token usage: not available")

    except AuthenticationError:
        logger.error(
            "Authentication failed (401): "
            "Check your API key."
        )

    except RateLimitError:
        logger.error(
            "Rate limit exceeded (429): "
            "Please wait and try again later."
        )

    except APIConnectionError:
        logger.error(
            "Connection error: "
            "Could not connect to the API endpoint."
        )

    except APIError as error:
        logger.error("API error: %s", error)

    except Exception as error:
        logger.error("Unexpected error: %s", error)


if __name__ == "__main__":
    main()