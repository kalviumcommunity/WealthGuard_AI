import json
import logging
import os

import tiktoken
from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIError,
)

from prompts.templates import render_wealthguard_prompt


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ENCODING_NAME = "o200k_base"

# Keep the history safely below the model context limit.
MAX_HISTORY_TOKENS = 6000

# Keep the newest messages when trimming.
RECENT_MESSAGE_TOKENS = 4000


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

encoding = tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text):
    """
    Count the number of tokens in a text string.
    """
    if not text:
        return 0

    return len(encoding.encode(text))


def count_message_tokens(message):
    """
    Count tokens for one chat message.
    """

    content = message.get("content", "")

    role = message.get("role", "")

    return count_tokens(role) + count_tokens(content)


def total_tokens(messages):
    """
    Count the total tokens used by all messages.
    """

    return sum(count_message_tokens(message) for message in messages)


# ---------------------------------------------------------
# History management
# ---------------------------------------------------------

def trim_history(messages, max_tokens=MAX_HISTORY_TOKENS):
    """
    Remove the oldest messages until the history fits
    inside the configured token budget.

    The system message is always preserved.
    """

    if not messages:
        return messages

    system_message = None
    other_messages = []

    for message in messages:
        if message.get("role") == "system" and system_message is None:
            system_message = message
        else:
            other_messages.append(message)

    trimmed_messages = []

    if system_message:
        trimmed_messages.append(system_message)

    remaining_tokens = max_tokens

    if system_message:
        remaining_tokens -= count_message_tokens(system_message)

    # Keep the newest messages first.
    for message in reversed(other_messages):

        message_tokens = count_message_tokens(message)

        if message_tokens <= remaining_tokens:
            trimmed_messages.insert(
                1 if system_message else 0,
                message
            )

            remaining_tokens -= message_tokens

        else:
            break

    return trimmed_messages


def summarise_old_messages(messages):
    """
    Create a compact summary of older conversation messages.

    This function does not call the model. It creates a simple
    local summary that can be kept inside the conversation.
    """

    if not messages:
        return ""

    parts = []

    for message in messages:

        role = message.get("role", "unknown")
        content = message.get("content", "")

        if content:
            parts.append(f"{role}: {content}")

    if not parts:
        return ""

    summary = "Previous conversation summary:\n"

    summary += "\n".join(parts)

    return summary


def manage_history(messages, max_tokens=MAX_HISTORY_TOKENS):
    """
    Measure the history and trim it when it exceeds the budget.
    """

    before_tokens = total_tokens(messages)

    logger.info(
        "History tokens before management: %s",
        before_tokens,
    )

    if before_tokens <= max_tokens:
        return messages

    logger.info(
        "History exceeded %s tokens. Trimming old messages.",
        max_tokens,
    )

    managed_messages = trim_history(
        messages,
        max_tokens,
    )

    after_tokens = total_tokens(managed_messages)

    logger.info(
        "History tokens after management: %s",
        after_tokens,
    )

    return managed_messages


# ---------------------------------------------------------
# Chat function
# ---------------------------------------------------------

def ask(history, user_message):
    """
    Add a user message, manage the context window,
    send the request, and store the assistant response.
    """

    history.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    # Measure and manage the history before the API request.
    history = manage_history(history)

    logger.info(
        "Sending request with %s messages and %s tokens",
        len(history),
        total_tokens(history),
    )

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
        )

        reply = response.choices[0].message.content

        history.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        return reply, history

    except AuthenticationError:

        logger.error(
            "Authentication failed (401): Check your API key."
        )

        return None, history

    except RateLimitError:

        logger.error(
            "Rate limit exceeded (429)."
        )

        return None, history

    except APIConnectionError:

        logger.error(
            "Could not connect to the API endpoint."
        )

        return None, history

    except APIError as error:

        logger.error(
            "API error: %s",
            error,
        )

        return None, history

    except Exception as error:

        logger.error(
            "Unexpected error: %s",
            error,
        )

        return None, history


# ---------------------------------------------------------
# Demo conversation
# ---------------------------------------------------------

def main():

    context = """
    WealthGuard AI is an evidence-based assistant for
    relationship managers.

    It helps users find and understand information from
    approved wealth-management knowledge sources.
    """

    first_prompt = render_wealthguard_prompt(
        context=context,
        question="What is the purpose of WealthGuard AI?",
    )

    history = [
        {
            "role": "system",
            "content": """
            You are WealthGuard AI.

            Answer questions using the provided context.
            Do not invent unsupported financial information.
            Keep responses clear and evidence-based.
            """,
        }
    ]

    # -----------------------------------------------------
    # First turn
    # -----------------------------------------------------

    reply, history = ask(
        history,
        first_prompt,
    )

    if reply:

        print("\nModel response:")
        print(reply)

    # -----------------------------------------------------
    # Second turn
    # -----------------------------------------------------

    reply, history = ask(
        history,
        "Can you explain that in simple terms?",
    )

    if reply:

        print("\nModel response:")
        print(reply)

    # -----------------------------------------------------
    # Third turn
    # -----------------------------------------------------

    reply, history = ask(
        history,
        "Why is approved documentation important?",
    )

    if reply:

        print("\nModel response:")
        print(reply)

    # -----------------------------------------------------
    # Final history information
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)

    print(
        "Messages:",
        len(history),
    )

    print(
        "Total tokens:",
        total_tokens(history),
    )

    print("\nHistory:")

    print(
        json.dumps(
            history,
            indent=2,
        )
    )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    main()