import tiktoken


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Use a modern OpenAI tokenizer.
# The exact tokenizer used by a provider/model may differ,
# so this script treats the encoding as a configurable choice.
ENCODING_NAME = "o200k_base"

# Example prices in USD per 1 million tokens.
# Replace these with the actual rates of your chosen provider/model.
INPUT_PRICE_PER_1M = 1.00
OUTPUT_PRICE_PER_1M = 2.00


# ---------------------------------------------------------
# Sample project texts
# ---------------------------------------------------------

samples = [
    {
        "name": "Short question",
        "input": (
            "What is the purpose of WealthGuard AI?"
        ),
        "output": (
            "WealthGuard AI helps relationship managers retrieve "
            "accurate and approved wealth-management information."
        ),
    },
    {
        "name": "Medium paragraph",
        "input": (
            "Wealth divisions manage extensive documentation such as "
            "policies, regulations, product brochures, and compliance "
            "guidelines. Relationship managers may spend significant "
            "time searching these resources when responding to customer "
            "inquiries, which can lead to inconsistent information and "
            "reliance on outdated documents."
        ),
        "output": (
            "WealthGuard AI uses evidence-based information retrieval "
            "to help relationship managers find relevant information "
            "from approved organizational documents."
        ),
    },
    {
        "name": "Long project document",
        "input": (
            "WealthGuard AI is an evidence-based AI assistant for wealth "
            "management. It enables relationship managers to access "
            "accurate, consistent, and current information from approved "
            "organizational documents. Wealth divisions manage extensive "
            "documentation including policies, regulations, product "
            "brochures, and compliance guidelines. Manual searching can "
            "lead to inconsistent information delivery, outdated document "
            "reliance, extended response times, and compliance risks. "
            "The system uses Retrieval-Augmented Generation to ground "
            "responses in approved organizational documents. It provides "
            "source citations so users can verify the information. "
            "Document filtering prioritizes valid and approved versions "
            "using metadata. Conflict detection identifies contradictions "
            "across documents. Human-in-the-loop design ensures that the "
            "system supports professional decision-making without "
            "replacing human judgment. The technology stack includes "
            "React.js, Tailwind CSS, Python, FastAPI, large language "
            "models, embedding models, LangChain or LlamaIndex, "
            "PostgreSQL, and a vector database such as Qdrant, Chroma, "
            "or pgvector. Document processing can use PyMuPDF and "
            "PDF/DOCX parsers. Docker can be used for consistent "
            "deployment environments. The system can support tax and "
            "regulatory queries, product information, policy queries, "
            "and document verification. The responsible AI approach "
            "prioritizes approved documentation, considers document "
            "versions and validity periods, provides supporting sources, "
            "flags conflicts or insufficient information, avoids "
            "unsupported financial recommendations, and keeps final "
            "customer-specific decisions with qualified professionals."
        ),
        "output": (
            "WealthGuard AI retrieves relevant approved information and "
            "provides a concise response with supporting sources. It is "
            "designed to improve consistency and transparency while "
            "keeping final customer-specific decisions with qualified "
            "professionals."
        ),
    },
]


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

encoding = tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text):
    """Return the number of tokens in a text string."""
    return len(encoding.encode(text))


def calculate_cost(input_tokens, output_tokens):
    """Calculate estimated cost in USD."""
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M

    return input_cost, output_cost, input_cost + output_cost


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("WEALTHGUARD AI - TOKEN AND COST ANALYSIS")
    print("=" * 70)

    print(f"\nTokenizer: {ENCODING_NAME}")
    print(f"Input price: ${INPUT_PRICE_PER_1M:.2f} per 1M tokens")
    print(f"Output price: ${OUTPUT_PRICE_PER_1M:.2f} per 1M tokens")

    total_input_tokens = 0
    total_output_tokens = 0

    results = []

    for sample in samples:
        input_text = sample["input"]
        output_text = sample["output"]

        input_tokens = count_tokens(input_text)
        output_tokens = count_tokens(output_text)

        input_chars = len(input_text)
        output_chars = len(output_text)

        input_cost, output_cost, total_cost = calculate_cost(
            input_tokens,
            output_tokens,
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        results.append({
            "name": sample["name"],
            "input_chars": input_chars,
            "input_tokens": input_tokens,
            "output_chars": output_chars,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        })

        print("\n" + "-" * 70)
        print(sample["name"])
        print("-" * 70)

        print(f"Input characters : {input_chars}")
        print(f"Input tokens     : {input_tokens}")
        print(f"Output characters: {output_chars}")
        print(f"Output tokens    : {output_tokens}")

        print(f"Input cost       : ${input_cost:.8f}")
        print(f"Output cost      : ${output_cost:.8f}")
        print(f"Total cost       : ${total_cost:.8f}")

    # -----------------------------------------------------
    # Overall totals
    # -----------------------------------------------------

    total_input_cost, total_output_cost, total_cost = calculate_cost(
        total_input_tokens,
        total_output_tokens,
    )

    print("\n" + "=" * 70)
    print("TOTAL")
    print("=" * 70)

    print(f"Total input tokens : {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

    print(f"Input cost         : ${total_input_cost:.8f}")
    print(f"Output cost        : ${total_output_cost:.8f}")
    print(f"Estimated total    : ${total_cost:.8f}")

    # -----------------------------------------------------
    # Length-token relationship
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("LENGTH VS TOKEN RELATIONSHIP")
    print("=" * 70)

    for result in results:
        input_chars = result["input_chars"]
        input_tokens = result["input_tokens"]

        ratio = input_chars / input_tokens if input_tokens else 0

        print(
            f"{result['name']}: "
            f"{input_chars} characters -> "
            f"{input_tokens} tokens "
            f"({ratio:.2f} characters/token)"
        )

    print(
        """
Observation:
Text length and token count generally increase together, but they
are not exactly proportional. Tokenizers split text according to
subword patterns rather than simply counting characters or words.
Long or uncommon words, punctuation, code, numbers, and other
languages can therefore produce different token-to-character ratios.
"""
    )


if __name__ == "__main__":
    main()