from pathlib import Path

from prompts.templates import render_wealthguard_prompt


def main():
    examples = [
        {
            "name": "Product Information Query",
            "context": """
Product: Secure Growth Plan
Annual management fee: 1.2%
Minimum investment: ₹50,000
Lock-in period: 3 years
Capital protection: Not guaranteed
""",
            "question": "What is the annual management fee?",
        },
        {
            "name": "Investment Query",
            "context": """
Product: Secure Growth Plan
Minimum investment: ₹50,000
Lock-in period: 3 years
Early withdrawal: Subject to applicable terms and conditions
""",
            "question": "What is the minimum investment and lock-in period?",
        },
        {
            "name": "Insufficient Information Query",
            "context": """
Product: Secure Growth Plan
Annual management fee: 1.2%
Minimum investment: ₹50,000
Lock-in period: 3 years
""",
            "question": "What tax benefit is available on this product?",
        },
    ]

    output = []

    output.append("WEALTHGUARD AI - PROMPT TEMPLATE EXAMPLES")
    output.append("=" * 60)

    for example in examples:
        rendered_prompt = render_wealthguard_prompt(
            context=example["context"].strip(),
            question=example["question"],
        )

        output.append("\n" + "=" * 60)
        output.append(example["name"])
        output.append("=" * 60)
        output.append(rendered_prompt.strip())

    output.append("\n" + "=" * 60)
    output.append("TEMPLATE REUSE")
    output.append("=" * 60)
    output.append(
        "The same render_wealthguard_prompt() function is reused "
        "by chat_completion.py and batch_cli.py."
    )

    Path("examples").mkdir(exist_ok=True)

    Path("examples/prompt_renders.txt").write_text(
        "\n".join(output),
        encoding="utf-8",
    )

    print("Example prompt renders saved to:")
    print("examples/prompt_renders.txt")


if __name__ == "__main__":
    main()