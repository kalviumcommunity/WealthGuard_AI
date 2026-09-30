# Chat UI Sample Interaction

Run the UI with:

```bash
streamlit run streamlit_app.py
```

Enable **Use demo response** and ask:

> What is the annual management fee for the Secure Growth Plan?

Expected answer area:

> The Secure Growth Plan has an annual management fee of 1.2 percent.
>
> Capital protection is not guaranteed. Review the applicable product terms before investing.

Expected source area:

| Source | Chunk | Similarity |
| --- | --- | --- |
| product_brochure.txt | demo_product_0 | 0.912 |
| compliance.txt | demo_compliance_0 | 0.335 |

When the API is unavailable with demo mode off, the UI shows an error explaining
that the backend must be running. An empty question shows a validation warning,
and a successful API response with no results shows a no-match message.