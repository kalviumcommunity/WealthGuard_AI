import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/query"


st.set_page_config(
    page_title="WealthGuard AI",
    page_icon="💼"
)


st.title("💼 WealthGuard AI")
st.write(
    "Evidence-based assistant for wealth management information."
)


# User input

question = st.text_input(
    "Ask your question:"
)


if st.button("Submit"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        try:
            with st.spinner("Searching knowledge base..."):

                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=30
                )


            if response.status_code != 200:
                st.error(
                    f"API Error: {response.status_code}"
                )

            else:

                data = response.json()


                st.success("Answer generated")


                # Answer section

                st.subheader("Answer")

                results = data.get(
                    "results",
                    []
                )


                if results:

                    answer_text = results[0]["text"]

                    st.write(answer_text)


                else:

                    st.info(
                        "I don't have enough verified information "
                        "to answer this reliably."
                    )


                # Sources

                st.subheader("Retrieved Sources")


                for index, item in enumerate(results):

                    with st.expander(
                        f"Source {index+1}"
                    ):

                        st.write(
                            "Chunk ID:",
                            item.get("chunk_id")
                        )

                        st.write(
                            "Source:",
                            item.get("source")
                        )

                        st.write(
                            "Similarity:",
                            item.get("similarity")
                        )

                        st.write(
                            "Content:"
                        )

                        st.write(
                            item.get("text")
                        )


        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to RAG API. "
                "Start FastAPI server first."
            )


        except Exception as error:

            st.error(
                f"Unexpected error: {error}"
            )