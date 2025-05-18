import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Transcript Search and Q&A")

option = st.radio("Choose Action", ["Search", "Ask"])

query = st.text_input("Enter your query")

if st.button("Submit") and query:
    if option == "Search":
        res = requests.get(f"{API_URL}/search", params={"q": query})
        results = res.json()
        st.subheader("Top Matching Chunks")
        for r in results:
            st.write(f"**Text**: {r['text']}")
            st.write(f"**Metadata**: {r['metadata']}")
            st.markdown("---")
    elif option == "Ask":
        res = requests.get(f"{API_URL}/ask", params={"q": query})
        answer = res.json()
        st.subheader("Question")
        st.write("original_query: ", answer["original_query"])
        st.write("rewritten_query: ", answer["rewritten_query"])
        st.subheader("Answer")
        st.write(answer["answer"])
        st.subheader("Source Chunks")
        for s in answer["sources"]:
            st.write(f"**Text**: {s['text']}")
            st.write(f"**Metadata**: {s['metadata']}")
            st.markdown("---")