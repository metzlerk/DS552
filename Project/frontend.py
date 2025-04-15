from transformers import pipeline

# Load the fine-tuned model and tokenizer
generator = pipeline("text-generation", model="./fine_tuned_model", tokenizer="./fine_tuned_model")

# Streamlit app
import streamlit as st

st.title("AI-Powered D&D Content Generator")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Generate Encounter", "Generate Dungeon", "Generate Character"])

if page == "Generate Encounter":
    st.header("Generate Encounter")
    prompt = st.text_area("Enter your encounter prompt:", value="Generate a D&D encounter for a party of level 5 in a forest terrain with a mystery theme.")
    if st.button("Generate"):
        response = generator(prompt, max_length=200, num_return_sequences=1)
        st.text_area("Generated Encounter:", value=response[0]["generated_text"], height=200)

elif page == "Generate Dungeon":
    st.header("Generate Dungeon")
    prompt = st.text_area("Enter your dungeon prompt:", value="Generate a D&D dungeon with 5 rooms, traps, and puzzles.")
    if st.button("Generate"):
        response = generator(prompt, max_length=200, num_return_sequences=1)
        st.text_area("Generated Dungeon:", value=response[0]["generated_text"], height=200)

elif page == "Generate Character":
    st.header("Generate Character")
    prompt = st.text_area("Enter your character prompt:", value="Generate a D&D character named Aragon with a ranger background.")
    if st.button("Generate"):
        response = generator(prompt, max_length=200, num_return_sequences=1)
        st.text_area("Generated Character:", value=response[0]["generated_text"], height=200)