import streamlit as st
import requests

st.title("AI-Powered D&D Content Generator")

# Encounter Generation
st.header("Generate Encounter")
party_level = st.number_input("Party Level", min_value=1, max_value=20, value=5)
terrain = st.text_input("Terrain", value="forest")
theme = st.text_input("Theme", value="mystery")

if st.button("Generate Encounter"):
    response = requests.post(
        "http://127.0.0.1:8000/generate/encounter",
        json={"party_level": party_level, "terrain": terrain, "theme": theme}
    )
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Error generating encounter")

# Dungeon Generation
st.header("Generate Dungeon")
rooms = st.number_input("Number of Rooms", min_value=1, max_value=50, value=5)
traps = st.checkbox("Include Traps")
puzzles = st.checkbox("Include Puzzles")

if st.button("Generate Dungeon"):
    response = requests.post(
        "http://127.0.0.1:8000/generate/dungeon",
        json={"rooms": rooms, "traps": traps, "puzzles": puzzles}
    )
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Error generating dungeon")

# Character Generation
st.header("Generate Character")
name = st.text_input("Character Name", value="Aragorn")
background = st.text_input("Background", value="Ranger")
motivation = st.text_input("Motivation", value="Protect the realm")

if st.button("Generate Character"):
    response = requests.post(
        "http://127.0.0.1:8000/generate/character",
        json={"name": name, "background": background, "motivation": motivation}
    )
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Error generating character")