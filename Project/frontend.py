import streamlit as st
import requests
from fastapi import FastAPI

app = FastAPI(
    title="AI-Powered D&D 5th Edition Module Generator",
    description="""
    ## Instructions
    Welcome to the AI-Powered D&D 5th Edition Module Generator! This API helps you generate:
    - **Encounters**: Create balanced combat and non-combat encounters.
    - **Dungeons**: Design dungeons with rooms, traps, and puzzles.
    - **Characters**: Generate unique NPCs with names, backgrounds, and motivations.

    ### How to Use
    1. Use the `/generate/encounter` endpoint to create encounters.
    2. Use the `/generate/dungeon` endpoint to design dungeons.
    3. Use the `/generate/character` endpoint to generate NPCs.

    Visit `/docs` for the interactive API documentation.
    """,
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI-Powered D&D Content Generator!",
        "instructions": {
            "encounter": "Use /generate/encounter to create encounters.",
            "dungeon": "Use /generate/dungeon to design dungeons.",
            "character": "Use /generate/character to generate NPCs.",
            "docs": "Visit /docs for interactive API documentation."
        }
    }

st.title("AI-Powered D&D Content Generator")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Generate Encounter", "Generate Dungeon", "Generate Character"])

# Instructions
st.sidebar.info(
    """
    **Instructions**:
    - Use the sidebar to navigate between sections.
    - Fill in the required fields and click the corresponding button to generate content.
    - Results will appear below the form.
    """
)

# Encounter Generation
if page == "Generate Encounter":
    st.header("Generate Encounter")
    col1, col2, col3 = st.columns(3)
    party_level = col1.number_input("Party Level", min_value=1, max_value=20, value=5)
    terrain = col2.text_input("Terrain", value="forest")
    theme = col3.text_input("Theme", value="mystery")

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
elif page == "Generate Dungeon":
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
elif page == "Generate Character":
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