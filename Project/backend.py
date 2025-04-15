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