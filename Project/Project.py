from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import os
import uvicorn
from monsters import get_monsters
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Authenticate with Hugging Face
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if not huggingface_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set in the environment variables.")
login(token=huggingface_token)

# Load the GPT-Neo model and tokenizer
try:
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

# Initialize FastAPI app
app = FastAPI()

# Models for input
class EncounterRequest(BaseModel):
    party_level: int
    terrain: str
    theme: str

class DungeonRequest(BaseModel):
    rooms: int
    traps: bool
    puzzles: bool

class CharacterRequest(BaseModel):
    name: str
    background: str
    motivation: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered D&D Content Generator! Proceed to the current URL plus /docs to see play with the encounter generator!"}

@app.post("/generate/encounter")
def generate_encounter(request: EncounterRequest):
    try:
        # Log the input request
        logger.info(f"Received request: {request}")

        # Create the prompt for the Mistral model
        prompt = (
            f"You are a Dungeons & Dragons encounter generator. Create a detailed encounter for a party of level {request.party_level} "
            f"in a {request.terrain} terrain with a {request.theme} theme. "
            f"Include the following details:\n"
            f"- A brief narrative description of the encounter.\n"
            f"- The monsters involved, their names, and challenge ratings.\n"
            f"- Any environmental hazards or special features of the terrain.\n"
            f"- Suggested tactics for the monsters.\n"
            f"- Any treasure or rewards the players might find."
        )

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate text using the Mistral model
        outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Log the generated text
        logger.info(f"Generated encounter: {generated_text}")

        # Post-process the generated text
        generated_text = generated_text.replace("\n", " ").strip()

        # Extract key sections (e.g., monsters, tactics, treasure) using simple parsing
        if "Monsters:" in generated_text:
            monsters_section = generated_text.split("Monsters:")[1].split("Tactics:")[0].strip()
        else:
            monsters_section = "No monsters found."

        if "Tactics:" in generated_text:
            tactics_section = generated_text.split("Tactics:")[1].split("Treasure:")[0].strip()
        else:
            tactics_section = "No tactics provided."

        if "Treasure:" in generated_text:
            treasure_section = generated_text.split("Treasure:")[1].strip()
        else:
            treasure_section = "No treasure mentioned."

        # Determine monsters
        if not monsters_section or monsters_section == "No monsters found.":
            monsters = get_monsters(request.party_level, request.terrain, request.theme)
        else:
            monsters = [{"name": "Goblin", "challenge_rating": 1}, {"name": "Wolf", "challenge_rating": 2}]

        # Return the structured response
        return {
            "encounter": {
                "description": generated_text,
                "monsters": monsters,
                "tactics": tactics_section,
                "treasure": treasure_section,
                "terrain": request.terrain,
                "theme": request.theme
            }
        }
    except Exception as e:
        logger.error(f"Error generating encounter: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating encounter: {e}")

@app.post("/generate/dungeon")
def generate_dungeon(request: DungeonRequest):
    try:
        # Create the prompt for the Mistral model
        prompt = (
            f"Generate a D&D dungeon with {request.rooms} rooms. "
            f"Include {'traps' if request.traps else 'no traps'} and "
            f"{'puzzles' if request.puzzles else 'no puzzles'}. "
            f"Provide details about the layout, enemies, and treasure."
        )

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate text using the Mistral model
        outputs = model.generate(**inputs, max_length=300, num_return_sequences=1)

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the structured response
        return {
            "dungeon": {
                "description": generated_text,
                "rooms": request.rooms,
                "traps": request.traps,
                "puzzles": request.puzzles
            }
        }
    except Exception as e:
        logger.error(f"Error generating dungeon: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating dungeon: {e}")

@app.post("/generate/character")
def generate_character(request: CharacterRequest):
    try:
        # Create the prompt for the Mistral model
        prompt = (
            f"Generate a D&D character named {request.name}. "
            f"The character's background is {request.background}, and their motivation is {request.motivation}. "
            f"Provide details about their personality, abilities, and role in the story."
        )

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate text using the Mistral model
        outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the structured response
        return {
            "character": {
                "description": generated_text,
                "name": request.name,
                "background": request.background,
                "motivation": request.motivation
            }
        }
    except Exception as e:
        logger.error(f"Error generating character: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating character: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
