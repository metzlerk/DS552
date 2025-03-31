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

# Load the Mistral model and tokenizer
try:
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, huggingface_token)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

# Initialize FastAPI app
app = FastAPI()

# Models for input
class EncounterRequest(BaseModel):
    party_level: int
    terrain: str
    theme: str

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
            f"Generate a D&D encounter for a party of level {request.party_level} "
            f"in a {request.terrain} terrain with a {request.theme} theme. "
            f"Include details about the monsters and their challenge ratings."
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

        # Attempt to extract monsters from the generated text using AI
        try:
            # Example: Use the AI-generated text to extract monster details
            # This assumes the generated text contains structured monster information
            generated_lines = generated_text.split("\n")
            monsters = []
            for line in generated_lines:
                if "CR" in line:  # Look for lines containing challenge rating (CR)
                    parts = line.split("-")
                if len(parts) == 2:
                    name = parts[0].strip()
                    cr = parts[1].strip().replace("CR", "").strip()
                    monsters.append({"name": name, "challenge_rating": float(cr)})
                if not monsters:
                    raise ValueError("No monsters found in AI-generated text.")
        except Exception:
            # Fallback: Load monsters from a local file (monsters.py)
            try:
                monsters = get_monsters(request.party_level, request.terrain, request.theme)
            except ImportError as e:
                raise RuntimeError(f"Failed to load fallback monsters: {e}")

        # Return a structured response
        return {
            "encounter": {
                "description": generated_text,
                "monsters": monsters,
                "terrain": request.terrain,
                "theme": request.theme,
                "difficulty": "Medium",
                "treasure": "50 gold coins and a magical sword",
                "tactics": "The goblins will try to flank the party while the wolf charges head-on."
            }
        }
    except Exception as e:
        logger.error(f"Error generating encounter: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating encounter: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
