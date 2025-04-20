from enum import Enum
from typing import Type
from pydantic import BaseModel, Field
import openai
from google import genai
import os



# Base Incident Model
class IncidentType(str, Enum):
    FIRE = "FIRE"
    MEDICAL_ASSISTANCE = "MEDICAL_ASSISTANCE"
    THEFT = "THEFT"
    GENERAL = "GENERAL"

class WindDirectionType(str, Enum):
    FROM_SOUTH_TO_WEST = "from South to West"
    FROM_WEST_TO_SOUTH = "from West to South"
    FROM_SOUTH_TO_NORTH = "from South to North"
    FROM_NORTH_TO_SOUTH = "from North to South"
    FROM_EAST_TO_WEST = "from East to West"
    FROM_WEST_TO_EAST = "from West to East"
    FROM_NORTH_EAST = "from North to East"
    FROM_EAST_TO_NORTH = "from East to North"

class Incident(BaseModel):
    incident_type: IncidentType | None = Field(..., description="Type of incident")
    caller_name: str | None = Field(..., description="Name of the caller reporting the incident")
    caller_badge: int | None = Field(..., description="Badge number of the caller")

    model_config = {
        "json_schema_extra": {
            "example": {
                "incident_type": "GENERAL",
                "caller_name": "John Doe",
                "caller_badge": "73987123"
            }
        },
        "extra": "forbid"  # This sets additionalProperties to false
    }



class FireIncident(Incident):
    location: str | None = Field(..., description="Location where the incident occurred")
    wind_direction: WindDirectionType | None = Field(
        ...,
        description="Wind direction range (e.g., 'from south to west')"
    )
    injury_count: int |None  = Field(..., description="Number of people injured")
    timestamp: str | None = Field(..., description="Time of the incident as XX:XX AM or XX:XX PM")
    description: str | None = Field(..., description="Detailed description of the incident")

    def is_dispatch_ready(self) -> bool:
        required_fields = {
            'location': self.location,
            'wind_direction': self.wind_direction
        }
        return all([v is not None for v in required_fields.values()])

    model_config = {
        "json_schema_extra": {
            "example": {
                "incident_type": "Fire",
                "location": "123 Main Street, Building A",
                "wind_direction": "from south to west",
                "injury_count": 2,
                "timestamp": "10:00 AM",
                "description": "Kitchen fire caused by unattended cooking",
                "caller_name": "John Doe",
                "caller_badge": "4324123"
            },
            "dispatch_required_fields": [
                "location",
                "wind_direction"
            ]
        },
        "extra": "forbid"
    }


class MedicalAssistanceIncident(Incident):
    location: str | None = Field(..., description="Location where the incident occurred")
    timestamp: str | None = Field(..., description="Time of the incident as XX:XX AM or XX:XX PM")
    number_of_people_involved: int | None = Field(..., description="Number of people involved")
    illness_nature: str | None = Field(..., description="Nature of illness")
    patient_details: str | None = Field(..., description="Details about the patient")

    def is_dispatch_ready(self) -> bool:
        required_fields = {
            'location': self.location,
            'illness_nature': self.illness_nature
        }
        return all([v is not None for v in required_fields.values()])

    model_config = {
        "json_schema_extra": {
            "example": {
                "incident_type": "MEDICAL_ASSISTANCE",
                "location": "Office Building C, 2nd Floor",
                "timestamp": "10:00 AM",
                "people_involved": 1,
                "illness_nature": "Chest pain",
                "patient_details": "Male, 45 years old, conscious",
                "caller_name": "Sarah Davis",
                "caller_badge": "67890"
            },
            "dispatch_required_fields": [
                "location",
                "illness_nature"
            ]
        },
        "extra": "forbid"
    }

class TheftIncident(Incident):
    location: str | None = Field(..., description="Location where theft occurred")
    timestamp: str | None = Field(..., description="Date and time of theft discovery")
    last_seen: str | None = Field(..., description="Last time and location items were seen")
    incident_description: str | None = Field(..., description="Description of the incident")
    suspect_info: str | None = Field(..., description="Information about suspects/vehicles if any")
    stolen_items: str | None = Field(..., description="Description of stolen items")
    items_value: float | None = Field(..., description="Estimated value of stolen items")

    def is_dispatch_ready(self) -> bool:
        required_fields = {
            'location': self.location,
            'incident_description': self.incident_description,
            'stolen_items': self.stolen_items
        }
        return all([v is not None for v in required_fields.values()])

    model_config = {
        "json_schema_extra": {
            "example": {
                "incident_type": "THEFT",
                "location": "Office Building C, 2nd Floor",
                "timestamp": "10:00 AM",
                "last_seen": "2024-01-23 17:00",
                "incident_description": "Tools missing from locked cabinet",
                "suspect_info": "No suspects identified",
                "stolen_items": "Power tools and hand tools",
                "items_value": 5000.00,
                "caller_name": "William Carter",
                "caller_badge": "12345"
            },
            "dispatch_required_fields": [
                "location",
                "incident_description",
                "stolen_items"
            ]
        },
        "extra": "forbid"
        }


INCIDENT_TYPES_MAP = {
    "FIRE": FireIncident,
    "MEDICAL_ASSISTANCE": MedicalAssistanceIncident,
    "THEFT": TheftIncident,
    "GENERAL":  Incident  # Fallback 
}

def create_openai_generator(model_name: str, model_class: BaseModel, base_url:str|None):
    client = openai.OpenAI(base_url=base_url)

    def generator(prompt: str, max_tokens: int = 100, temperature: float = 0.0):
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": f"You must respond with a valid JSON object that matches this Pydantic model structure: {model_class.schema_json()}"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=model_class
        )
        message = response.choices[0].message
        if message.parsed:
            return model_class.parse_obj(message.parsed)
        else:
            raise ValueError(f"Failed to parse response: {message.refusal}")
        

    return generator

def create_gemini_generator(model_name: str, model_class: Type[BaseModel]) -> callable:
    """
    Creates a generator function that uses the Gemini API to generate structured output
    based on a Pydantic model.

    Args:
        model_name: The name of the Gemini model to use.
        model_class: The Pydantic model class defining the desired output structure.

    Returns:
        A generator function that takes a prompt and returns an instance of the model class.
    """

    # google API doesn't support model_config in pydantic model, so it will be removed
    if 'model_config' in model_class.__dict__:
        delattr(model_class, 'model_config')

    def generator(prompt: str, max_tokens: int = 100, temperature: float = 0.0):
        """
        Generates structured output using the Gemini API.

        Args:
            prompt: The prompt to send to the Gemini model.
            max_tokens: The maximum number of tokens to generate (not directly used in this implementation).
            temperature: The temperature to use for generation (not directly used in this implementation).

        Returns:
            An instance of the model class, or None if parsing fails.
        """
        try:
            client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    'system_instruction':f"You must respond with a valid JSON object that matches this Pydantic model structure: {model_class.schema_json()}",
                    'response_mime_type': 'application/json',
                    'response_schema': model_class,
                    'max_output_tokens':max_tokens,
                    'temperature':temperature
                },
            )


            # Gemini may return an empty response or have issues parsing
            if response:
                return response.parsed
            else:
                raise ValueError(f"Failed to parse response: {response.text}")

        except Exception as e:
            print(f"Error generating or parsing response: {e}")
            return None

    return generator

