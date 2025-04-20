from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from typing import Dict, Any,List
from incident_types import Incident, INCIDENT_TYPES_MAP, create_openai_generator, create_gemini_generator

BASE_URL= None # None means it will use openai base url
OPERATOR_MODEL_NAME ="gemini-2.0-flash"
OBSERVER_MODEL_NAME ="gemini-2.0-flash"


RESPONSE_PLANS = {
    "FIRE": """
Steps:
1- Ask the caller about the Incident Location (Required - string).
2- Ask the caller about the Wind direction (Required - string, e.g., 'from south to west').
3- Ask the caller about the Number of injuries (Optional - integer).
4- Ask the caller about the TIME of the Incident (Optional - string, format XX:XX AM/PM).
5- Ask the caller about the Description of the incident (i.e., about the fire, the cause, or any related thing) (Optional - string).
6- Ask the caller about the Caller's name (Required - string).
7- Ask the caller about the Caller's badge number (Required - integer).
8- Summarize the previous points to confirm accuracy with the caller.
9- If ALL the required information (Location, Wind Direction, Caller Name, Caller Badge) is correct, use the appropriate function to save the case information.
""",

    "MEDICAL_ASSISTANCE": """
Steps:
1- Ask the caller about the Incident Location (Required - string).
2- Ask the caller about the TIME of the Incident (Optional - string, format XX:XX AM/PM).
3- Ask the caller about the Number of people involved (Optional - integer).
4- Ask the caller about the Nature of illness (Required - string).
5- Ask the caller about the Patient details (Optional - string).
6- Ask the caller about the Caller's name (Required - string).
7- Ask the caller about the Caller's badge number (Required - integer).
8- Summarize the previous points to confirm accuracy with the caller.
9- If ALL the required information (Location, Illness Nature, Caller Name, Caller Badge) is correct, use the appropriate function to save the case information.
""",

    "THEFT": """
Steps:
1- Ask the caller about the Location where the theft occurred (Required - string).
2- Ask the caller about the TIME of theft discovery (Optional - string, date and time).
3- Ask the caller about the Last time and location items were seen (Optional - string).
4- Ask the caller about the Description of the incident (Required - string).
5- Ask the caller about the Information about suspects/vehicles if any (Optional - string).
6- Ask the caller about the Description of stolen items (Required - string).
7- Ask the caller about the Estimated value of stolen items (Optional - float).
8- Ask the caller about the Caller's name (Required - string).
9- Ask the caller about the Caller's badge number (Required - integer).
10- Summarize the previous points to confirm accuracy with the caller.
11- If ALL the required information (Location, Incident Description, Stolen Items, Caller Name, Caller Badge) is correct, use the appropriate function to save the case information.
""",
"GENERAL":"""
1- Ask the caller about the Caller's name (Required - string).
2- Ask the caller about the Caller's badge number (Required - integer).
"""
}


OPERATOR_SYSTEM_PROMPT =f"""

You are a 911 helpful assistant with tool calling capabilities.
Think very carefully before calling tools and use ONLY one tool at a time.
Follow these instructions very carefully:

1. Begin with: Hello, this is 911 emergency center. How may I assist you?

2. Determine the type of incident (This is all the classes: [{','.join(INCIDENT_TYPES_MAP.keys())}]).

3. Once you have determined incident type, quickily get the instructions for the incident type by caling 'get_instructions_for_incident' while keeping these important guidelines in mind:
    - Ask only one question at a time. Do not overwhelm the caller by asking multiple questions at once.
    - If the caller gave you some of the informations in the steps DON'T ask him again about it.
    - If any answer is unclear or incomplete, ask for clarification before proceeding.


4. Follow the instructions for the incident type.

5. Maintain a calm, professional tone throughout the call.

6. Be Concise and Clear.
"""

OBSERVER_SYSTEM_PROMPT ="""
Extract the fields from the emergency call transcript:

Important:
- Only fill fields if they are explicitly mentioned else put null
- Be precise with the selection


Example Input:
911 Operator: 911, what's your emergency?

Caller: My name is Sarah Jennings. My house is on fire! Please help us!

911 Operator: Okay, Sarah, stay calm. Can you give me your address, please?

Example Output:
{{
"incident_type": "Fire",
"caller_name": "Sarah Jennings,
}}

Now extract from this transcript:
{}
"""


# Abstract functions without logic, just descriptions
@tool
def save_emergency_case_info(gathered_info: str) -> str:
    """
    Save information about an emergency case reported by a caller.
    
    Args:
        gathered_info (str): All the necessary information.
    
    Returns:
        str: Confirmation message.
    """
    pass

@tool
def get_instructions_for_incident(incident_type: str) -> str:
    """
    Get the list of required information that should be gathered or instructions to follow for an incident type.
    
    Args:
        incident_type (str): The determined incident type.
    
    Returns:
        str: Instructions for handling the incident.
    """
    pass





class Operator:
    def __init__(self, temperature: float = 0.01):
        self.functions = {
            'save_emergency_case_info': self.save_emergency_case_info, # workaround to update the state
            'get_instructions_for_incident':self.get_instructions_for_incident,
        }
        if 'gemini' in OPERATOR_MODEL_NAME:
            self.llm = ChatGoogleGenerativeAI(model=OPERATOR_MODEL_NAME,temperature=temperature).bind_tools([
            save_emergency_case_info,
            get_instructions_for_incident,
        ])
        else:
            self.llm = ChatOpenAI(model=OPERATOR_MODEL_NAME,temperature=temperature,base_url= BASE_URL).bind_tools([
            save_emergency_case_info,
            get_instructions_for_incident,
        ])
            

        self.opening_message = "This is 911 emergency center. How may I assist you?"
        self.conversation_history = [
            SystemMessage(content=OPERATOR_SYSTEM_PROMPT),
            HumanMessage(content="Hello?"),
            AIMessage(content=self.opening_message)
        ]
        self.is_finished = False

    def get_message(self, user_message: str) -> str:

        # Add message to conversation history
        self.conversation_history.append(HumanMessage(content=user_message))
        
        # Get LLM response
        response = self.llm.invoke(self.conversation_history)
        
        # Process tool calls
        self.conversation_history.append(response)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_output = self.functions[tool_call['name']](**tool_call['args'])
                self.conversation_history.append(
                    ToolMessage(content=tool_output, tool_call_id=tool_call['id'])
                )
                response = self.llm.invoke(self.conversation_history)
                self.conversation_history.append(response)

        if self.is_finished:
            return 'Stay Safe, Thank You'
        
        return response.content

    def save_emergency_case_info(self, gathered_info: str) -> str:
        """
        Save emergency case information including extracted features.
        """
        self.is_finished = True
        return "Emergency case information saved successfully."
    

    def get_instructions_for_incident(self, incident_type: str) -> str:
        incident_type = incident_type.upper()
        incident_data = RESPONSE_PLANS.get(incident_type, False)
        if incident_data:
            required_list = incident_data
        else:
            required_list = "No specific instructions found for this incident type. Gather all relevant information."

        return required_list
    


class Observer:
    def __init__(self):

        if 'gemini' in OBSERVER_MODEL_NAME:
            self.incident_generators = {k:create_gemini_generator(OBSERVER_MODEL_NAME,v) for k,v in INCIDENT_TYPES_MAP.items()}
        else:
            self.incident_generators = {k:create_openai_generator(OBSERVER_MODEL_NAME,v, base_url=BASE_URL) for k,v in INCIDENT_TYPES_MAP.items()} 
        self.current_incident : Incident | None  = None
        self.is_dispatch_ready = False

    def _perpare_transcript(self, messages:List[BaseMessage|tuple | str]):
        """
        Prepares a coherent transcript string from a list of messages.

        This method takes a list of messages (representing the 911 call transcript)
        and formats it into a single string, identifying the speaker (Caller or 911 Operator)
        for each line.

        Args:
            messages (List[BaseMessage| Dict | str]): A list of messages representing the call transcript.
                Each message can be either a Langchain BaseMessage object (with 'type' and 'content' attributes)
                or a simple string representing a line of dialogue.

        Returns:
            str: A coherent transcript string with speaker identification.

        Raises:
            TypeError: If the input is not a list.

        Example:
            >>> messages = [
            ...     {"type": "human", "content": "Hello, 911, what's your emergency?"},
            ...     {"type": "ai", "content": "My house is on fire!"},
            ...     "Please send help!"
            ... ]
            >>> observer._prepare_transcript(messages)
            'Caller: Hello, 911, what\'s your emergency?\n911 Operator: My house is on fire!\nCaller: Please send help!'
        """

    
        transcript_lines = []
        for i, message in enumerate(messages):
            if hasattr(message, 'type') and hasattr(message, 'content'):
                if message.type in ['human', 'ai']:
                    speaker = 'Caller' if message.type == 'human' else '911 Operator'
                    transcript_lines.append(f"{speaker}: {message.content}")
            elif isinstance(message, tuple):
                speaker, message = message
                transcript_lines.append(f"{speaker}: {message}")
            elif isinstance(message, str):
                speaker = '911 Operator' if i % 2 == 0 else 'Caller'
                transcript_lines.append(f"{speaker}: {message}")
        return '\n'.join(transcript_lines)
    

    def extract_features(self, messages: List[BaseMessage|tuple | str]) -> tuple[Any, bool]:
        """
        Extracts structured information from a 911 call transcript.

        This method orchestrates the entire information extraction pipeline, including:
        - Preparing the transcript.
        - Calling the Gemini LLM generator.
        - Updating the current incident object.
        - Checking for dispatch readiness.

        Args:
            messages (List[BaseMessage | str]): A list of messages representing the call transcript.

        Returns:
            tuple[Any, bool]: A tuple containing:
                - The updated incident object (containing the extracted information).
                - A boolean indicating whether the incident is ready for dispatch.

        Raises:
            Exception: If any error occurs during the extraction process.

        Example:
            >>> messages = [
            ...     {"type": "human", "content": "There's a fire at 123 Main Street!"},
            ...     {"type": "ai", "content": "What is your name?"},
            ...     {"type": "human", "content": "John Doe"}
            ... ]
            >>> incident, is_ready = observer.extract_features(messages)
            >>> print(incident.location)
            '123 Main Street'
            >>> print(is_ready)
            True
        """
        try:
            transcript = self._perpare_transcript(messages)

            # Extract two times if it's first time to pick the right pydantic model which is not needed 
            # unlees we provided the whole transcpit once. However, it could be more efficient.
            if self.current_incident is None: 
                self.current_incident =Incident(incident_type='GENERAL', caller_name=None, caller_badge=None)
                self.extract_features(messages) 


            
            current_incident_type = self.current_incident.incident_type.value

            generator = self.incident_generators.get(
                current_incident_type,
                self.incident_generators["GENERAL"]  # Fallback to general incident
            )
            
            # Generate structured information using pre-compiled generator
            extracted_info = generator(
                OBSERVER_SYSTEM_PROMPT.format(transcript),
                max_tokens=1000,
                temperature=0.0,
            )
            
            
            print(extracted_info)
            # Check if the incident type changed
            if current_incident_type!=extracted_info.incident_type:
                incident_class =INCIDENT_TYPES_MAP[extracted_info.incident_type]
                args = {key:None for key in incident_class.model_json_schema()['properties'].keys()} # since the feild required make all args None, and we will fill it later
                self.current_incident=incident_class(**args)
            
            # Update current incident with new information since The LLM somtimes writes null instead of None so we handle it here
            for key, value in extracted_info:
                if value is not None and hasattr(self.current_incident, key):
                    if value =='null': 
                        value=None
                    setattr(self.current_incident, key, value)

            # Check if the incident requires dispatch and dispatch readiness
            if extracted_info and hasattr(extracted_info, 'is_dispatch_ready'):
                self.is_dispatch_ready = extracted_info.is_dispatch_ready()

            return self.current_incident, self.is_dispatch_ready
        
        except Exception as e:
            print(f"Error extracting features: {e}")
            return Incident(incident_type='GENERAL', caller_name=None, caller_badge=None), self.is_dispatch_ready