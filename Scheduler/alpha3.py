import sys
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Tool, FunctionDeclaration

import json
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timedelta # Keep this
from dateutil import parser as date_parser # Keep this
from dateutil.parser import ParserError # Keep this
import re
import traceback # Import traceback for better error logging

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("FATAL ERROR: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-pro-exp-03-25"
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
MEMORY_FILE = "alpha_memory_daily.json" # Holds both types now
HISTORY_FILE = "alpha_chat_history.log"


# --- Helper Functions ---

def _load_memory_data(filename=MEMORY_FILE):
    """Loads the entire memory structure (PersonalInfo and Schedule)."""
    default_structure = {"PersonalInfo": {}, "Schedule": {}}
    if not os.path.exists(filename):
        print(f"DEBUG: Memory file {filename} not found. Initializing with default structure.")
        return default_structure
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                print(f"DEBUG: Memory file {filename} is empty. Initializing with default structure.")
                return default_structure
            data = json.loads(content)

            # --- Structure Validation and Migration ---
            if isinstance(data, dict):
                # Check for the new expected structure
                if "PersonalInfo" in data and isinstance(data["PersonalInfo"], dict) and \
                   "Schedule" in data and isinstance(data["Schedule"], dict):
                     # Validate Schedule keys format (optional but good)
                     if all(re.match(r'^\d{4}-\d{2}-\d{2}$', k) for k in data["Schedule"].keys()):
                         # print(f"DEBUG: Successfully loaded memory data from {filename}.") # Less verbose
                         return data
                     else:
                         print(f"Warning: Some keys in Schedule section of {filename} do not match YYYY-MM-DD format. File may need correction.")
                         return data # Return as is for now
                else:
                    # Check if it looks like the OLD structure (only date keys at top level)
                    if all(re.match(r'^\d{4}-\d{2}-\d{2}$', k) for k in data.keys()) and \
                       all(isinstance(v, dict) for v in data.values()):
                        print(f"Warning: Old memory format detected in {filename}. Migrating to new structure (PersonalInfo/Schedule).")
                        migrated_data = {"PersonalInfo": {}, "Schedule": data}
                        # Attempt to save the migrated structure immediately
                        if _save_memory_data(migrated_data, filename):
                            print(f"DEBUG: Successfully migrated and saved data to new format in {filename}.")
                            return migrated_data
                        else:
                            print(f"ERROR: Failed to save migrated data. Returning migrated structure in memory only.")
                            return migrated_data # Return the migrated structure even if save failed
                    else:
                         # Neither new nor old structure looks right
                         print(f"Warning: Data in {filename} does not match expected structure (PersonalInfo/Schedule) or old format. Initializing with default.")
                         return default_structure
            else:
                 print(f"Warning: Data in {filename} is not a dictionary. Initializing with default structure.")
                 return default_structure

    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filename}. Initializing with default structure.")
        return default_structure
    except Exception as e:
        print(f"An error occurred loading data from {filename}: {e}. Initializing with default structure.")
        return default_structure

def _save_memory_data(data, filename=MEMORY_FILE):
    """Saves the entire memory structure (PersonalInfo and Schedule)."""
    if not isinstance(data, dict) or "PersonalInfo" not in data or "Schedule" not in data:
         print(f"ERROR: Attempted to save invalid data structure to {filename}. Save aborted.")
         return False
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, sort_keys=True) # Sort keys for readability
        # print(f"DEBUG: Successfully saved memory data to {filename}.") # Can be verbose
        return True
    except Exception as e:
        print(f"ERROR: Failed to write memory data to {filename}: {e}")
        return False

def _parse_and_standardize_date(date_input: str) -> tuple[str | None, str | None]:
    """
    Parses various date formats (incl. 'today', 'yesterday', 'last friday')
    Returns a tuple: (standardized 'YYYY-MM-DD' string or None, error message or None).
    """
    today = datetime.now().date()
    date_input_lower = date_input.lower().strip()
    error_msg = None
    standardized_date = None

    if date_input_lower == "today":
        dt = today
    elif date_input_lower == "yesterday":
        dt = today - timedelta(days=1)
    elif date_input_lower == "tomorrow":
         dt = today + timedelta(days=1)
    else:
        try:
            # Use fuzzy=False for stricter parsing if needed, default is False anyway
            parsed_dt = date_parser.parse(date_input, default=datetime.now())
            dt = parsed_dt.date()
        except (ParserError, ValueError, OverflowError, TypeError) as e:
            error_detail = f"{type(e).__name__}: {e}"
            # print(f"DEBUG: Date parsing failed for '{date_input}'. Details: {error_detail}") # Less verbose
            error_msg = f"Could not understand the date '{date_input}'. Reason: {error_detail}. Please try a different format (e.g., YYYY-MM-DD, 'today', 'Aug 5', 'last Friday')."
            dt = None

    if dt:
        standardized_date = dt.strftime('%Y-%m-%d')

    return standardized_date, error_msg


# --- Alpha Class (Corrected Structure) ---
class Alpha:
    # Class attributes for tool registration    
    tool_declarations = []
    registered_functions = {}

    # SINGLE CORRECT __init__ method
    def __init__(self):
        print("DEBUG: Initializing Alpha instance...")
        # Define system prompt directly within __init__
        self.system_prompt = (
            "You are Alpha, my personal assistant. Be helpful, concise, and friendly. "
            "You have tools to manage two types of memory:\n"
            "1.  **Personal Info:** Static details about the user (e.g., name, occupation, preferences). Use `add_personal_info`, `get_personal_info`, `delete_personal_info` for these.\n"
            "2.  **Daily Schedule/Memories:** Activities, notes, or events tied to specific dates. Use `add_daily_memory`, `get_daily_memory`, `delete_daily_memory` for these.\n\n"
            "**IMPORTANT GUIDELINES:**\n"
            "-   For **daily memories** (functions like `add_daily_memory`, `get_daily_memory`), pass the **full relevant date phrase** from the user's request as the `date_input` argument (e.g., 'today', 'yesterday', 'August 2nd 2023', 'last friday'). The system will parse it.\n"
            "-   **CRITICAL DATE HANDLING:** If the user asks about a specific date using only the month and day (e.g., 'march 28', 'what happened on august 5th') and NOT a relative term ('today', 'yesterday', 'last friday'), YOU MUST FIRST ASK the user 'Which year?'. \n"
            "    *   **AFTER THE USER RESPONDS with the year** (e.g., 'this year', '2025', 'last year'), construct the `date_input` string by combining the original date phrase and the user's year clarification (e.g., if the original query was 'march 28' and the user replies 'this year', the `date_input` for the function call should be the single string 'march 28 this year'. If they reply '2025', the `date_input` should be 'march 28 2025').\n"
            "    *   Only after constructing this combined date string should you call the relevant memory function (`get_daily_memory`, `delete_daily_memory`).\n"
            "-   Use `add_daily_memory` to save details for a specific date.\n"
            "-   Use `get_daily_memory` to retrieve details for a specific date (optionally by label). *Remember to ask for the year and construct the date string correctly if needed.*\n"
            "-   Use `delete_daily_memory` to remove details or entire dates from the schedule. *Remember to ask for the year and construct the date string correctly if needed.*\n"
            "-   Use `list_memory_dates` to see which dates have schedule entries.\n"
            "-   Use `search_memories_by_content` to find keywords across all **daily schedule** entries.\n"
            "-   Use `add_personal_info` to save or update static personal details (e.g., 'My name is...', 'My favorite color is...').\n"
            "-   Use `get_personal_info` to retrieve stored personal details (all or by label).\n"
            "-   Use `delete_personal_info` to remove a specific piece of personal info.\n"
            "-   Use `get_current_date_time` for the current time.\n"
            "-   Be clear about whether you are accessing personal info or daily schedule info in your responses."
        )

        # Initialize the model and chat session
        try:
            print(f"DEBUG: Configuring GenerativeModel: {MODEL_NAME}")
            self.model = genai.GenerativeModel(
                MODEL_NAME,
                safety_settings=SAFETY_SETTINGS,
                system_instruction=self.system_prompt,
            )
            print("DEBUG: GenerativeModel created. Starting chat session...")
            self.chat = self.model.start_chat(history=[]) # <--- Creates self.chat HERE
            print(f"DEBUG: Chat session successfully started with model {MODEL_NAME}.")
        except Exception as e:
            print(f"FATAL ERROR: Could not initialize GenerativeModel or ChatSession: {e}")
            traceback.print_exc() # Print full traceback for initialization errors
            sys.exit(1)

    # --- Methods of the Alpha class ---

    def _log_interaction(self, user_input, alpha_response):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} | You: {user_input}\n")
                f.write(f"{timestamp} | Alpha: {alpha_response}\n")
                f.write("-" * 20 + "\n")
        except Exception as e:
            print(f"\nWarning: Could not write to history file {HISTORY_FILE}: {e}")

    @classmethod
    def _create_function_declaration(cls, func):
        """Helper to create Gemini FunctionDeclaration from docstring."""
        if not func.__doc__:
             print(f"Warning: Function '{func.__name__}' has no docstring. Cannot register as tool.")
             return None
        doc_lines = [line.strip() for line in func.__doc__.strip().split('\n') if line.strip()]
        if not doc_lines:
            print(f"Warning: Function '{func.__name__}' has an empty docstring. Cannot register as tool.")
            return None

        description = doc_lines[0]
        parameters_schema = {'type': 'object', 'properties': {}, 'required': []}

        if len(doc_lines) > 1:
            for line in doc_lines[1:]:
                # Improved parsing: handles "param: type: description" more reliably
                match = re.match(r'^([\w_]+)\s*:\s*(\w+)\s*(?:\[(optional)\])?\s*:\s*(.*)$', line, re.IGNORECASE)
                if match:
                    param_name, param_type_desc, optional_flag, param_desc = match.groups()
                    param_name = param_name.strip()
                    param_type_desc = param_type_desc.strip().lower()
                    param_desc = param_desc.strip()

                    param_type = 'string' # Default
                    if 'int' in param_type_desc or 'integer' in param_type_desc: param_type = 'integer'
                    elif 'float' in param_type_desc or 'number' in param_type_desc: param_type = 'number'
                    elif 'bool' in param_type_desc or 'boolean' in param_type_desc: param_type = 'boolean'
                    elif 'list' in param_type_desc or 'array' in param_type_desc: param_type = 'array'
                    # Add other types if needed (e.g., object)

                    parameters_schema['properties'][param_name] = {'type': param_type, 'description': param_desc}
                    if not optional_flag: # Only add to required if not marked optional
                         parameters_schema['required'].append(param_name)
                else:
                    print(f"DEBUG: Could not parse parameter line in docstring for {func.__name__}: '{line}'")


        if not parameters_schema['properties']:
             # print(f"DEBUG: Function {func.__name__} declared with no parameters.")
             return FunctionDeclaration(name=func.__name__, description=description)
        else:
            # Remove 'required' list if it's empty
            if not parameters_schema['required']:
                del parameters_schema['required']
            # print(f"DEBUG: Function {func.__name__} declared with parameters: {parameters_schema}")
            return FunctionDeclaration(name=func.__name__, description=description, parameters=parameters_schema)

    @classmethod
    def add_func(cls, func):
        """Decorator to register a function as a tool."""
        declaration = cls._create_function_declaration(func)
        if declaration:
            cls.registered_functions[func.__name__] = func
            cls.tool_declarations.append(declaration)
            # print(f"DEBUG: Registered function '{func.__name__}' as a tool.") # Less verbose
        else:
             print(f"DEBUG: Failed to register function '{func.__name__}' (check docstring/parsing).")
        return func # Return the original function so it's still callable normally if needed

    def _prepare_tools(self):
        """Prepares the list of tools for the Gemini API call."""
        if self.tool_declarations:
             valid_declarations = [decl for decl in self.tool_declarations if decl is not None]
             if valid_declarations:
                return Tool(function_declarations=valid_declarations)
        return None # Return None if no valid tools are registered

    def _execute_function_call(self, function_call):
        """Executes a function call requested by the model."""
        function_name = function_call.name
        args = function_call.args # This is usually a Struct or similar type

        if function_name in self.registered_functions:
            function_to_call = self.registered_functions[function_name]
            try:
                 # Convert proto Struct/dict-like args to a standard Python dict
                 args_dict = dict(args)
                 print(f"DEBUG: Calling function: {function_name} with args: {args_dict}")
            except Exception as e:
                 print(f"Warning: Could not convert args for {function_name} directly to dict: {e}. Trying item iteration.")
                 try:
                     args_dict = {key: value for key, value in args.items()}
                     print(f"DEBUG: Calling function (iter): {function_name} with args: {args_dict}")
                 except Exception as e2:
                     print(f"ERROR: Failed to extract arguments for {function_name}: {e2}. Calling with no args.")
                     args_dict = {} # Fallback to no arguments if conversion fails

            try:
                # Call the actual Python function with unpacked arguments
                function_response = function_to_call(**args_dict)
                # print(f"DEBUG: Func {function_name} returned: {type(function_response)} -> {str(function_response)[:100]}...") # Less verbose
                return function_name, function_response
            except TypeError as te:
                 # Specific error for incorrect arguments passed to the function
                 print(f"ERROR: TypeError executing '{function_name}' with {args_dict}: {te}")
                 traceback.print_exc() # Print stack trace for this specific error
                 error_message = f"Error: Called '{function_name}' with incompatible arguments. Details: {te}"
                 return function_name, error_message
            except Exception as e:
                # Catch other exceptions during function execution
                print(f"ERROR: Exception executing '{function_name}' with {args_dict}: {e}")
                traceback.print_exc() # Print stack trace for debugging
                error_message = f"Error executing function '{function_name}': {type(e).__name__}: {e}"
                return function_name, error_message
        else:
            print(f"Warning: Model tried to call unknown function: {function_name}")
            error_message = f"Error: Function '{function_name}' is not registered or available."
            return function_name, error_message # Return the name and error message

    def _prepare_function_response_dict(self, function_name, function_response_content):
         """Formats the function execution result into the structure Gemini expects."""
         # Ensure the content is serializable (usually needs to be a string or simple types)
         if not isinstance(function_response_content, (str, int, float, bool, list, dict, type(None))):
             try:
                 # Attempt standard JSON serialization for complex objects
                 serialized_content = json.dumps(function_response_content)
             except TypeError:
                 # Fallback to string representation if JSON fails
                 print(f"Warning: Could not JSON serialize response from {function_name}. Using str().")
                 serialized_content = str(function_response_content)
         else:
              # If it's already a basic type, use it directly (or serialize dict/list for safety)
              if isinstance(function_response_content, (dict, list)):
                   serialized_content = json.dumps(function_response_content)
              else:
                   serialized_content = function_response_content # Keep str, int, float, bool, None as is

         # Structure required by the Gemini API
         response_dict = {
             "function_response": {
                 "name": function_name,
                 "response": {
                     "content": serialized_content, # The result must be under 'content'
                 }
             }
         }
         # print(f"DEBUG: Prepared function response dict: {response_dict}") # Can be verbose
         return response_dict


    def chat_with_gemini(self, user_input):
        """Handles the chat interaction, including function calls."""
        prompt = f"{user_input}"
        current_tools = self._prepare_tools()
        tools_list = [current_tools] if current_tools else None # API expects a list of tools, or None/empty list

        # --- Robustness Check ---
        if not hasattr(self, 'chat') or self.chat is None:
             print("FATAL INTERNAL ERROR: self.chat attribute is missing or None before send_message. Initialization likely failed.")
             error_message = "Sorry, there's an internal setup error preventing me from chatting. Please check the logs or restart."
             self._log_interaction(user_input, f"[ERROR: {error_message}]")
             return error_message

        try:
            print(f"DEBUG: Sending prompt to Gemini: '{prompt[:100]}...' with tools: {'Yes' if tools_list else 'No'}")
            # Initial message to the model
            response = self.chat.send_message(prompt, tools=tools_list)

            # Loop to handle potential function calls
            while True:
                function_call = None
                # Check if the response contains a function call request
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            function_call = part.function_call
                            print(f"DEBUG: Model requested function call: {function_call.name}")
                            break # Found a function call, exit the part loop

                if function_call:
                    # Execute the function
                    function_name, function_result = self._execute_function_call(function_call)
                    # Prepare the response to send back to the model
                    function_response_dict = self._prepare_function_response_dict(function_name, function_result)

                    # Send the function execution result back to the model
                    print(f"DEBUG: Sending function response back to model for {function_name}")
                    response = self.chat.send_message(function_response_dict) # Send the PART back
                else:
                    # No function call requested (or handled), break the loop
                    break

            # --- Extract Final Text Response ---
            final_text = ""
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 # Concatenate text parts, ignoring non-text parts (like function calls/responses)
                 final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))

            # Fallback if text is directly in response (less common with functions)
            if not final_text and hasattr(response, 'text'):
                final_text = response.text

            # Handle cases where response might be blocked or empty
            if not final_text:
                 block_reason = None
                 safety_ratings = None
                 if response.prompt_feedback and response.prompt_feedback.block_reason:
                      block_reason = response.prompt_feedback.block_reason
                      print(f"Warning: Response blocked. Reason: {block_reason}")
                      # You might want to check safety ratings too if available
                      if response.candidates and response.candidates[0].safety_ratings:
                           safety_ratings = response.candidates[0].safety_ratings
                           print(f"Safety Ratings: {safety_ratings}")
                      return f"[Response blocked due to: {block_reason}. Please rephrase or adjust safety settings.]"
                 else:
                      # No block reason, but still no text
                      print(f"Warning: Received empty final text response. Full response object parts: {response.candidates[0].content.parts if response.candidates else 'No candidates'}")
                      return "[Alpha had no text response, and no block reason found. There might have been an issue or the model chose not to reply.]"

            # Log and return the final text
            self._log_interaction(user_input, final_text)
            return final_text

        except Exception as e:
            print(f"\nERROR: Unexpected chat error in chat_with_gemini: {type(e).__name__} - {e}")
            traceback.print_exc() # Print stack trace for debugging unexpected errors
            error_message = f"Sorry, an unexpected error occurred ({type(e).__name__}) while processing your request. Please check the logs."
            # Log the error interaction
            self._log_interaction(user_input, f"[ERROR: {error_message}]")
            return error_message


    def speak(self, output):
        """Prints the output with a typing effect."""
        print("\nAlpha: ", end='')
        output_text = output if output else "[No response received]" # Handle None or empty output
        try:
            for char in output_text:
                print(char, end='', flush=True)
                # Adjust sleep time for desired typing speed
                time.sleep(0.02) # Slightly faster than 0.025
        except Exception as e:
             # Avoid crashing the main loop if printing fails
             print(f"[Speak animation error: {e}]", end='')
        print("\n") # Ensure a newline after speaking


# --- Tool Functions ---

# === Personal Info Functions ===

@Alpha.add_func
def add_personal_info(label: str, value: str):
    """
    Save or update a piece of static personal information (like name, preference, occupation).
    label: string: The name or key for this piece of personal info (e.g., 'Name', 'Favorite Color').
    value: string: The content or value associated with the label.
    """
    if not label or not label.strip(): # Check if label is None or empty/whitespace
        return "Error: A non-empty label is required for personal info."
    if value is None: # Allow empty string "" but not None
         return "Error: A value is required for personal info (can be an empty string)."

    data = _load_memory_data()
    # Ensure PersonalInfo exists and is a dict
    if "PersonalInfo" not in data or not isinstance(data.get("PersonalInfo"), dict):
        data["PersonalInfo"] = {}

    label_clean = label.strip()
    data["PersonalInfo"][label_clean] = value # Store original value (might be empty string)
    print(f"DEBUG: Added/Updated Personal Info: '{label_clean}': '{value}'")

    if _save_memory_data(data):
        return f"Okay, I've updated the personal info for '{label_clean}'."
    else:
        # In-memory change happened, but save failed. State is inconsistent.
        print(f"ERROR: Failed to save update for personal info label '{label_clean}'. Data in memory was changed but not persisted.")
        return f"Error: I tried to update personal info for '{label_clean}', but failed to save it."

@Alpha.add_func
def get_personal_info(label: str = None):
    """
    Retrieve stored personal information. Can get all details or just one specific label.
    label: string [optional]: The specific label to retrieve. If omitted, returns all stored personal info.
    """
    data = _load_memory_data()
    personal_info = data.get("PersonalInfo", {})

    if not isinstance(personal_info, dict) or not personal_info:
        return "I don't have any personal information stored yet."

    if label:
        label_strip = label.strip()
        if label_strip in personal_info:
            # Use f-string for clarity
            return f"Personal Info for '{label_strip}': {personal_info[label_strip]}"
        else:
            return f"Sorry, I couldn't find any personal info stored under the label '{label_strip}'."
    else:
        # Return all info if no specific label is requested
        details_list = [f"- {lbl}: {val}" for lbl, val in personal_info.items()]
        if not details_list:
             return "I don't have any personal information stored yet." # Should be caught earlier, but safe
        return "Here's the personal information I have stored:\n" + "\n".join(details_list)

@Alpha.add_func
def delete_personal_info(label: str):
    """
    Delete a specific piece of personal information by its label.
    label: string: The specific label to delete from personal info.
    """
    if not label or not label.strip():
        return "Error: You need to provide the label of the personal info you want to delete."

    data = _load_memory_data()
    label_strip = label.strip()

    # Check if PersonalInfo exists and if the label is present
    if "PersonalInfo" not in data or not isinstance(data.get("PersonalInfo"), dict) or label_strip not in data["PersonalInfo"]:
        return f"I couldn't find any personal info with the label '{label_strip}', so nothing was deleted."

    # Delete the item
    del data["PersonalInfo"][label_strip]
    print(f"DEBUG: Deleted Personal Info label '{label_strip}'.")

    if _save_memory_data(data):
        return f"Okay, I've deleted the personal info labeled '{label_strip}'."
    else:
        # Deletion happened in memory but save failed. State is inconsistent.
        print(f"ERROR: Failed to save deletion of personal info label '{label_strip}'. Data in memory was changed but not persisted.")
        # Optionally try to restore the deleted item in memory here if critical, or just report error.
        return f"Error: I tried to delete personal info for '{label_strip}', but failed to save the change."


# === Daily Schedule/Memory Functions (Uses Schedule sub-key) ===

@Alpha.add_func
def add_daily_memory(date_input: str, label: str, value: str):
    """
    Save or update a specific labeled detail for a given date's schedule/memory. Understands dates like 'today', 'yesterday', 'Aug 5', '2024-08-05', 'last friday'.
    date_input: string: The date for the memory (e.g., 'today', '2024-08-05', 'next monday'). The model should pass the user's original phrase.
    label: string: The name or key for this piece of information (e.g., 'Activity', 'Meeting Topic', 'Lunch').
    value: string: The content or value associated with the label for that date.
    """
    if not date_input or not date_input.strip():
        return "Error: A date input (like 'today' or '2024-09-15') is required."
    if not label or not label.strip():
        return "Error: A non-empty label is required for daily memories."
    if value is None: # Allow empty string "" but not None
         return "Error: A value is required for daily memories (can be an empty string)."

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg:
        # Return the specific parsing error message
        return f"Error processing date for daily memory: {error_msg}"

    # Proceed if date parsing was successful
    data = _load_memory_data()

    # Ensure Schedule key exists and is a dict
    if "Schedule" not in data or not isinstance(data.get("Schedule"), dict):
        data["Schedule"] = {}

    schedule_data = data["Schedule"]
    # Use setdefault to get the date's dict or create it if it doesn't exist
    date_entry = schedule_data.setdefault(date_str, {})

    label_clean = label.strip()
    date_entry[label_clean] = value # Add/update the label-value pair for the specific date
    print(f"DEBUG: Added/Updated daily memory for {date_str}: '{label_clean}': '{value}'")

    if _save_memory_data(data):
        # Provide confirmation including the parsed date
        return f"Okay, I've added/updated the memory for {date_str} (parsed from '{date_input}') with label '{label_clean}'."
    else:
        print(f"ERROR: Failed to save daily memory update for date {date_str} (parsed from '{date_input}'). Data changed in memory but not persisted.")
        return f"Error: I tried to update the memory for {date_str}, but failed to save it."

@Alpha.add_func
def get_daily_memory(date_input: str, label: str = None):
    """
    Retrieve schedule/memory details for a specific date. Understands dates like 'today', 'yesterday', 'Aug 5', '2024-08-05', 'last friday'. Can get all details or just one label for the date.
    date_input: string: The date to retrieve memories for (e.g., 'today', '2024-08-05'). Model should pass user phrase, potentially after year clarification.
    label: string [optional]: The specific label to retrieve for that date. If omitted, returns all details for the date.
    """
    if not date_input or not date_input.strip():
        return "Error: A date input (like 'today' or '2024-09-15') is required to get memories."

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg:
        # Return the specific parsing error message
        return f"Error processing date for getting daily memory: {error_msg}"

    # Proceed if date parsing was successful
    data = _load_memory_data()
    schedule_data = data.get("Schedule", {}) # Get schedule safely

    if not isinstance(schedule_data, dict) or date_str not in schedule_data:
        # Handle case where Schedule key doesn't exist or date is not found
        return f"I don't have any daily memories stored for the date {date_str} (parsed from '{date_input}')."

    date_entry = schedule_data[date_str] # We know date_str exists now

    if label:
        label_strip = label.strip()
        if label_strip in date_entry:
            # Return the specific detail
            return f"On {date_str}, the detail for '{label_strip}' is: {date_entry[label_strip]}"
        else:
            # Label not found for that date
            return f"I found memories for {date_str}, but couldn't find the specific label '{label_strip}'."
    else:
        # Return all details for the date if no specific label is requested
        if not date_entry: # Check if the date entry itself is empty
             return f"I have an entry for {date_str}, but there are no specific details stored for that day."
        details_list = [f"- {lbl}: {val}" for lbl, val in date_entry.items()]
        # Format the output clearly
        return f"Here are the daily memories I have for {date_str} (parsed from '{date_input}'):\n" + "\n".join(details_list)


@Alpha.add_func
def list_memory_dates():
    """
    List all dates (YYYY-MM-DD) that currently have daily schedule/memory entries stored.
    """
    data = _load_memory_data()
    schedule_data = data.get("Schedule", {})

    # Ensure schedule_data is a dictionary before getting keys
    if not isinstance(schedule_data, dict):
        print("Warning: Schedule data in memory file is not a dictionary.")
        return "Could not retrieve memory dates due to an internal format issue."

    dates = sorted(list(schedule_data.keys())) # Get keys safely

    if not dates:
        return "I don't have any daily memories stored for any dates yet."
    else:
        # Improve readability for many dates (optional)
        if len(dates) > 15:
            return f"I have daily memory entries for {len(dates)} dates, including:\n" + "\n".join(dates[:15]) + "\n...and more."
        else:
            return "I have daily memory entries for the following dates:\n" + "\n".join(dates)


@Alpha.add_func
def delete_daily_memory(date_input: str, label: str = None):
    """
    Delete schedule/memory details for a specific date. Understands dates like 'today', 'Aug 5', 'last friday'. Can delete a specific label or the entire date's entry.
    date_input: string: The date to delete memories from (e.g., 'today', '2024-08-05'). Model passes user phrase.
    label: string [optional]: The specific label to delete for that date. If omitted, deletes all entries for the entire date.
    """
    if not date_input or not date_input.strip():
        return "Error: A date input (like 'today' or '2024-09-15') is required to delete memories."

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg:
        # Return the specific parsing error message
        return f"Error processing date for deleting daily memory: {error_msg}"

    # Proceed if date parsing was successful
    data = _load_memory_data()

    # Check if Schedule exists and is a dict
    if "Schedule" not in data or not isinstance(data.get("Schedule"), dict):
         return f"I don't have any daily schedule/memory data to delete from."

    schedule_data = data["Schedule"]

    # Check if the date exists in the schedule
    if date_str not in schedule_data:
        return f"I couldn't find any daily memories for the date {date_str} (parsed from '{date_input}'), so nothing was deleted."

    if label:
        # Delete a specific label
        label_strip = label.strip()
        if label_strip in schedule_data[date_str]:
            del schedule_data[date_str][label_strip]
            print(f"DEBUG: Deleted daily memory label '{label_strip}' for date {date_str}.")

            # If the date entry becomes empty after deleting the label, remove the date key itself for cleanliness
            if not schedule_data[date_str]:
                del schedule_data[date_str]
                print(f"DEBUG: Daily memory entry for {date_str} became empty and was removed.")

            if _save_memory_data(data):
                 return f"Okay, I've deleted the memory label '{label_strip}' for date {date_str}."
            else:
                 print(f"ERROR: Failed to save deletion of label '{label_strip}' for {date_str}. Data changed in memory but not persisted.")
                 return f"Error: I tried to delete the label '{label_strip}' for {date_str}, but failed to save the change."
        else:
            # Label not found for that date
            return f"I found memories for {date_str}, but couldn't find the label '{label_strip}' to delete."
    else:
        # Delete the entire entry for the date if no label is specified
        del schedule_data[date_str]
        print(f"DEBUG: Deleted all daily memory entries for date {date_str}.")
        if _save_memory_data(data):
            return f"Okay, I've deleted all daily memory entries for date {date_str}."
        else:
            print(f"ERROR: Failed to save deletion of date {date_str}. Data changed in memory but not persisted.")
            return f"Error: I tried to delete all memories for {date_str}, but failed to save the change."


@Alpha.add_func
def search_memories_by_content(search_term: str):
    """
    Searches through all DAILY schedule/memory entries (dates, labels, and values) for a specific keyword or phrase (case-insensitive). Does NOT search Personal Info.
    search_term: string: The text to search for within the daily memories.
    """
    if not search_term or not search_term.strip():
        return "Error: Please provide a search term to look for in daily memories."

    data = _load_memory_data()
    schedule_data = data.get("Schedule", {})

    if not isinstance(schedule_data, dict) or not schedule_data:
        return "I don't have any daily schedule/memory entries stored to search through."

    search_term_lower = search_term.lower().strip()
    matches = []
    # Iterate through the Schedule part only
    for date_str in sorted(schedule_data.keys()): # Sort for consistent results
        details = schedule_data[date_str]
        if isinstance(details, dict): # Ensure the entry for the date is a dict
            for label, value in details.items():
                # Check date, label, and value (converted to string) for the search term
                # Value conversion to string handles potential non-string data gracefully
                value_str = str(value)
                if search_term_lower in date_str or \
                   search_term_lower in label.lower() or \
                   search_term_lower in value_str.lower():
                    matches.append({"date": date_str, "label": label, "value": value_str})
        else:
             print(f"Warning: Entry for date {date_str} in schedule is not a dictionary, skipping.")


    if not matches:
        return f"I searched through all the daily memories, but couldn't find any mention of '{search_term}'."
    else:
        results_str = f"I found mentions of '{search_term}' in daily memories on the following dates:\n"
        # Limit results shown for brevity if needed
        max_results_to_show = 20
        count = 0
        for match in matches:
             if count >= max_results_to_show:
                 results_str += f"... (and {len(matches) - count} more matches)\n"
                 break
             # Truncate long values in the output
             truncated_value = match['value'][:80] + ('...' if len(match['value']) > 80 else '')
             results_str += f"- {match['date']} ({match['label']}): {truncated_value}\n"
             count += 1
        return results_str.strip() # Remove trailing newline

@Alpha.add_func
def get_current_date_time():
    """
    Get the current local date and time, formatted for readability.
    """
    now = datetime.now()
    # Example format: "2023-10-27 15:30:05 (Friday)"
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S (%A)")
    return f"The current date and time is {formatted_datetime}"


# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Alpha Assistant (Gemini Edition - Combined Memory v1) ---")
    print("Initializing...")

    # Initialize memory file with correct structure if it doesn't exist or is empty/invalid
    initial_data = _load_memory_data()
    # Ensure the basic structure {PersonalInfo: {}, Schedule: {}} exists
    if not isinstance(initial_data, dict) or \
       "PersonalInfo" not in initial_data or not isinstance(initial_data.get("PersonalInfo"), dict) or \
       "Schedule" not in initial_data or not isinstance(initial_data.get("Schedule"), dict):
         print(f"Warning: Memory file {MEMORY_FILE} was missing, empty, or invalid. Re-initializing with default structure.")
         if not _save_memory_data({"PersonalInfo": {}, "Schedule": {}}):
              print(f"FATAL ERROR: Could not write initial structure to {MEMORY_FILE}. Exiting.")
              sys.exit(1)
         else:
              print(f"DEBUG: Successfully initialized {MEMORY_FILE}.")

    try:
        # Instantiate the corrected Alpha class
        ai = Alpha()
        print(f"Using model: {MODEL_NAME}")
        print(f"Memory file: {MEMORY_FILE}")
        print(f"History file: {HISTORY_FILE}")
        registered_tool_names = list(ai.registered_functions.keys())
        if registered_tool_names:
             print(f"Registered tools: {', '.join(registered_tool_names)}")
        else:
             print("Warning: No tools were successfully registered.")
        print("-" * 30 + "\nAlpha is ready. Type 'exit' or 'quit' to end.\n" + "-" * 30)
    except Exception as init_error:
        # Catch errors during Alpha() initialization (e.g., API key issues, model init failure)
        print(f"\nFATAL ERROR during Alpha initialization: {init_error}")
        traceback.print_exc()
        sys.exit(1)

    # --- Main Chat Loop ---
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                 print("\n\nAlpha: End of input detected. Shutting down...")
                 ai.speak("Goodbye!")
                 break # Exit loop gracefully on EOF

            if not user_input:
                continue # Skip empty input

            if user_input.lower() in ["exit", "quit", "goodbye", "bye", "0"]:
                ai.speak("Okay, goodbye!")
                break # Exit loop on quit command

            # Process the input using the chat method
            output = ai.chat_with_gemini(user_input)

            # Speak the response
            ai.speak(output)

    except KeyboardInterrupt:
        print("\n\nAlpha: Detected interrupt. Shutting down...")
        # Try to speak, but handle potential errors if ai object is broken
        try: ai.speak("Okay, exiting now.")
        except Exception: print("Exiting immediately.")
    except Exception as main_loop_error:
        # Catch unexpected errors during the main loop conversation
        print(f"\nFATAL ERROR during main chat loop: {main_loop_error}")
        traceback.print_exc() # Print stack trace for debugging
        try: ai.speak("A critical error occurred. I need to shut down.")
        except Exception: print("Critical error. Shutting down.")
    finally:
        # This block runs whether the loop finished normally or due to an exception/interrupt
        print("\n" + "-" * 30 + "\nAlpha: Session ended.")
        print(f"Chat history saved in: {HISTORY_FILE}")
        print(f"Memory data stored in: {MEMORY_FILE}")
        print("-" * 30)