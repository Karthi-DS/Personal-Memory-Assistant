import sys
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Tool, FunctionDeclaration, Part # Import Part

import json
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from dateutil.parser import ParserError
import re
import traceback
import configparser # Import configparser

# --- Configuration Loading ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

config = configparser.ConfigParser()
# Provide default values in case the file is missing sections/keys
config.read_dict({'Gemini': {'ModelName': 'gemini-1.5-flash-latest'},
                   'Files': {'MemoryFile': 'alpha_memory_default.json',
                             'HistoryFile': 'alpha_history_default.log'},
                   'Settings': {'TypingSpeed': '0.02'}})
try:
    config.read('config.ini')
except Exception as e:
    print(f"Warning: Could not read config.ini: {e}. Using defaults.")

if not GOOGLE_API_KEY:
    print("FATAL ERROR: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

MODEL_NAME = config.get('Gemini', 'ModelName', fallback='gemini-1.5-flash-latest')
MEMORY_FILE = config.get('Files', 'MemoryFile', fallback='alpha_memory_v2.json')
HISTORY_FILE = config.get('Files', 'HistoryFile', fallback='alpha_chat_history_v2.log')
TYPING_SPEED = config.getfloat('Settings', 'TypingSpeed', fallback=0.02)

# Safety Settings remain the same
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- Helper Functions (Memory Loading/Saving - Updated for New Structure) ---

def _load_memory_data(filename=MEMORY_FILE):
    """Loads the entire memory structure (PersonalInfo, Schedule, Notes)."""
    # New default structure includes "Notes"
    default_structure = {"PersonalInfo": {}, "Schedule": {}, "Notes": {}}
    if not os.path.exists(filename):
        print(f"DEBUG: Memory file {filename} not found. Initializing.")
        # Save the default structure immediately if the file doesn't exist
        _save_memory_data(default_structure, filename)
        return default_structure
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                print(f"DEBUG: Memory file {filename} is empty. Initializing.")
                return default_structure # Return default if empty
            data = json.loads(content)

            # --- Structure Validation and Migration ---
            migrated = False
            if isinstance(data, dict):
                # Ensure all top-level keys exist
                if "PersonalInfo" not in data or not isinstance(data["PersonalInfo"], dict):
                    print(f"Warning: Missing or invalid 'PersonalInfo' in {filename}. Initializing.")
                    data["PersonalInfo"] = {}
                    migrated = True
                if "Schedule" not in data or not isinstance(data["Schedule"], dict):
                    print(f"Warning: Missing or invalid 'Schedule' in {filename}. Initializing.")
                    data["Schedule"] = {}
                    migrated = True
                if "Notes" not in data or not isinstance(data["Notes"], dict):
                    print(f"DEBUG: Adding missing 'Notes' section to {filename}.")
                    data["Notes"] = {} # Add the new Notes section if missing
                    migrated = True

                # Optional: Validate Schedule keys format (existing check)
                if not all(re.match(r'^\d{4}-\d{2}-\d{2}$', k) for k in data["Schedule"].keys()):
                    print(f"Warning: Some keys in Schedule section of {filename} do not match YYYY-MM-DD format.")
                    # No migration needed for this warning, just report

                if migrated:
                    print(f"DEBUG: Attempting to save migrated/corrected structure to {filename}.")
                    if _save_memory_data(data, filename):
                         print(f"DEBUG: Successfully saved updated structure to {filename}.")
                    else:
                         print(f"ERROR: Failed to save migrated structure to {filename}. Using in-memory version.")
                # print(f"DEBUG: Successfully loaded memory data from {filename}.") # Less verbose
                return data

            else:
                 # Data is not a dictionary at all
                 print(f"Warning: Data in {filename} is not a dictionary. Initializing with default structure.")
                 # Attempt to overwrite with default structure
                 _save_memory_data(default_structure, filename)
                 return default_structure

    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filename}. Initializing with default structure.")
        _save_memory_data(default_structure, filename)
        return default_structure
    except Exception as e:
        print(f"An error occurred loading data from {filename}: {e}. Initializing with default structure.")
        _save_memory_data(default_structure, filename) # Try to save default even on other errors
        return default_structure


def _save_memory_data(data, filename=MEMORY_FILE):
    """Saves the entire memory structure (PersonalInfo, Schedule, Notes)."""
    # Validate the expected top-level structure
    if not isinstance(data, dict) or \
       "PersonalInfo" not in data or not isinstance(data["PersonalInfo"], dict) or \
       "Schedule" not in data or not isinstance(data["Schedule"], dict) or \
       "Notes" not in data or not isinstance(data["Notes"], dict):
         print(f"ERROR: Attempted to save invalid data structure to {filename}. Save aborted.")
         print(f"DEBUG: Invalid structure was: {type(data)} - Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
         return False
    try:
        # Use a temporary file for atomic write
        temp_filename = filename + ".tmp"
        with open(temp_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, sort_keys=True)
        # If write succeeds, replace the original file
        os.replace(temp_filename, filename)
        # print(f"DEBUG: Successfully saved memory data to {filename}.") # Less verbose
        return True
    except Exception as e:
        print(f"ERROR: Failed to write memory data to {filename}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e_rem:
                print(f"ERROR: Failed to remove temporary file {temp_filename}: {e_rem}")
        return False

# _parse_and_standardize_date remains the same

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
            # Setting default to now() helps parse relative terms like "next friday" correctly
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


# --- Alpha Class (Updated) ---
class Alpha:
    # Class attributes for tool registration
    tool_declarations = []
    registered_functions = {}

    def __init__(self):
        print("DEBUG: Initializing Alpha instance...")
        # Updated System Prompt
        self.system_prompt = (
            "You are Alpha, my personal assistant. Be helpful, concise, and friendly. "
            "You have tools to manage three types of memory:\n"
            "1.  **Personal Info:** Static details about the user (e.g., name, preferences). Use `add_personal_info`, `get_personal_info`, `edit_personal_info`, `delete_personal_info`.\n"
            "2.  **Daily Schedule/Memories/Tasks:** Activities, notes, or events tied to specific dates. Items can optionally have a 'status' (e.g., 'pending', 'done'). Use `add_daily_memory`, `get_daily_memory`, `edit_daily_memory`, `delete_daily_memory`, `mark_task_done`, `get_pending_tasks`, `list_memory_dates`, `search_daily_memories`.\n"
            "3.  **General Notes:** Information not tied to a specific date (e.g., ideas, project details, lists). Use `add_note`, `get_note`, `search_notes`, `list_notes`, `delete_note`.\n\n"
            "**IMPORTANT GUIDELINES:**\n"
            "-   For **daily memories/tasks** (functions like `add_daily_memory`, `get_daily_memory`, `mark_task_done`), pass the **full relevant date phrase** from the user's request as the `date_input` argument (e.g., 'today', 'yesterday', 'August 2nd 2023', 'last friday'). The system will parse it.\n"
            "-   **CRITICAL DATE HANDLING:** If the user asks about a specific date using only the month and day (e.g., 'march 28', 'what happened on august 5th') and NOT a relative term ('today', 'yesterday', 'last friday'), YOU MUST FIRST ASK the user 'Which year?'.\n"
            "    *   **AFTER THE USER RESPONDS** with the year (e.g., 'this year', '2025', 'last year'), construct the `date_input` string by combining the original date phrase and the user's year clarification (e.g., if the original query was 'march 28' and the user replies 'this year', the `date_input` for the function call should be the single string 'march 28 this year'. If they reply '2025', the `date_input` should be 'march 28 2025').\n"
            "    *   Only after constructing this combined date string should you call the relevant memory function (`get_daily_memory`, `delete_daily_memory`, `mark_task_done`).\n"
            "-   Use `add_daily_memory` to save details (optionally as a task with 'pending' status) for a specific date.\n"
            "-   Use `get_daily_memory` to retrieve details for a specific date (optionally by label). *Remember date handling.*\n"
            "-   Use `edit_daily_memory` to change the *value* of an existing item for a date. *Remember date handling.*\n"
            "-   Use `delete_daily_memory` to remove details or entire dates from the schedule. *Remember date handling.*\n"
            "-   Use `mark_task_done` to change the status of a daily item to 'done'. *Remember date handling.*\n"
            "-   Use `get_pending_tasks` to list items not marked as 'done' for a specific date or all dates.\n"
            "-   Use `list_memory_dates` to see which dates have schedule entries.\n"
            "-   Use `search_daily_memories` to find keywords across all **daily schedule** entries.\n"
            "-   Use `add_note`, `get_note`, `search_notes`, `list_notes`, `delete_note` for general, non-dated information.\n"
            "-   Use `add_personal_info` (for new info) or `edit_personal_info` (to change existing info) for static personal details.\n"
            "-   Use `get_personal_info` to retrieve stored personal details.\n"
            "-   Use `delete_personal_info` to remove a specific piece of personal info.\n"
            "-   Use `get_current_date_time` for the current time.\n"
            "-   Be clear about whether you are accessing personal info, daily schedule, or general notes in your responses.\n"
            "-   If asked to summarize information (e.g., 'summarize my week', 'summarize notes on X'), use the appropriate 'get' or 'search' functions first to retrieve the relevant data, then present the summary based on the retrieved information."
        )

        try:
            print(f"DEBUG: Configuring GenerativeModel: {MODEL_NAME}")
            self.model = genai.GenerativeModel(
                MODEL_NAME,
                safety_settings=SAFETY_SETTINGS,
                system_instruction=self.system_prompt,
            )
            print("DEBUG: GenerativeModel created. Starting chat session...")
            # Initialize with empty history. History is managed by the ChatSession object itself.
            self.chat = self.model.start_chat(history=[])
            print(f"DEBUG: Chat session successfully started with model {MODEL_NAME}.")
        except Exception as e:
            print(f"FATAL ERROR: Could not initialize GenerativeModel or ChatSession: {e}")
            traceback.print_exc()
            sys.exit(1)


    # --- Methods of the Alpha class ---

    def _log_interaction(self, user_input, alpha_response):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} | You: {user_input}\n")
                # Ensure alpha_response is a string before writing
                f.write(f"{timestamp} | Alpha: {str(alpha_response)}\n")
                f.write("-" * 20 + "\n")
        except Exception as e:
            print(f"\nWarning: Could not write to history file {HISTORY_FILE}: {e}")

    # _create_function_declaration remains the same
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
                    # Add other types if needed

                    parameters_schema['properties'][param_name] = {'type': param_type, 'description': param_desc}
                    if not optional_flag: # Only add to required if not marked optional
                         parameters_schema['required'].append(param_name)
                # else: # Reduced verbosity
                    # print(f"DEBUG: Could not parse parameter line in docstring for {func.__name__}: '{line}'")

        if not parameters_schema['properties']:
             return FunctionDeclaration(name=func.__name__, description=description)
        else:
            if not parameters_schema['required']:
                del parameters_schema['required']
            return FunctionDeclaration(name=func.__name__, description=description, parameters=parameters_schema)

    # add_func remains the same
    @classmethod
    def add_func(cls, func):
        """Decorator to register a function as a tool."""
        declaration = cls._create_function_declaration(func)
        if declaration:
            cls.registered_functions[func.__name__] = func
            cls.tool_declarations.append(declaration)
            # print(f"DEBUG: Registered function '{func.__name__}' as a tool.")
        else:
             print(f"DEBUG: Failed to register function '{func.__name__}' (check docstring/parsing).")
        return func

    # _prepare_tools remains the same
    def _prepare_tools(self):
        """Prepares the list of tools for the Gemini API call."""
        if self.tool_declarations:
             valid_declarations = [decl for decl in self.tool_declarations if decl is not None]
             if valid_declarations:
                return Tool(function_declarations=valid_declarations)
        return None

    # _execute_function_call remains the same
    def _execute_function_call(self, function_call):
        """Executes a function call requested by the model."""
        function_name = function_call.name
        args = function_call.args

        if function_name in self.registered_functions:
            function_to_call = self.registered_functions[function_name]
            try:
                 args_dict = dict(args)
                 print(f"DEBUG: Calling function: {function_name} with args: {args_dict}")
            except Exception as e:
                 print(f"Warning: Could not convert args for {function_name} directly to dict: {e}. Trying item iteration.")
                 try:
                     args_dict = {key: value for key, value in args.items()}
                     print(f"DEBUG: Calling function (iter): {function_name} with args: {args_dict}")
                 except Exception as e2:
                     print(f"ERROR: Failed to extract arguments for {function_name}: {e2}. Calling with no args.")
                     args_dict = {}

            try:
                # Call the actual Python function with unpacked arguments
                function_response = function_to_call(self, **args_dict) # Pass self to tool functions
                # Ensure response is JSON serializable for the API
                if not isinstance(function_response, (str, int, float, bool, list, dict, type(None))):
                    function_response = str(function_response)
                elif isinstance(function_response, (list, dict)):
                     # Attempt to serialize complex types, fallback to string
                     try:
                          function_response = json.dumps(function_response)
                     except TypeError:
                          print(f"Warning: Could not JSON serialize complex response from {function_name}. Using str().")
                          function_response = str(function_response)

                return function_name, function_response
            except TypeError as te:
                 if "'self'" in str(te):
                      print(f"ERROR: TypeError likely due to missing 'self' in function call for '{function_name}'. Trying again without passing self.")
                      try:
                          # Try calling without explicitly passing self (if it's not needed)
                          function_response = function_to_call(**args_dict)
                          if not isinstance(function_response, (str, int, float, bool, list, dict, type(None))): function_response = str(function_response)
                          elif isinstance(function_response, (list, dict)): function_response = json.dumps(function_response)
                          return function_name, function_response
                      except Exception as te2:
                          print(f"ERROR: Second attempt failed for '{function_name}': {te2}")
                          traceback.print_exc()
                          error_message = f"Error: Internal error calling '{function_name}'. Details: {te2}"
                          return function_name, error_message
                 else:
                      print(f"ERROR: TypeError executing '{function_name}' with {args_dict}: {te}")
                      traceback.print_exc()
                      error_message = f"Error: Called '{function_name}' with incompatible arguments. Details: {te}"
                      return function_name, error_message
            except Exception as e:
                print(f"ERROR: Exception executing '{function_name}' with {args_dict}: {e}")
                traceback.print_exc()
                error_message = f"Error executing function '{function_name}': {type(e).__name__}: {e}"
                return function_name, error_message
        else:
            print(f"Warning: Model tried to call unknown function: {function_name}")
            error_message = f"Error: Function '{function_name}' is not registered or available."
            return function_name, error_message

    # _prepare_function_response_dict remains the same (using genai.Part now)
    def _prepare_function_response_part(self, function_name, function_response_content):
         """Formats the function execution result into the Part structure Gemini expects."""
         # Ensure the content is serializable (usually needs to be a string or simple types)
         # Convert dict/list to JSON string for the API if not already a simple type
         if isinstance(function_response_content, (dict, list)):
             try:
                 serializable_content = json.dumps(function_response_content)
             except TypeError:
                 print(f"Warning: Could not JSON serialize dict/list response from {function_name}. Using str().")
                 serializable_content = str(function_response_content)
         elif not isinstance(function_response_content, (str, int, float, bool, type(None))):
             # Convert other non-simple types to string
             serializable_content = str(function_response_content)
         else:
             serializable_content = function_response_content

         # Structure required by the Gemini API (FunctionResponse part)
         return Part.from_function_response(
             name=function_name,
             response={
                 "content": serializable_content, # The result must be under 'content' key
             }
         )

    # chat_with_gemini remains largely the same, uses prepare_function_response_part
    def chat_with_gemini(self, user_input):
        """Handles the chat interaction, including function calls."""
        prompt = f"{user_input}"
        current_tools = self._prepare_tools()
        tools_list = [current_tools] if current_tools else None

        if not hasattr(self, 'chat') or self.chat is None:
             print("FATAL INTERNAL ERROR: self.chat attribute is missing or None before send_message.")
             error_message = "Sorry, there's an internal setup error preventing me from chatting."
             self._log_interaction(user_input, f"[ERROR: {error_message}]")
             return error_message

        try:
            print(f"DEBUG: Sending prompt to Gemini: '{prompt[:100]}...' with tools: {'Yes' if tools_list else 'No'}")
            response = self.chat.send_message(prompt, tools=tools_list)

            while True:
                # Check for function call in the latest candidate
                latest_candidate = response.candidates[0]
                if latest_candidate.content.parts and latest_candidate.content.parts[0].function_call:
                    function_call = latest_candidate.content.parts[0].function_call
                    print(f"DEBUG: Model requested function call: {function_call.name}")

                    function_name, function_result = self._execute_function_call(function_call)
                    function_response_part = self._prepare_function_response_part(function_name, function_result)

                    print(f"DEBUG: Sending function response back to model for {function_name}")
                    response = self.chat.send_message(function_response_part) # Send the Part back
                else:
                    # No more function calls, break the loop
                    break

            # --- Extract Final Text Response ---
            final_text = ""
            # Access text safely using response.text property
            if hasattr(response, 'text'):
                 final_text = response.text
            # Fallback check in parts if .text is missing (less common now)
            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))


            # Handle blocked or empty responses
            if not final_text:
                 feedback = response.prompt_feedback
                 if feedback and feedback.block_reason:
                      print(f"Warning: Response blocked. Reason: {feedback.block_reason}")
                      safety_ratings_str = ""
                      if latest_candidate.safety_ratings:
                           safety_ratings_str = f" Safety Ratings: {latest_candidate.safety_ratings}"
                      return f"[Response blocked due to: {feedback.block_reason}.{safety_ratings_str}]"
                 else:
                      print(f"Warning: Received empty final text response. Parts: {latest_candidate.content.parts if latest_candidate else 'N/A'}")
                      return "[Alpha had no text response, and no block reason found.]"

            self._log_interaction(user_input, final_text)
            return final_text

        except Exception as e:
            print(f"\nERROR: Unexpected chat error in chat_with_gemini: {type(e).__name__} - {e}")
            traceback.print_exc()
            error_message = f"Sorry, an unexpected error occurred ({type(e).__name__}) while processing your request."
            self._log_interaction(user_input, f"[ERROR: {error_message}]")
            return error_message

    # speak remains the same
    def speak(self, output):
        """Prints the output with a typing effect."""
        print("\nAlpha: ", end='')
        output_text = output if output else "[No response received]"
        try:
            for char in str(output_text): # Ensure output is string
                print(char, end='', flush=True)
                time.sleep(TYPING_SPEED)
        except Exception as e:
             print(f"[Speak animation error: {e}]", end='')
        print("\n")


# --- Tool Functions (Updated and New) ---
# NOTE: Tool functions now need 'self' as the first argument if they need access
# to instance attributes, though for static-like file operations, it's often not
# strictly needed but good practice if interaction with self.chat or other state arises.
# For simplicity here, we'll mostly keep them operating directly on files via helpers.
# If a function *needed* instance state (e.g., self.chat), add 'self' parameter.

# === Personal Info Functions (Add Edit) ===

@Alpha.add_func
def add_personal_info(self, label: str, value: str):
    """
    Save a NEW piece of static personal information (like name, preference, occupation). Use edit_personal_info to change existing info.
    label: string: The name or key for this NEW piece of personal info (e.g., 'Name', 'Favorite Color').
    value: string: The content or value associated with the label.
    """
    if not label or not label.strip():
        return "Error: A non-empty label is required for personal info."
    if value is None:
         return "Error: A value is required for personal info."

    data = _load_memory_data()
    label_clean = label.strip()

    if "PersonalInfo" not in data: data["PersonalInfo"] = {} # Ensure section exists

    if label_clean in data["PersonalInfo"]:
         return f"Error: Personal info with label '{label_clean}' already exists. Use 'edit_personal_info' to modify it."

    data["PersonalInfo"][label_clean] = value
    print(f"DEBUG: Added Personal Info: '{label_clean}': '{value}'")

    if _save_memory_data(data):
        return f"Okay, I've added the personal info for '{label_clean}'."
    else:
        print(f"ERROR: Failed to save new personal info label '{label_clean}'.")
        # Revert in-memory change if save fails? Maybe not critical here.
        return f"Error: I tried to add personal info for '{label_clean}', but failed to save it."

@Alpha.add_func
def edit_personal_info(self, label: str, new_value: str):
    """
    Edit an EXISTING piece of static personal information. Use add_personal_info for new entries.
    label: string: The label of the personal info item to edit.
    new_value: string: The new content or value for the label.
    """
    if not label or not label.strip():
        return "Error: A label is required to edit personal info."
    if new_value is None:
         return "Error: A new value is required when editing."

    data = _load_memory_data()
    label_clean = label.strip()

    if "PersonalInfo" not in data or label_clean not in data["PersonalInfo"]:
        return f"Error: Cannot edit. Personal info with label '{label_clean}' not found. Did you mean to use 'add_personal_info'?"

    old_value = data["PersonalInfo"][label_clean]
    data["PersonalInfo"][label_clean] = new_value
    print(f"DEBUG: Edited Personal Info: '{label_clean}' from '{old_value}' to '{new_value}'")

    if _save_memory_data(data):
        return f"Okay, I've updated the personal info for '{label_clean}'."
    else:
        # Attempt to revert in-memory change if save fails
        data["PersonalInfo"][label_clean] = old_value
        print(f"ERROR: Failed to save edit for personal info label '{label_clean}'. Reverted in-memory change.")
        return f"Error: I tried to update personal info for '{label_clean}', but failed to save it. The change was not applied."


# get_personal_info remains the same
@Alpha.add_func
def get_personal_info(self, label: str = None):
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
            return f"Personal Info for '{label_strip}': {personal_info[label_strip]}"
        else:
            return f"Sorry, I couldn't find any personal info stored under the label '{label_strip}'."
    else:
        details_list = [f"- {lbl}: {val}" for lbl, val in personal_info.items()]
        if not details_list:
             return "I don't have any personal information stored yet."
        return "Here's the personal information I have stored:\n" + "\n".join(details_list)


# delete_personal_info remains the same
@Alpha.add_func
def delete_personal_info(self, label: str):
    """
    Delete a specific piece of personal information by its label.
    label: string: The specific label to delete from personal info.
    """
    if not label or not label.strip():
        return "Error: You need to provide the label of the personal info you want to delete."

    data = _load_memory_data()
    label_strip = label.strip()

    if "PersonalInfo" not in data or not isinstance(data.get("PersonalInfo"), dict) or label_strip not in data["PersonalInfo"]:
        return f"I couldn't find any personal info with the label '{label_strip}', so nothing was deleted."

    # Store old value in case save fails and we need to revert
    old_value = data["PersonalInfo"].pop(label_strip) # Pop returns the value and removes
    print(f"DEBUG: Deleted Personal Info label '{label_strip}'.")

    if _save_memory_data(data):
        return f"Okay, I've deleted the personal info labeled '{label_strip}'."
    else:
        # Attempt to revert deletion if save fails
        data["PersonalInfo"][label_strip] = old_value # Put it back
        print(f"ERROR: Failed to save deletion of personal info label '{label_strip}'. Reverted in-memory change.")
        return f"Error: I tried to delete personal info for '{label_strip}', but failed to save the change. It was not deleted."


# === Daily Schedule/Memory/Task Functions (Updated for Status, Add Edit) ===

@Alpha.add_func
def add_daily_memory(self, date_input: str, label: str, value: str, status: str = 'pending'):
    """
    Save or update a detail/task for a given date. Items added this way default to 'pending' status unless specified otherwise (e.g., 'done', 'note'). Use edit_daily_memory to change value later.
    date_input: string: The date for the memory/task (e.g., 'today', '2024-08-05').
    label: string: The name or key for this piece of information (e.g., 'Task', 'Meeting Notes').
    value: string: The content associated with the label.
    status: string [optional]: The status of the item (e.g., 'pending', 'done', 'note', 'appointment'). Defaults to 'pending'.
    """
    if not date_input or not date_input.strip(): return "Error: A date input is required."
    if not label or not label.strip(): return "Error: A non-empty label is required."
    if value is None: return "Error: A value is required."
    status_clean = status.strip().lower() if status else 'note' # Default to 'note' if status empty/None

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg: return f"Error processing date: {error_msg}"

    data = _load_memory_data()
    if "Schedule" not in data: data["Schedule"] = {}

    schedule_data = data["Schedule"]
    date_entry = schedule_data.setdefault(date_str, {}) # Get or create date dict
    label_clean = label.strip()

    # Store as a dictionary including status
    date_entry[label_clean] = {"value": value, "status": status_clean}
    print(f"DEBUG: Added/Updated daily memory for {date_str}: '{label_clean}': {{'value': '{value}', 'status': '{status_clean}'}}")

    if _save_memory_data(data):
        return f"Okay, I've added/updated the memory for {date_str} (parsed from '{date_input}') with label '{label_clean}' and status '{status_clean}'."
    else:
        # Revert needed? More complex as it might overwrite existing. Let's report error.
        print(f"ERROR: Failed to save daily memory update for date {date_str}.")
        return f"Error: I tried to update the memory for {date_str}, but failed to save it."

@Alpha.add_func
def edit_daily_memory(self, date_input: str, label: str, new_value: str):
    """
    Edit the description/value of an EXISTING daily memory/task item for a given date. Does not change status.
    date_input: string: The date of the item to edit.
    label: string: The label of the item to edit.
    new_value: string: The new description or value for the item.
    """
    if not date_input or not date_input.strip(): return "Error: A date input is required."
    if not label or not label.strip(): return "Error: A label is required."
    if new_value is None: return "Error: A new value is required."

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg: return f"Error processing date: {error_msg}"

    data = _load_memory_data()
    label_clean = label.strip()

    if "Schedule" not in data or date_str not in data["Schedule"] or label_clean not in data["Schedule"][date_str]:
        return f"Error: Cannot edit. Item with label '{label_clean}' not found for date {date_str}."

    # Check if the item has the expected structure
    item = data["Schedule"][date_str][label_clean]
    if isinstance(item, dict) and "value" in item:
        old_value = item["value"]
        item["value"] = new_value # Update only the value
        print(f"DEBUG: Edited daily memory value for {date_str}, label '{label_clean}'.")
    elif isinstance(item, str): # Handle older format gracefully if encountered?
        print(f"Warning: Editing daily memory item '{label_clean}' on {date_str} which was stored as a simple string. Overwriting.")
        old_value = item
        # Overwrite with new structure (assuming 'note' status if old format)
        data["Schedule"][date_str][label_clean] = {"value": new_value, "status": "note"}
    else:
        return f"Error: The item '{label_clean}' on {date_str} has an unexpected format and cannot be edited."


    if _save_memory_data(data):
        return f"Okay, I've updated the value for item '{label_clean}' on {date_str}."
    else:
        # Revert change
        if isinstance(data["Schedule"][date_str][label_clean], dict):
             data["Schedule"][date_str][label_clean]["value"] = old_value
        else: # Revert potential overwrite of old string format
             data["Schedule"][date_str][label_clean] = old_value
        print(f"ERROR: Failed to save edit for daily memory {date_str} label '{label_clean}'. Reverted in-memory change.")
        return f"Error: I tried to update the item '{label_clean}' on {date_str}, but failed to save it."


@Alpha.add_func
def get_daily_memory(self, date_input: str, label: str = None):
    """
    Retrieve schedule/memory/task details for a specific date, including status. Understands various date formats. Can get all details or just one label for the date.
    date_input: string: The date to retrieve memories for (e.g., 'today', '2024-08-05'). Handles year clarification if needed.
    label: string [optional]: The specific label to retrieve for that date. If omitted, returns all details for the date.
    """
    if not date_input or not date_input.strip(): return "Error: A date input is required."

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg: return f"Error processing date: {error_msg}"

    data = _load_memory_data()
    schedule_data = data.get("Schedule", {})

    if not isinstance(schedule_data, dict) or date_str not in schedule_data:
        return f"I don't have any daily memories stored for {date_str} (parsed from '{date_input}')."

    date_entry = schedule_data[date_str]
    if not date_entry: # Handle empty dict for the date
        return f"I have an entry for {date_str}, but no specific details are stored for that day."

    if label:
        label_strip = label.strip()
        if label_strip in date_entry:
            item = date_entry[label_strip]
            # Format based on whether it's new structure (dict) or potentially old (str)
            if isinstance(item, dict):
                 value = item.get('value', '[No Value]')
                 status = item.get('status', 'unknown')
                 return f"On {date_str}, the detail for '{label_strip}' is: '{value}' (Status: {status})"
            else: # Assume old string format
                 return f"On {date_str}, the detail for '{label_strip}' is: {item} (Status: unknown - old format)"
        else:
            return f"I found memories for {date_str}, but couldn't find the specific label '{label_strip}'."
    else:
        details_list = []
        for lbl, item in date_entry.items():
            if isinstance(item, dict):
                 value = item.get('value', '[No Value]')
                 status = item.get('status', 'unknown')
                 details_list.append(f"- {lbl}: '{value}' (Status: {status})")
            else: # Assume old string format
                 details_list.append(f"- {lbl}: {item} (Status: unknown - old format)")

        if not details_list:
             return f"I have an entry for {date_str}, but no specific details are stored for that day."
        return f"Here are the daily memories for {date_str} (parsed from '{date_input}'):\n" + "\n".join(details_list)


@Alpha.add_func
def mark_task_done(self, date_input: str, label: str):
    """
    Marks a specific daily schedule item (task) as 'done' for a given date.
    date_input: string: The date of the task (e.g., 'today', '2024-08-05').
    label: string: The label of the task item to mark as done.
    """
    if not date_input or not date_input.strip(): return "Error: A date input is required."
    if not label or not label.strip(): return "Error: A label is required."

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg: return f"Error processing date: {error_msg}"

    data = _load_memory_data()
    label_clean = label.strip()

    if "Schedule" not in data or date_str not in data["Schedule"] or label_clean not in data["Schedule"][date_str]:
        return f"Error: Cannot mark as done. Item '{label_clean}' not found for date {date_str}."

    item = data["Schedule"][date_str][label_clean]
    original_status = 'unknown'

    if isinstance(item, dict) and "status" in item:
        original_status = item["status"]
        if original_status == 'done':
             return f"Item '{label_clean}' on {date_str} is already marked as 'done'."
        item["status"] = "done"
        print(f"DEBUG: Marked daily item '{label_clean}' on {date_str} as done.")
    elif isinstance(item, str):
        # If it was a string, convert to dict format and mark done
        print(f"Warning: Marking item '{label_clean}' on {date_str} as done. It was previously stored as a simple string. Converting format.")
        value = item
        data["Schedule"][date_str][label_clean] = {"value": value, "status": "done"}
        original_status = '[string format]' # Indicate it was converted
    else:
        return f"Error: The item '{label_clean}' on {date_str} has an unexpected format and cannot be marked as done."

    if _save_memory_data(data):
        return f"Okay, I've marked item '{label_clean}' on {date_str} as 'done'."
    else:
        # Revert status change
        if isinstance(data["Schedule"][date_str][label_clean], dict):
            data["Schedule"][date_str][label_clean]["status"] = original_status
        elif original_status == '[string format]': # Revert conversion
             data["Schedule"][date_str][label_clean] = data["Schedule"][date_str][label_clean]["value"] # Put string back
        print(f"ERROR: Failed to save status change for item '{label_clean}' on {date_str}. Reverted in-memory change.")
        return f"Error: I tried to mark '{label_clean}' on {date_str} as done, but failed to save it."


@Alpha.add_func
def get_pending_tasks(self, date_input: str = None):
    """
    List all daily schedule items that are NOT marked as 'done'. Can filter by a specific date or show all pending items across all dates.
    date_input: string [optional]: The specific date to check for pending tasks (e.g., 'today', '2024-08-06'). If omitted, checks all dates.
    """
    data = _load_memory_data()
    schedule_data = data.get("Schedule", {})
    if not schedule_data:
        return "There are no daily schedule items stored."

    pending_tasks = []
    target_date_str = None
    error_msg = None

    if date_input and date_input.strip():
        target_date_str, error_msg = _parse_and_standardize_date(date_input)
        if error_msg: return f"Error processing date: {error_msg}"
        if target_date_str not in schedule_data:
             return f"No schedule items found for {target_date_str} (parsed from '{date_input}')."
        dates_to_check = [target_date_str]
        date_scope_msg = f"on {target_date_str}"
    else:
        dates_to_check = sorted(schedule_data.keys())
        date_scope_msg = "across all dates"

    for date_str in dates_to_check:
        if date_str in schedule_data: # Check again in case target_date_str was the only key
            for label, item in schedule_data[date_str].items():
                 is_pending = False
                 if isinstance(item, dict):
                      # Consider pending if status is not 'done'
                      if item.get('status', 'note') != 'done':
                           is_pending = True
                           status = item.get('status', 'note')
                           value = item.get('value', '[No Value]')
                 elif isinstance(item, str):
                      # Treat old string format as pending 'note' if no date filter,
                      # or if date filter matches. Assume it's not 'done'.
                       is_pending = True
                       status = '[note - old format]'
                       value = item

                 if is_pending:
                      pending_tasks.append({"date": date_str, "label": label, "value": value, "status": status})

    if not pending_tasks:
        return f"No pending tasks found {date_scope_msg} (items not marked as 'done')."
    else:
        results_str = f"Here are the pending tasks/items {date_scope_msg} (not marked 'done'):\n"
        for task in pending_tasks:
             results_str += f"- {task['date']} - {task['label']}: '{task['value']}' (Status: {task['status']})\n"
        return results_str.strip()


# list_memory_dates remains the same
@Alpha.add_func
def list_memory_dates(self):
    """
    List all dates (YYYY-MM-DD) that currently have daily schedule/memory entries stored.
    """
    data = _load_memory_data()
    schedule_data = data.get("Schedule", {})
    if not isinstance(schedule_data, dict):
        print("Warning: Schedule data in memory file is not a dictionary.")
        return "Could not retrieve memory dates due to an internal format issue."
    dates = sorted(list(schedule_data.keys()))
    if not dates:
        return "I don't have any daily memories stored for any dates yet."
    else:
        if len(dates) > 15:
            return f"I have daily memory entries for {len(dates)} dates, including:\n" + "\n".join(dates[:15]) + "\n...and more."
        else:
            return "I have daily memory entries for the following dates:\n" + "\n".join(dates)


# delete_daily_memory remains the same (logically, operates on labels within date)
@Alpha.add_func
def delete_daily_memory(self, date_input: str, label: str = None):
    """
    Delete schedule/memory details for a specific date. Can delete a specific label or the entire date's entry.
    date_input: string: The date to delete memories from (e.g., 'today', '2024-08-05').
    label: string [optional]: The specific label to delete for that date. If omitted, deletes all entries for the entire date.
    """
    if not date_input or not date_input.strip(): return "Error: A date input is required."

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg: return f"Error processing date: {error_msg}"

    data = _load_memory_data()

    if "Schedule" not in data or not isinstance(data.get("Schedule"), dict):
         return "I don't have any daily schedule/memory data to delete from."

    schedule_data = data["Schedule"]

    if date_str not in schedule_data:
        return f"I couldn't find any daily memories for {date_str} (parsed from '{date_input}'), so nothing was deleted."

    if label:
        label_strip = label.strip()
        if label_strip in schedule_data[date_str]:
            # Pop the item to potentially revert if save fails
            popped_item = schedule_data[date_str].pop(label_strip)
            print(f"DEBUG: Deleted daily memory label '{label_strip}' for date {date_str}.")

            if not schedule_data[date_str]: # If date entry becomes empty
                del schedule_data[date_str]
                print(f"DEBUG: Daily memory entry for {date_str} became empty and was removed.")

            if _save_memory_data(data):
                 return f"Okay, I've deleted the memory label '{label_strip}' for date {date_str}."
            else:
                 # Revert deletion
                 if date_str not in schedule_data: schedule_data[date_str] = {} # Recreate date dict if removed
                 schedule_data[date_str][label_strip] = popped_item # Put item back
                 print(f"ERROR: Failed to save deletion of label '{label_strip}' for {date_str}. Reverted in-memory change.")
                 return f"Error: I tried to delete the label '{label_strip}' for {date_str}, but failed to save the change."
        else:
            return f"I found memories for {date_str}, but couldn't find the label '{label_strip}' to delete."
    else:
        # Delete the entire entry for the date
        popped_date_entry = schedule_data.pop(date_str) # Pop the whole date dict
        print(f"DEBUG: Deleted all daily memory entries for date {date_str}.")
        if _save_memory_data(data):
            return f"Okay, I've deleted all daily memory entries for date {date_str}."
        else:
            # Revert deletion
            schedule_data[date_str] = popped_date_entry # Put whole dict back
            print(f"ERROR: Failed to save deletion of date {date_str}. Reverted in-memory change.")
            return f"Error: I tried to delete all memories for {date_str}, but failed to save the change."


# Rename search_memories_by_content to be more specific
@Alpha.add_func
def search_daily_memories(self, search_term: str):
    """
    Searches through all DAILY schedule/memory entries (labels, values, statuses) for a specific keyword (case-insensitive). Does NOT search Personal Info or General Notes.
    search_term: string: The text to search for within the daily memories.
    """
    if not search_term or not search_term.strip():
        return "Error: Please provide a search term for daily memories."

    data = _load_memory_data()
    schedule_data = data.get("Schedule", {})

    if not isinstance(schedule_data, dict) or not schedule_data:
        return "I don't have any daily schedule entries stored to search."

    search_term_lower = search_term.lower().strip()
    matches = []
    for date_str in sorted(schedule_data.keys()):
        details = schedule_data[date_str]
        if isinstance(details, dict):
            for label, item in details.items():
                 value_str = ""
                 status_str = ""
                 if isinstance(item, dict):
                      value_str = str(item.get("value", ""))
                      status_str = str(item.get("status", ""))
                 elif isinstance(item, str): # Handle old format
                      value_str = item

                 # Check label, value, and status
                 if search_term_lower in label.lower() or \
                    search_term_lower in value_str.lower() or \
                    (status_str and search_term_lower in status_str.lower()):
                     matches.append({"date": date_str, "label": label, "value": value_str, "status": status_str})
        # else: Pass silently on non-dict date entries

    if not matches:
        return f"I searched daily memories but couldn't find any mention of '{search_term}'."
    else:
        results_str = f"I found mentions of '{search_term}' in daily memories:\n"
        max_results = 20
        for i, match in enumerate(matches):
             if i >= max_results:
                 results_str += f"... (and {len(matches) - max_results} more matches)\n"
                 break
             status_part = f" (Status: {match['status']})" if match['status'] else ""
             truncated_value = match['value'][:60] + ('...' if len(match['value']) > 60 else '')
             results_str += f"- {match['date']} - {match['label']}: '{truncated_value}'{status_part}\n"
        return results_str.strip()


# === General Notes Functions (New Section) ===

@Alpha.add_func
def add_note(self, topic: str, content: str):
    """
    Save or update a general note identified by a topic/title. Not tied to a specific date.
    topic: string: The title or key for this note (e.g., 'Project Ideas', 'Shopping List').
    content: string: The main text content of the note.
    """
    if not topic or not topic.strip(): return "Error: A topic/title is required for notes."
    if content is None: return "Error: Content is required for notes." # Allow empty string

    data = _load_memory_data()
    topic_clean = topic.strip()
    # Ensure Notes section exists
    if "Notes" not in data or not isinstance(data["Notes"], dict):
         data["Notes"] = {}

    data["Notes"][topic_clean] = content
    print(f"DEBUG: Added/Updated Note: '{topic_clean}'")

    if _save_memory_data(data):
        return f"Okay, I've saved the note under the topic '{topic_clean}'."
    else:
        print(f"ERROR: Failed to save note '{topic_clean}'.")
        return f"Error: I tried to save the note '{topic_clean}', but failed."

@Alpha.add_func
def get_note(self, topic: str):
    """
    Retrieve the content of a general note by its topic/title.
    topic: string: The exact topic/title of the note to retrieve.
    """
    if not topic or not topic.strip(): return "Error: A topic/title is required to get a note."

    data = _load_memory_data()
    notes_data = data.get("Notes", {})
    topic_clean = topic.strip()

    if not isinstance(notes_data, dict):
        return "Error: The notes section seems to be corrupted."

    if topic_clean in notes_data:
        content = notes_data[topic_clean]
        return f"Note for topic '{topic_clean}':\n---\n{content}\n---"
    else:
        return f"Sorry, I couldn't find a note with the topic '{topic_clean}'."

@Alpha.add_func
def list_notes(self):
    """
    List the topics/titles of all saved general notes.
    """
    data = _load_memory_data()
    notes_data = data.get("Notes", {})

    if not isinstance(notes_data, dict):
        return "Error: The notes section seems to be corrupted."

    topics = sorted(list(notes_data.keys()))

    if not topics:
        return "I don't have any general notes saved yet."
    else:
        return "Here are the topics of the saved general notes:\n- " + "\n- ".join(topics)

@Alpha.add_func
def search_notes(self, search_term: str):
    """
    Search through the topics and content of all general notes for a keyword (case-insensitive).
    search_term: string: The text to search for within notes.
    """
    if not search_term or not search_term.strip():
        return "Error: Please provide a term to search for in notes."

    data = _load_memory_data()
    notes_data = data.get("Notes", {})
    if not isinstance(notes_data, dict): return "Error: Notes section corrupted."
    if not notes_data: return "No general notes to search through."

    search_term_lower = search_term.lower().strip()
    matches = []
    for topic, content in notes_data.items():
        content_str = str(content) # Ensure content is string
        if search_term_lower in topic.lower() or search_term_lower in content_str.lower():
            matches.append({"topic": topic, "content_preview": content_str[:100] + ('...' if len(content_str) > 100 else '')})

    if not matches:
        return f"I searched the general notes but couldn't find any mention of '{search_term}'."
    else:
        results_str = f"Found mentions of '{search_term}' in the following notes:\n"
        for match in matches:
             results_str += f"- Topic: {match['topic']} (Preview: {match['content_preview']})\n"
        return results_str.strip()

@Alpha.add_func
def delete_note(self, topic: str):
    """
    Delete a general note identified by its topic/title.
    topic: string: The exact topic/title of the note to delete.
    """
    if not topic or not topic.strip(): return "Error: A topic/title is required to delete a note."

    data = _load_memory_data()
    topic_clean = topic.strip()

    if "Notes" not in data or not isinstance(data.get("Notes"), dict):
         return "There are no notes stored to delete from." # Or Error: Notes section corrupted.

    notes_data = data["Notes"]

    if topic_clean not in notes_data:
        return f"I couldn't find a note with the topic '{topic_clean}', so nothing was deleted."

    # Pop the item for potential revert
    popped_content = notes_data.pop(topic_clean)
    print(f"DEBUG: Deleted note with topic '{topic_clean}'.")

    if _save_memory_data(data):
        return f"Okay, I've deleted the note with topic '{topic_clean}'."
    else:
        # Revert deletion
        notes_data[topic_clean] = popped_content
        print(f"ERROR: Failed to save deletion of note '{topic_clean}'. Reverted in-memory change.")
        return f"Error: I tried to delete the note '{topic_clean}', but failed to save the change."


# === Utility Functions ===

# get_current_date_time remains the same
@Alpha.add_func
def get_current_date_time(self):
    """
    Get the current local date and time, formatted for readability.
    """
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S (%A)")
    return f"The current date and time is {formatted_datetime}"


# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Alpha Assistant (Gemini Edition - Enhanced Memory v2) ---")
    print("Initializing...")

    # Initialize memory file with correct structure if it doesn't exist or needs update
    initial_data = _load_memory_data() # This now handles creation/migration
    # Simple check after load confirms basic structure before proceeding
    if not isinstance(initial_data, dict) or \
       "PersonalInfo" not in initial_data or \
       "Schedule" not in initial_data or \
       "Notes" not in initial_data:
         print(f"FATAL ERROR: Memory file {MEMORY_FILE} structure issue persists after loading attempts. Exiting.")
         sys.exit(1)
    else:
        print(f"DEBUG: Memory file {MEMORY_FILE} loaded/initialized successfully.")


    try:
        ai = Alpha() # Instantiate Alpha
        print(f"Using model: {MODEL_NAME}")
        print(f"Memory file: {MEMORY_FILE}")
        print(f"History file: {HISTORY_FILE}")
        registered_tool_names = list(ai.registered_functions.keys())
        if registered_tool_names:
             print(f"Registered tools ({len(registered_tool_names)}): {', '.join(sorted(registered_tool_names))}")
        else:
             print("Warning: No tools were successfully registered.")
        print("-" * 30 + f"\nAlpha is ready. Typing speed: {TYPING_SPEED}s/char. Type 'exit' or 'quit' to end.\n" + "-" * 30)

    except Exception as init_error:
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
                 break

            if not user_input: continue
            if user_input.lower() in ["exit", "quit", "goodbye", "bye", "0"]:
                ai.speak("Okay, goodbye!")
                break

            output = ai.chat_with_gemini(user_input)
            ai.speak(output)

    except KeyboardInterrupt:
        print("\n\nAlpha: Detected interrupt. Shutting down...")
        try: ai.speak("Okay, exiting now.")
        except Exception: print("Exiting immediately.")
    except Exception as main_loop_error:
        print(f"\nFATAL ERROR during main chat loop: {main_loop_error}")
        traceback.print_exc()
        try: ai.speak("A critical error occurred. I need to shut down.")
        except Exception: print("Critical error. Shutting down.")
    finally:
        print("\n" + "-" * 30 + "\nAlpha: Session ended.")
        # Ensure files are mentioned even if path is relative/default
        print(f"Chat history saved in: {os.path.abspath(HISTORY_FILE)}")
        print(f"Memory data stored in: {os.path.abspath(MEMORY_FILE)}")
        print(f"Config read from: {os.path.abspath('config.ini')}")
        print("-" * 30)