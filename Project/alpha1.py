# -*- coding: utf-8 -*-
import sys
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Tool, FunctionDeclaration
# Explicitly import specific exceptions if needed later for finer control
# from google.api_core.exceptions import GoogleAPIError
from google.generativeai.types import StopCandidateException # Import specific exception for handling

import json
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timedelta # Keep this
from dateutil import parser as date_parser # Keep this
from dateutil.parser import ParserError # Keep this
import re
import traceback # Import traceback for better error logging
import tempfile # For atomic file writes
import uuid # For unique reminder IDs
import schedule # For scheduling reminder checks
import threading # For running scheduler in background (CLI ONLY)
from typing import Any # For type hinting

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("FATAL ERROR: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
# Use a production-ready model, adjust if necessary
# MODEL_NAME = "gemini-1.5-pro-latest"
MODEL_NAME = "gemini-1.5-flash-latest" # Use Flash if speed/cost is a concern

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
MEMORY_FILE = "alpha_memory_v3.json" # Keep version suffix if structure changed
HISTORY_FILE = "alpha_chat_history_v3.log" # Keep version suffix
# --- Default structure includes "Reminders" list ---
DEFAULT_MEMORY_STRUCTURE = {"PersonalInfo": {}, "Schedule": {}, "Reminders": []}

# --- Helper Functions ---

def _load_memory_data(filename=MEMORY_FILE) -> dict:
    """Loads the entire memory structure (PersonalInfo, Schedule, Reminders). Handles migration."""
    if not os.path.exists(filename):
        print(f"DEBUG: Memory file {filename} not found. Initializing with default structure.")
        return DEFAULT_MEMORY_STRUCTURE.copy() # Return a copy
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                print(f"DEBUG: Memory file {filename} is empty. Initializing with default structure.")
                return DEFAULT_MEMORY_STRUCTURE.copy()
            data = json.loads(content)

        # --- Structure Validation and Migration ---
        if isinstance(data, dict):
            # Check for the new expected structure (includes Reminders)
            is_valid_new = (
                "PersonalInfo" in data and isinstance(data.get("PersonalInfo"), dict) and
                "Schedule" in data and isinstance(data.get("Schedule"), dict) and
                "Reminders" in data and isinstance(data.get("Reminders"), list) # Check Reminders list
            )
            if is_valid_new:
                 # print(f"DEBUG: Loaded valid v3 structure from {filename}.") # Optional: reduce verbosity
                 return data # Return loaded data

            # --- Migration Logic ---
            # Check if it has PersonalInfo and Schedule but *not* Reminders (previous version v2)
            has_prev_structure = (
                "PersonalInfo" in data and isinstance(data.get("PersonalInfo"), dict) and
                "Schedule" in data and isinstance(data.get("Schedule"), dict) and
                "Reminders" not in data # Explicitly check Reminders is missing
            )
            if has_prev_structure:
                print(f"Warning: Old memory format (v2) detected in {filename}. Adding 'Reminders' section.")
                data["Reminders"] = [] # Add the empty list
                 # Attempt to save the migrated structure immediately
                if _save_memory_data(data, filename):
                    print(f"DEBUG: Successfully added 'Reminders' section and saved to {filename}.")
                    return data
                else:
                    print(f"ERROR: Failed to save migrated data (added Reminders) to {filename}. Using structure in memory only.")
                    return data # Return the migrated structure even if save failed

            # --- Handle other invalid/old formats ---
            # Check if it looks like the VERY OLD structure (only date keys at top level, v1)
            is_very_old_format = all(re.match(r'^\d{4}-\d{2}-\d{2}$', k) for k in data.keys()) and \
                                all(isinstance(v, dict) for v in data.values())
            if is_very_old_format and "PersonalInfo" not in data and "Schedule" not in data:
                print(f"Warning: Very old memory format (v1) detected in {filename}. Migrating to new structure.")
                migrated_data = {"PersonalInfo": {}, "Schedule": data, "Reminders": []}
                if _save_memory_data(migrated_data, filename):
                    print(f"DEBUG: Successfully migrated v1 data and saved to new format in {filename}.")
                    return migrated_data
                else:
                    print(f"ERROR: Failed to save migrated v1 data to {filename}. Using structure in memory only.")
                    return migrated_data
            else:
                # Doesn't match known structures
                print(f"Warning: Data in {filename} does not match known structures (v1, v2, v3). Initializing with default.")
                return DEFAULT_MEMORY_STRUCTURE.copy()
        else:
             print(f"Warning: Data in {filename} is not a dictionary. Initializing with default structure.")
             return DEFAULT_MEMORY_STRUCTURE.copy()

    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {filename}. File might be corrupted. Initializing with default structure.")
        traceback.print_exc()
        return DEFAULT_MEMORY_STRUCTURE.copy()
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading data from {filename}: {e}. Initializing with default structure.")
        traceback.print_exc()
        return DEFAULT_MEMORY_STRUCTURE.copy()

def _save_memory_data(data: dict, filename: str = MEMORY_FILE) -> bool:
    """Saves the entire memory structure atomically (PersonalInfo, Schedule, Reminders)."""
    # --- Structure Validation before saving ---
    if not isinstance(data, dict):
         print(f"ERROR: Attempted to save non-dictionary data structure to {filename}. Save aborted.")
         return False
    if "PersonalInfo" not in data or not isinstance(data.get("PersonalInfo"), dict):
         print(f"ERROR: Attempted to save invalid data structure (missing/invalid 'PersonalInfo') to {filename}. Save aborted.")
         return False
    if "Schedule" not in data or not isinstance(data.get("Schedule"), dict):
        print(f"ERROR: Attempted to save invalid data structure (missing/invalid 'Schedule') to {filename}. Save aborted.")
        return False
    if "Reminders" not in data or not isinstance(data.get("Reminders"), list):
         print(f"ERROR: Attempted to save invalid data structure (missing/invalid 'Reminders' list) to {filename}. Save aborted.")
         return False
    # --- End Validation ---

    temp_file_path = None
    try:
        file_dir = os.path.dirname(filename)
        if not file_dir: file_dir = "." # Handle case where filename has no directory part
        os.makedirs(file_dir, exist_ok=True)

        # Write to a temporary file first
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=file_dir, suffix='.tmp') as tf:
            temp_file_path = tf.name
            json.dump(data, tf, indent=4, sort_keys=True) # Use sort_keys for consistent file diffs

        # Atomically replace the original file with the temporary file
        os.replace(temp_file_path, filename)
        # print(f"DEBUG: Memory data saved successfully to {filename}") # Optional: Confirmation log
        return True

    except IOError as e:
        print(f"ERROR: File I/O error saving memory data to {filename}: {e}")
        traceback.print_exc()
        if temp_file_path and os.path.exists(temp_file_path): # Clean up temp file on IO error
             try: os.remove(temp_file_path)
             except OSError as ose: print(f"Warning: Could not remove temporary file {temp_file_path} after IO error: {ose}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error saving memory data to {filename}: {e}")
        traceback.print_exc()
        # Don't remove the temp file here automatically, it might contain the data we wanted to save
        # If os.replace fails, temp_file_path might still exist but wasn't moved.
        return False
    finally:
        # Ensure temporary file is removed *if it still exists* after a failed os.replace or other exception *after* writing
        if temp_file_path and os.path.exists(temp_file_path) and not os.path.exists(filename):
             pass
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                 os.remove(temp_file_path)
            except OSError as e:
                print(f"Warning: Could not remove temporary file {temp_file_path}: {e}")


def _parse_and_standardize_date(date_input: str) -> tuple[str | None, str | None]:
    """
    Parses various date string inputs into a standardized YYYY-MM-DD format.
    Handles relative terms ('today', 'yesterday', 'tomorrow') and checks for year ambiguity.
    Returns (standardized_date, error_message). error_message is None on success.
    """
    if not date_input or not date_input.strip():
        return None, "Error: Date input cannot be empty."

    today = datetime.now().date()
    date_input_lower = date_input.lower().strip()
    standardized_date = None
    error_msg = None
    needs_year_clarification = False
    dt = None # Initialize dt

    try:
        # Use fuzzy=True primarily for better natural language handling
        print(f"DEBUG [_parse_and_standardize_date]: Attempting parse: '{date_input}' (fuzzy=True)")
        # Provide a default datetime (now) for context, especially for relative terms
        parsed_dt_obj = date_parser.parse(date_input, default=datetime.now(), fuzzy=True)
        dt = parsed_dt_obj.date() # We only care about the date part for daily memories
        print(f"DEBUG [_parse_and_standardize_date]: Parsed successfully to: {dt}")

        # Check for year ambiguity ONLY if the input doesn't explicitly mention a year (4 digits)
        has_year_regex = re.compile(r'\b\d{4}\b')
        # Simplified check for common relative terms that imply context
        is_explicitly_relative = any(term in date_input_lower for term in ["today", "yesterday", "tomorrow", "last ", "next ", "this "])

        # If no year mentioned, not explicitly relative, and year defaulted to current year...
        if not has_year_regex.search(date_input) and \
           not is_explicitly_relative and \
           parsed_dt_obj.year == datetime.now().year:
             # ...and it looks like a month-day format (could be ambiguous)
             # Regex: Month name/num, separator, Day num
             month_day_pattern = re.compile(r'^\s*([a-z]{3,}|[0-9]{1,2})[\s/.-]+([0-9]{1,2})\s*$')
             # Examples: "march 5", "aug 10", "12/25", "01-15"
             if month_day_pattern.match(date_input_lower):
                 needs_year_clarification = True
                 print(f"DEBUG [_parse_and_standardize_date]: Flagged '{date_input}' for potential year clarification (parsed as {dt}).")


    except (ParserError, ValueError, OverflowError, TypeError) as e:
        error_detail = f"{type(e).__name__}: {e}"
        print(f"ERROR [_parse_and_standardize_date]: Parsing failed for '{date_input}'. Details: {error_detail}")
        error_msg = f"Could not understand the date '{date_input}'. Error: {error_detail}. Please use formats like YYYY-MM-DD, 'today', 'Aug 5 2023', 'last Friday'."
    except Exception as e: # Catch unexpected parsing errors
        error_detail = f"Unexpected {type(e).__name__}: {e}"
        print(f"ERROR [_parse_and_standardize_date]: Unexpected parsing error for '{date_input}'. Details: {error_detail}")
        traceback.print_exc()
        error_msg = f"An unexpected error occurred while parsing the date '{date_input}'. Error: {error_detail}."

    # Final determination
    if dt:
        if needs_year_clarification:
             # Specific message for the LLM to handle ambiguity
             error_msg = f"AMBIGUOUS_DATE: The date '{date_input}' parsed as {dt.strftime('%Y-%m-%d')}, but the year was not specified. Please ask the user to clarify the year (e.g., 'this year', 'last year', '2023') and call the tool again with the combined phrase (e.g., '{date_input} this year')."
             standardized_date = None # Don't return a date if ambiguous
             print(f"DEBUG [_parse_and_standardize_date]: Returning AMBIGUOUS_DATE error for '{date_input}'.")
        else:
            standardized_date = dt.strftime('%Y-%m-%d')
    elif not error_msg:
         print(f"WARN [_parse_and_standardize_date]: Date parsing resulted in None for '{date_input}' without a specific error message captured.")
         error_msg = f"Could not determine a valid date from '{date_input}'."

    return standardized_date, error_msg


# --- Alpha Class ---
class Alpha:
    tool_declarations: list[FunctionDeclaration] = []
    registered_functions: dict[str, callable] = {}

    def __init__(self):
        print("DEBUG: Initializing Alpha instance...")
        # <<< Refined System Prompt >>>
        self.system_prompt = (
            "You are Alpha, my personal assistant. Be helpful, concise, and friendly. "
            "You have tools to manage memory and set reminders:\n"
            "1.  **Personal Info:** Static details (name, preferences). Use `add_personal_info`, `get_personal_info`, `delete_personal_info`.\n"
            "2.  **Daily Schedule/Memories:** Date-based activities/notes. Use `add_daily_memory`, `get_daily_memory`, `delete_daily_memory`, `list_memory_dates`, `search_memories_by_content`.\n"
            "   - **For `add_daily_memory`:** You MUST extract and provide arguments for `date_input`, a short descriptive `label` (e.g., 'Work', 'Event', 'Note', 'Appointment'), and the detailed `value` (the actual memory content). Do not omit the label or value.\n"
            "3.  **Reminders:** Set reminders for specific times. Use `set_reminder`. Reminders will trigger automatically when due (the user will be notified outside this chat).\n\n"
            "**IMPORTANT GUIDELINES:**\n"
            "-   **Pass Full Date Phrase:** When calling memory or reminder tools needing a date/time, use the user's full original phrase for the `date_input` or `reminder_time_input` argument.\n"
            "-   **YEAR CLARIFICATION:** If a memory tool (like `get_daily_memory` or `delete_daily_memory`) returns an error starting with `AMBIGUOUS_DATE:`, you MUST ask the user to clarify the year. Then, call the tool again with the *original phrase combined with the year clarification* (e.g., user says 'march 5', you get ambiguity error, ask user, user says 'last year', you call tool again with date_input='march 5 last year'). Do NOT guess the year.\n"
            "-   **REMINDER TIME:** For `set_reminder`, the `reminder_time_input` needs a date and time (e.g., 'today 5pm', 'tomorrow 9:30 AM', 'August 15th 2024 14:00'). Provide a clear `message`.\n"
            "-   **ERROR HANDLING:** If a tool call returns an error message (e.g., 'Error: ...', 'AMBIGUOUS_DATE: ...'), inform the user clearly about the specific error returned by the tool. Do not invent reasons for failure. If the tool succeeds, summarize the outcome based on the success message from the tool.\n"
            "-   **CURRENT TIME:** Use `get_current_date_time` if asked for the current time.\n"
        )

        self.model = None
        self.chat = None
        try:
            print(f"DEBUG: Configuring GenerativeModel: {MODEL_NAME}")
            self.model = genai.GenerativeModel(
                MODEL_NAME,
                safety_settings=SAFETY_SETTINGS,
                system_instruction=self.system_prompt,
            )
            print("DEBUG: GenerativeModel created. Starting chat session...")
            # Enable function calling
            self.chat = self.model.start_chat(enable_automatic_function_calling=False) # Control manually
            if not self.chat:
                 raise ValueError("Failed to start chat session (self.chat is None).")
            print(f"DEBUG: Chat session successfully started with model {MODEL_NAME}.")
        except Exception as e:
            print(f"FATAL ERROR: Could not initialize GenerativeModel or ChatSession: {e}")
            traceback.print_exc()
            # Re-raise to ensure the main script knows initialization failed
            raise RuntimeError(f"Failed to initialize AI model/chat: {e}") from e

    # --- Methods ---

    def _log_interaction(self, user_input: str, alpha_response: str):
        """Logs user input and Alpha's response to the history file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Truncate potentially very long responses/inputs for logging sanity
        log_entry = (
            f"{timestamp} | User: {user_input[:1000]}{'...' if len(user_input) > 1000 else ''}\n"
            f"{timestamp} | Alpha: {alpha_response[:2000]}{'...' if len(alpha_response) > 2000 else ''}\n"
            f"{'-'*20}\n"
        )
        try:
            log_dir = os.path.dirname(HISTORY_FILE)
            if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
            with open(HISTORY_FILE, 'a', encoding='utf-8') as f: f.write(log_entry)
        except Exception as e: print(f"\nWarning: Could not write to history file {HISTORY_FILE}: {e}")

    @classmethod
    def _create_function_declaration(cls, func: callable) -> FunctionDeclaration | None:
        """Creates a FunctionDeclaration from a function's docstring."""
        if not hasattr(func, '__doc__') or not func.__doc__:
             print(f"Warning: Function '{func.__name__}' is missing a docstring. Cannot create declaration.")
             return None
        doc_lines = [line.strip() for line in func.__doc__.strip().split('\n') if line.strip()]
        if not doc_lines:
             print(f"Warning: Docstring for '{func.__name__}' is empty. Cannot create declaration.")
             return None

        description = doc_lines[0]
        parameters_schema = {'type': 'object', 'properties': {}, 'required': []}
        # Improved regex: param_name : type [optional] : description
        param_pattern = re.compile(r'^([\w_]+)\s*:\s*(\w+)\s*(?:\[(optional)\])?\s*:\s*(.*)$', re.IGNORECASE)

        for line in doc_lines[1:]:
            match = param_pattern.match(line)
            if match:
                param_name, param_type_desc, optional_flag, param_desc = match.groups()
                param_name = param_name.strip()
                param_type_desc = param_type_desc.strip().lower()
                param_desc = param_desc.strip()

                # Basic type mapping (can be expanded)
                param_type = 'string' # Default
                if 'int' in param_type_desc or 'integer' in param_type_desc: param_type = 'integer'
                elif 'float' in param_type_desc or 'number' in param_type_desc: param_type = 'number'
                elif 'bool' in param_type_desc or 'boolean' in param_type_desc: param_type = 'boolean'
                elif 'list' in param_type_desc or 'array' in param_type_desc: param_type = 'array'
                # Add 'object' if needed

                parameters_schema['properties'][param_name] = {'type': param_type, 'description': param_desc}
                if not optional_flag: # If it's not marked optional, it's required
                    parameters_schema['required'].append(param_name)

        # Finalize schema
        if not parameters_schema['properties']:
            # Function with no parameters
            return FunctionDeclaration(name=func.__name__, description=description)
        else:
            # Remove 'required' field if it's empty (API requirement)
            if not parameters_schema['required']:
                 del parameters_schema['required']
            return FunctionDeclaration(name=func.__name__, description=description, parameters=parameters_schema)

    @classmethod
    def add_func(cls, func: callable) -> callable:
        """Decorator to register functions as tools."""
        if func.__name__ not in cls.registered_functions:
            declaration = cls._create_function_declaration(func)
            if declaration:
                cls.registered_functions[func.__name__] = func
                cls.tool_declarations.append(declaration)
                print(f"DEBUG: Registered function tool: {func.__name__}")
            # else: warning already printed by _create_function_declaration
        return func

    def _prepare_tools(self) -> list[Tool] | None:
        """Prepares the list of Tool objects for the API call."""
        if Alpha.tool_declarations:
             valid_declarations = [decl for decl in Alpha.tool_declarations if decl is not None]
             if valid_declarations:
                 # Wrap the declarations in a Tool object
                 return [Tool(function_declarations=valid_declarations)]
        return None # Return None if no valid tools are registered

    def _execute_function_call(self, function_call: Any) -> tuple[str, any]:
        """Executes a function call requested by the model using registered functions."""
        function_name = function_call.name
        args = function_call.args # This is usually a Struct or dict-like object
        print(f"DEBUG [LLM Request]: Tool '{function_name}' requested.")

        if function_name in Alpha.registered_functions:
            function_to_call = Alpha.registered_functions[function_name]
            args_dict = {}
            try:
                 # Convert the Struct/dict-like args to a standard Python dict
                 args_dict = dict(args)
                 print(f"DEBUG [Tool Call Prep]: Calling {function_name} with args: {args_dict}")
            except Exception as e:
                 # Fallback conversion (less likely needed with current libraries)
                 print(f"WARN: Could not convert args for {function_name} directly to dict: {e}. Trying item iteration.")
                 try:
                    args_dict = {key: value for key, value in args.items()}
                    print(f"DEBUG [Tool Call Prep Iter]: Calling {function_name} with args: {args_dict}")
                 except Exception as e2:
                    print(f"ERROR: Failed to extract arguments for {function_name} via iteration: {e2}. Calling with no args.")
                    traceback.print_exc()
                    # Return an error immediately if args can't be processed
                    return function_name, f"Error: Internal issue processing arguments for tool '{function_name}': {e2}"

            try:
                # Execute the actual Python function
                print(f"DEBUG [Tool Call Exec]: Executing {function_name}(**{args_dict})")
                function_response = function_to_call(**args_dict)
                # Log result (truncated if very long)
                response_str = str(function_response)
                print(f"DEBUG [Tool Result]: {function_name} returned: {response_str[:500]}{'...' if len(response_str) > 500 else ''}")
                return function_name, function_response
            except TypeError as te:
                 # Specific error for wrong arguments passed to the Python function
                 error_message = (f"Error: Tool '{function_name}' was called with incompatible arguments. "
                                  f"Please check if all required arguments were provided and match expected types. "
                                  f"Details: {te}. Args received by tool: {args_dict}")
                 print(f"ERROR [Tool Execution TypeError]: {error_message}")
                 # traceback.print_exc() # Optional: uncomment for full trace
                 # Return the detailed error message TO THE MODEL
                 return function_name, error_message
            except Exception as e:
                # Catch-all for other errors during the registered function's execution
                error_message = f"Error executing tool '{function_name}': {type(e).__name__}: {e}"
                print(f"ERROR [Tool Execution Exception]: {error_message}")
                traceback.print_exc()
                # Return the error message TO THE MODEL
                return function_name, error_message
        else:
            # Model tried to call a function that wasn't registered/decorated correctly
            print(f"ERROR [Tool Execution]: Model tried to call unknown function: {function_name}")
            error_message = f"Error: Function tool '{function_name}' is not available or not registered correctly."
            # Return the error message TO THE MODEL
            return function_name, error_message

    def _prepare_function_response_dict(self, function_name: str, function_response_content: any) -> dict:
        """Formats the function execution result into the structure the API expects."""
        serialized_content: Any = None # Use Any type hint

        # Basic serialization logic (can be expanded)
        if isinstance(function_response_content, (str, int, float, bool)):
            serialized_content = function_response_content
        elif isinstance(function_response_content, (dict, list)):
            try:
                content_to_dump = function_response_content
                if isinstance(content_to_dump, dict):
                     content_to_dump = {str(k): v for k, v in content_to_dump.items()}
                serialized_content = json.dumps(content_to_dump)
            except TypeError as e:
                 print(f"Warning: Could not JSON serialize response from {function_name}. Using str(). Error: {e}")
                 serialized_content = str(function_response_content)
        elif function_response_content is None:
             serialized_content = "null"
        else:
            serialized_content = str(function_response_content)

        # Construct the response dictionary payload
        response_payload = {
            "name": function_name,
            "response": {
                "content": serialized_content
            }
        }

        # The send_message method expects a FunctionResponse object or a dict interpretable as one.
        function_response_part = {
            "function_response": response_payload
        }

        return function_response_part # Return the dict ready for send_message


    def chat_with_gemini(self, user_input: str) -> str | None:
        """Handles the chat interaction loop, including function calls."""
        prompt = user_input
        current_tools = self._prepare_tools() # Get Tool object list

        if not self.chat:
             error_message = "Internal Error: Chat session not initialized."
             print(f"FATAL INTERNAL ERROR: {error_message}")
             try: self._log_interaction(user_input, f"[FATAL ERROR: {error_message}]")
             except Exception: pass
             return f"Sorry, I encountered an internal setup error ({error_message}). Please restart or check logs."

        try:
            print(f"DEBUG: Sending prompt: '{prompt[:100]}...' ({'with' if current_tools else 'without'} tools)")
            # Initial message to the model, including the tools available
            response = self.chat.send_message(prompt, tools=current_tools)

            # Loop to handle potential sequence of function calls
            while True:
                # Check the response for a function call request
                function_call = None
                candidate = response.candidates[0] if response.candidates else None
                if candidate and candidate.content and candidate.content.parts:
                     function_call_part = next((part for part in candidate.content.parts if part.function_call), None)
                     if function_call_part:
                         function_call = function_call_part.function_call

                if function_call:
                    print(f"DEBUG: Model requested function call: {function_call.name}")
                    # Execute the requested function
                    function_name, function_result = self._execute_function_call(function_call)

                    # Prepare the function result to send back to the model
                    function_response_payload = self._prepare_function_response_dict(
                        function_name, function_result
                    )

                    print(f"DEBUG: Sending function response payload back for {function_name}: {str(function_response_payload)[:300]}...")

                    # Send the function response back to the model
                    response = self.chat.send_message(function_response_payload)
                    # Continue loop: The model might respond with text or another function call

                else:
                    # No function call requested in the latest response, break the loop
                    print("DEBUG: No further function call requested by the model.")
                    break

            # --- Extract Final Text Response ---
            final_text = ""
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))

            # --- Handle Response Issues & Finish Reason ---
            finish_reason_str = "UNKNOWN"
            block_reason_str = "N/A"
            safety_ratings_str = "N/A"

            if response.candidates:
                candidate = response.candidates[0]
                finish_reason_enum = candidate.finish_reason
                finish_reason_str = finish_reason_enum.name if finish_reason_enum else "UNKNOWN"
                if candidate.safety_ratings:
                     safety_ratings_str = str(candidate.safety_ratings)

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason_enum = response.prompt_feedback.block_reason
                 block_reason_str = block_reason_enum.name if block_reason_enum else "N/A"


            # Check for abnormal finish reasons or blocking
            if finish_reason_str not in ["STOP", "MAX_TOKENS"] or block_reason_str != "N/A":
                 print(f"Warning: Response finished abnormally or was blocked. FinishReason: {finish_reason_str}, BlockReason: {block_reason_str}, SafetyRatings: {safety_ratings_str}")
                 error_msg = "[Response generation issue]" # Default
                 if finish_reason_str == "SAFETY" or block_reason_str == "SAFETY":
                      error_msg = "[My response was blocked due to safety settings.]"
                 elif finish_reason_str == "RECITATION" or block_reason_str == "RECITATION":
                      error_msg = "[My response was blocked due to potential recitation concerns.]"
                 elif block_reason_str != "N/A":
                     error_msg = f"[My response was blocked. Reason: {block_reason_str}.]"
                 elif finish_reason_str not in ["STOP", "MAX_TOKENS"]:
                      error_msg = f"[Response generation incomplete. Reason: {finish_reason_str}.]"

                 self._log_interaction(user_input, error_msg)
                 return error_msg

            elif not final_text and finish_reason_str == "STOP":
                 print(f"Warning: Received empty final text response despite normal STOP finish reason. This might be okay if a tool executed silently.")
                 confirmation_msg = "[Action completed.]" # Or just return ""
                 self._log_interaction(user_input, confirmation_msg)
                 return confirmation_msg

            # Log the successful interaction
            self._log_interaction(user_input, final_text)
            return final_text.strip() # Return the final text response

        except StopCandidateException as sce:
            # Handle specific Gemini errors
            print(f"\nERROR: Caught StopCandidateException: {sce}")
            traceback.print_exc()
            error_message = "Sorry, I encountered an error processing that request. It seems there was an issue with my internal tools or response generation."
            finish_reason = "UNKNOWN"
            if hasattr(sce, 'candidate') and sce.candidate and sce.candidate.finish_reason:
                 finish_reason = sce.candidate.finish_reason.name
                 if "FUNCTION_CALL" in finish_reason:
                      error_message = "Sorry, I had trouble understanding exactly how to perform that action using my tools. Could you please try rephrasing?"
                 elif finish_reason == "SAFETY":
                      error_message = "Sorry, my response generation was stopped due to safety concerns."
                 else:
                      error_message = f"Sorry, my response generation was interrupted ({finish_reason}). Please try again."

            log_msg = f"[ERROR: StopCandidateException - {error_message} (FinishReason: {finish_reason})]"
            self._log_interaction(user_input, log_msg)
            return error_message

        # --- Commented out GoogleAPIError handling for brevity, can be added back if needed ---
        # except GoogleAPIError as api_error:
        #     print(f"\nERROR: Google API Error: {api_error}")
        #     traceback.print_exc()
        #     error_message = f"Sorry, I encountered an API error ({type(api_error).__name__}). Please try again later."
        #     self._log_interaction(user_input, f"[ERROR: {error_message}]")
        #     return error_message

        except Exception as e:
            # Catch any other unexpected errors
            print(f"\nERROR: Unexpected chat error in chat_with_gemini: {type(e).__name__} - {e}")
            traceback.print_exc()
            error_message = f"Sorry, an unexpected internal error occurred ({type(e).__name__}). Please check the logs or try again."
            try:
                self._log_interaction(user_input, f"[ERROR: {error_message}]")
            except Exception as log_e:
                 print(f"Warning: Failed to log error interaction: {log_e}")
            return error_message


    def speak(self, output: str | None):
        """Prints Alpha's output to the console with a slight delay effect."""
        print("\nAlpha: ", end='')
        output_text = str(output) if output is not None else "[No response generated]"
        try:
            for char in output_text:
                print(char, end='', flush=True)
                time.sleep(0.005)
        except Exception as e:
            print(f"\n[Speak animation error: {e}]", end='')
            print(output_text)
        print("\n") # Ensure a newline


# --- Tool Functions (Decorated with @Alpha.add_func) ---

# === Personal Info Functions ===
@Alpha.add_func
def add_personal_info(label: str, value: str):
    """
    Adds or updates a piece of personal information using a label and value.
    label: string : The unique identifier or name for the piece of info (e.g., 'Name', 'Email', 'Favorite Color'). Cannot be empty.
    value: string : The actual information to store for the label. Cannot be empty.
    """
    # --- Input Validation ---
    if not label or not label.strip():
        return "Error: A non-empty label is required for personal info."
    label_clean = label.strip()
    value_str = str(value).strip()
    if not value_str:
        return "Error: A non-empty value is required for personal info."
    # --- End Validation ---

    data = _load_memory_data()
    personal_info_data = data.setdefault("PersonalInfo", {})
    if not isinstance(personal_info_data, dict):
        print(f"Warning: PersonalInfo data in {MEMORY_FILE} was not a dict. Resetting.")
        personal_info_data = {}
        data["PersonalInfo"] = personal_info_data

    original_value = personal_info_data.get(label_clean)
    personal_info_data[label_clean] = value_str
    print(f"DEBUG [Tool add_personal_info]: Added/Updated '{label_clean}': '{value_str}'")

    if _save_memory_data(data):
        return f"Okay, I've saved the personal info for '{label_clean}'."
    else:
        print(f"ERROR [Tool add_personal_info]: Failed to save personal info for '{label_clean}'. Rolling back in-memory change.")
        if original_value is not None:
            personal_info_data[label_clean] = original_value
        else:
            if label_clean in personal_info_data:
                 del personal_info_data[label_clean]
        return f"Error: Failed to save personal info for '{label_clean}' persistently. The change was not saved."

@Alpha.add_func
def get_personal_info(label: str = None):
    """
    Retrieves stored personal information, either for a specific label or all info.
    label: string [optional]: The specific label to retrieve information for. If omitted, returns all stored info.
    """
    data = _load_memory_data()
    personal_info = data.get("PersonalInfo", {})

    if not isinstance(personal_info, dict):
        print(f"Warning: PersonalInfo data in {MEMORY_FILE} is not a dictionary. Returning empty.")
        return "I don't seem to have any personal information stored correctly right now."

    if not personal_info:
        return "I don't have any personal information stored."

    if label:
        label_strip = label.strip()
        if not label_strip:
             return "Error: Please provide a specific label to search for, or ask for all personal info."
        if label_strip in personal_info:
            return f"Personal Info for '{label_strip}': {personal_info[label_strip]}"
        else:
            return f"Sorry, I couldn't find personal info for the label '{label_strip}'."
    else:
        if not personal_info:
             return "I don't have any personal information stored."
        details_list = [f"- {lbl}: {val}" for lbl, val in sorted(personal_info.items())]
        return "Here's the personal information I have stored:\n" + "\n".join(details_list)

@Alpha.add_func
def delete_personal_info(label: str):
    """
    Deletes a piece of personal information identified by its label.
    label: string : The exact label of the personal information to delete. Cannot be empty.
    """
    if not label or not label.strip():
        return "Error: A non-empty label is required to delete personal info."
    label_strip = label.strip()

    data = _load_memory_data()
    personal_info_data = data.get("PersonalInfo")

    if not isinstance(personal_info_data, dict):
        print(f"Warning: Cannot delete from PersonalInfo as it's not a dictionary in {MEMORY_FILE}.")
        return f"Error: Could not access personal info storage. Cannot delete '{label_strip}'."

    if label_strip not in personal_info_data:
        return f"Couldn't find personal info with the label '{label_strip}' to delete."

    deleted_value = personal_info_data.pop(label_strip)
    print(f"DEBUG [Tool delete_personal_info]: Deleted label '{label_strip}'.")

    if _save_memory_data(data):
        return f"Okay, I've deleted the personal info labeled '{label_strip}'."
    else:
        print(f"ERROR [Tool delete_personal_info]: Failed to save deletion of '{label_strip}'. Rolling back.")
        personal_info_data[label_strip] = deleted_value
        return f"Error: Failed to save the deletion of '{label_strip}'. Reverted the change in memory for this session."


# === Daily Schedule/Memory Functions ===

@Alpha.add_func
def add_daily_memory(date_input: str, label: str, value: str):
    """
    Adds or updates a labeled memory, task, or activity for a specific date. Use distinct labels for different items on the same day.
    date_input: string : The date for the memory (e.g., 'today', '2024-12-25', 'next friday'). Cannot be empty.
    label: string : A short category or title for the memory (e.g., 'Meeting', 'Lunch', 'Workout', 'Note'). Cannot be empty.
    value: string : The details or description of the memory/activity. Cannot be empty.
    """
    if not date_input or not date_input.strip():
        return "Error: A date input (like 'today', 'tomorrow', 'YYYY-MM-DD') is required."
    if not label or not label.strip():
        return "Error: A non-empty label (like 'Meeting', 'Note') is required for the memory."
    label_clean = label.strip()
    value_str = str(value).strip()
    if not value_str:
        return "Error: A non-empty value (description) is required for the memory."

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg:
        return f"Error processing date for adding memory: {error_msg}"
    if not date_str:
        return "Error: Could not determine a valid date from the input."

    data = _load_memory_data()
    schedule_data = data.setdefault("Schedule", {})
    if not isinstance(schedule_data, dict):
         print(f"Warning: Schedule data in {MEMORY_FILE} was not a dict. Resetting.")
         schedule_data = {}
         data["Schedule"] = schedule_data

    date_entry = schedule_data.setdefault(date_str, {})
    if not isinstance(date_entry, dict):
        print(f"Warning: Entry for date {date_str} in {MEMORY_FILE} was not a dict. Resetting.")
        date_entry = {}
        schedule_data[date_str] = date_entry

    original_value_for_label = date_entry.get(label_clean)
    date_entry[label_clean] = value_str
    print(f"DEBUG [Tool add_daily_memory]: Added/Updated for {date_str}: '{label_clean}': '{value_str}'")

    if _save_memory_data(data):
        return f"Okay, I've added/updated memory for {date_str} (from '{date_input}') with label '{label_clean}'."
    else:
        print(f"ERROR [Tool add_daily_memory]: Failed to save daily memory for {date_str}. Rolling back.")
        if original_value_for_label is not None:
            date_entry[label_clean] = original_value_for_label
        else:
            if label_clean in date_entry:
                 del date_entry[label_clean]
        if not date_entry and date_str in schedule_data:
             del schedule_data[date_str]
             print(f"DEBUG [Tool add_daily_memory]: Removed empty date entry {date_str} after rollback.")

        return f"Error: Failed to save daily memory for {date_str} persistently. Please try again."

@Alpha.add_func
def get_daily_memory(date_input: str, label: str = None):
    """
    Retrieves stored daily memories/activities for a specific date, optionally filtered by label.
    date_input: string : The date to retrieve memories for (e.g., 'today', '2024-12-25', 'yesterday'). Cannot be empty.
    label: string [optional]: The specific label to retrieve for the given date. If omitted, returns all entries for the date.
    """
    if not date_input or not date_input.strip():
        return "Error: A date input (like 'today', 'yesterday', 'YYYY-MM-DD') is required."
    label_strip = label.strip() if label else None

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg: return error_msg
    if not date_str: return "Error: Could not determine a valid date from the input."

    data = _load_memory_data()
    schedule_data = data.get("Schedule", {})

    if not isinstance(schedule_data, dict):
        print(f"Warning: Schedule data in {MEMORY_FILE} is not a dictionary.")
        return "Error: Could not access the schedule data storage."

    if date_str not in schedule_data:
        return f"I don't have any daily memories stored for {date_str} (from '{date_input}')."
    date_entry = schedule_data.get(date_str)
    if not isinstance(date_entry, dict):
         print(f"Warning: Entry for date {date_str} in {MEMORY_FILE} is not a dictionary. Treating as empty/invalid.")
         return f"The stored entry for {date_str} (from '{date_input}') seems corrupted. I can't retrieve details."

    if label_strip:
        if label_strip in date_entry:
            value_str = str(date_entry[label_strip])
            return f"On {date_str} (from '{date_input}'), the memory for '{label_strip}' is: {value_str}"
        else:
            return f"I found memories for {date_str} (from '{date_input}'), but not specifically for the label '{label_strip}'."
    else:
        if not date_entry:
            return f"I have an entry for {date_str} (from '{date_input}'), but no specific details are recorded for that day."

        details_list = [f"- {lbl}: {str(val)}" for lbl, val in sorted(date_entry.items())]
        return f"Memories for {date_str} (from '{date_input}'):\n" + "\n".join(details_list)

@Alpha.add_func
def list_memory_dates():
    """
    Lists all dates for which daily memories/schedule entries are stored. Returns dates in YYYY-MM-DD format.
    """
    data = _load_memory_data()
    schedule_data = data.get("Schedule", {})

    if not isinstance(schedule_data, dict):
        print(f"Warning: Schedule data in {MEMORY_FILE} is not a dictionary.")
        return "Error: Could not retrieve the list of dates due to invalid storage format."

    dates = sorted([d for d, entry in schedule_data.items() if isinstance(entry, dict) and entry])

    if not dates:
        return "I don't have daily memories stored for any dates yet."

    count = len(dates)
    limit = 25
    if count > limit:
        return (f"I have entries for {count} dates. Here are the first {limit}:\n"
                + "\n".join(dates[:limit]) + f"\n...and {count - limit} more.")
    else:
        return f"I have entries for the following {count} date(s):\n" + "\n".join(dates)

@Alpha.add_func
def delete_daily_memory(date_input: str, label: str = None):
    """
    Deletes daily memories: either a specific labeled entry or all entries for a given date.
    date_input: string : The date to delete memories from (e.g., 'today', '2024-12-25'). Cannot be empty.
    label: string [optional]: The specific label to delete for the given date. If omitted, deletes all entries for the date.
    """
    if not date_input or not date_input.strip():
        return "Error: A date input (like 'today', 'YYYY-MM-DD') is required."
    label_strip = label.strip() if label else None

    date_str, error_msg = _parse_and_standardize_date(date_input)
    if error_msg: return error_msg
    if not date_str: return "Error: Could not determine a valid date from the input."

    data = _load_memory_data()
    schedule_data = data.get("Schedule")

    if not isinstance(schedule_data, dict):
        print(f"Warning: Schedule data in {MEMORY_FILE} is not a dictionary.")
        return "Error: Could not access schedule data storage to perform deletion."
    if date_str not in schedule_data:
        return f"Couldn't find any daily memories stored for {date_str} (from '{date_input}') to delete."

    original_date_entry = schedule_data.get(date_str)
    original_date_entry_copy = {}
    if isinstance(original_date_entry, dict):
         original_date_entry_copy = original_date_entry.copy()
    else:
         original_date_entry_copy = original_date_entry

    if label_strip: # Delete specific label
        date_entry_dict = schedule_data.get(date_str)

        if not isinstance(date_entry_dict, dict):
            print(f"Warning: Entry for date {date_str} in {MEMORY_FILE} is not a dictionary. Cannot delete specific label '{label_strip}'.")
            return f"Error: Cannot delete label '{label_strip}' because the entry for {date_str} (from '{date_input}') is not structured correctly. You could try deleting the whole day's entry."

        if label_strip in date_entry_dict:
            deleted_value = date_entry_dict.pop(label_strip)
            print(f"DEBUG [Tool delete_daily_memory]: Deleted label '{label_strip}' for {date_str}.")

            if not date_entry_dict:
                del schedule_data[date_str]
                print(f"DEBUG [Tool delete_daily_memory]: Removed date entry for {date_str} as it became empty.")

            if _save_memory_data(data):
                return f"Okay, I've deleted the memory labeled '{label_strip}' for {date_str} (from '{date_input}')."
            else:
                print(f"ERROR [Tool delete_daily_memory]: Failed to save deletion of label '{label_strip}' for {date_str}. Rolling back.")
                schedule_data[date_str] = original_date_entry_copy
                return f"Error: Failed to save the deletion of label '{label_strip}'. Reverted the change in memory for this session."
        else:
            return f"I found memories for {date_str} (from '{date_input}'), but couldn't find the specific label '{label_strip}' to delete."

    else: # Delete entire entry for the date
        deleted_entry = schedule_data.pop(date_str)
        print(f"DEBUG [Tool delete_daily_memory]: Deleted all daily memories for {date_str}.")

        if _save_memory_data(data):
            return f"Okay, I've deleted all memories recorded for {date_str} (from '{date_input}')."
        else:
            print(f"ERROR [Tool delete_daily_memory]: Failed to save deletion of date {date_str}. Rolling back.")
            schedule_data[date_str] = deleted_entry
            return f"Error: Failed to save the deletion of all entries for {date_str}. Reverted the change in memory for this session."

@Alpha.add_func
def search_memories_by_content(search_term: str):
    """
    Searches through all daily memory labels and values for a given search term (case-insensitive).
    search_term: string : The text to search for within memory labels and values. Cannot be empty.
    """
    if not search_term or not search_term.strip():
        return "Error: Please provide a non-empty search term."
    search_term_lower = search_term.strip().lower()

    data = _load_memory_data()
    schedule_data = data.get("Schedule", {})

    if not isinstance(schedule_data, dict):
        print(f"Warning: Schedule data in {MEMORY_FILE} is not a dictionary.")
        return "Error: Could not access the schedule data to perform search."

    if not schedule_data:
        return "I don't have any daily memories stored to search through."

    matches = []
    for date_str, details in sorted(schedule_data.items()):
        if isinstance(details, dict):
            for label, value in details.items():
                value_str = str(value)
                if search_term_lower in label.lower() or search_term_lower in value_str.lower():
                    matches.append({
                        "date": date_str,
                        "label": label,
                        "value": value_str
                    })
        else:
            print(f"Warning [Tool search_memories]: Skipping entry for date {date_str} as it is not a dictionary.")


    if not matches:
        return f"Couldn't find any mentions of '{search_term}' in your daily memories."
    else:
        results_str = f"Found {len(matches)} mention(s) of '{search_term}' in daily memories:\n"
        limit = 15
        count = 0
        for match in matches:
             if count >= limit:
                 results_str += f"... ({len(matches) - count} more matches found)\n"
                 break
             truncated_value = match['value'][:100] + ('...' if len(match['value']) > 100 else '')
             results_str += f"- {match['date']} ({match['label']}): {truncated_value}\n"
             count += 1
        return results_str.strip()

@Alpha.add_func
def get_current_date_time():
    """
    Returns the current system date and time, including the day of the week.
    """
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S (%A)")
    print(f"DEBUG [Tool get_current_date_time]: Returning current time: {formatted_datetime}")
    return f"The current date and time is {formatted_datetime}"

# === New Reminder Functions ===

# --- !!! CORRECTED set_reminder Function !!! ---
@Alpha.add_func
def set_reminder(reminder_time_input: str, message: str):
    """
    Sets a reminder for a specific date and time with a message. The reminder will trigger proactively when due (notification managed externally).
    reminder_time_input: string : The date and time for the reminder (e.g., 'tonight 8pm', '2024-08-15 09:30', 'next tuesday at 2 PM'). Cannot be empty.
    message: string : The message content for the reminder. Cannot be empty.
    """
    # --- Input Validation ---
    if not reminder_time_input or not reminder_time_input.strip():
        return "Error: Please provide a time for the reminder (e.g., 'tomorrow 10 AM', 'July 4th 9:00')."
    if not message or not message.strip():
        return "Error: Please provide a non-empty message for the reminder."
    message_clean = message.strip()
    # --- End Validation ---

    print(f"DEBUG [Tool set_reminder]: Received reminder_time_input='{reminder_time_input}', message='{message_clean}'")

    reminder_dt = None # Initialize

    # --- Attempt Parsing (Try standard first, then fuzzy as fallback) ---
    try:
        print(f"DEBUG [set_reminder]: Attempting standard date_parser.parse('{reminder_time_input}', fuzzy=False)")
        reminder_dt = date_parser.parse(reminder_time_input, fuzzy=False, default=datetime.now())
        print(f"DEBUG [set_reminder]: Standard parsing successful. Result: {reminder_dt}")
    except (ParserError, ValueError, OverflowError, TypeError) as e_strict:
        print(f"WARN [set_reminder]: Standard parsing failed for '{reminder_time_input}'. Error: {e_strict}. Trying fuzzy parsing.")
        try:
            print(f"DEBUG [set_reminder]: Attempting fuzzy date_parser.parse('{reminder_time_input}', fuzzy=True)")
            reminder_dt = date_parser.parse(reminder_time_input, fuzzy=True, default=datetime.now())
            print(f"DEBUG [set_reminder]: Fuzzy parsing successful. Result: {reminder_dt}")
        except (ParserError, ValueError, OverflowError, TypeError) as e_fuzzy:
            print(f"ERROR [set_reminder]: Fuzzy parsing also failed for '{reminder_time_input}'. Error: {e_fuzzy}")
            return (f"Error: Could not understand the reminder time '{reminder_time_input}'. "
                    f"Please try a clearer format (e.g., 'tomorrow 9 AM', '2024-08-15 14:30'). "
                    f"Parser Error: {type(e_fuzzy).__name__}")
        except Exception as e_fuzzy_unexpected:
            print(f"ERROR [set_reminder]: Unexpected error during fuzzy parsing for '{reminder_time_input}'. Error: {e_fuzzy_unexpected}")
            traceback.print_exc()
            return f"Error: An unexpected error occurred parsing reminder time '{reminder_time_input}'. Details: {e_fuzzy_unexpected}"
    except Exception as e_unexpected:
        print(f"ERROR [set_reminder]: Unexpected error during standard parsing for '{reminder_time_input}'. Error: {e_unexpected}")
        traceback.print_exc()
        return f"Error: An unexpected error occurred parsing reminder time '{reminder_time_input}'. Details: {e_unexpected}"
    # --- End Parsing ---

    if reminder_dt is None:
         print(f"ERROR [set_reminder]: reminder_dt is None after parsing attempts for '{reminder_time_input}', indicating an unexpected logic flow.")
         return f"Error: Failed to determine a valid date/time from '{reminder_time_input}' after parsing attempts. Please try rephrasing."

    # --- Check if time is in the past ---
    now = datetime.now()
    warning_msg = ""
    if reminder_dt < (now - timedelta(minutes=1)):
        warning_msg = f" (Warning: The specified time {reminder_dt.strftime('%Y-%m-%d %H:%M')} appears to be in the past)."
        print(f"DEBUG [set_reminder]: {warning_msg.strip()}")
        # Optional: Disallow past reminders
        # return (f"Error: Reminder time '{reminder_dt.strftime('%Y-%m-%d %H:%M:%S')}' is in the past. "
        #         f"Please provide a future time.")

    # --- Prepare reminder data ---
    reminder_id = str(uuid.uuid4())
    # <<< FIX 1: Use strftime with the correct format string >>>
    reminder_time_str = reminder_dt.strftime('%Y-%m-%d %H:%M:%S')
    # <<< FIX 2: Use consistent format for creation time (Optional but good) >>>
    created_at_str = now.strftime('%Y-%m-%d %H:%M:%S')

    new_reminder = {
        "id": reminder_id,
        # <<< FIX 3: Use the correct key name "reminder_time" >>>
        "reminder_time": reminder_time_str,
        "message": message_clean,
        "status": "pending",
        # <<< FIX 4: Use consistent key "created_at" (Optional but good) >>>
        "created_at": created_at_str
    }

    # --- Load, update, and save reminders list ---
    data = _load_memory_data()
    reminders_list = data.setdefault("Reminders", [])
    if not isinstance(reminders_list, list):
        print(f"ERROR [set_reminder]: 'Reminders' section in {MEMORY_FILE} is not a list! Reinitializing.")
        reminders_list = []
        data["Reminders"] = reminders_list

    reminders_list.append(new_reminder)
    print(f"DEBUG [set_reminder]: Prepared to add reminder: {new_reminder}")

    # --- Save ---
    if _save_memory_data(data):
        friendly_time = reminder_dt.strftime('%Y-%m-%d %I:%M %p') # e.g., 2024-08-15 09:30 AM
        confirmation = f"Okay, reminder set for {friendly_time}: '{message_clean}'."
        confirmation += warning_msg
        return confirmation
    else:
        print(f"ERROR [set_reminder]: Failed to save new reminder (ID: {reminder_id}). Rolling back in-memory addition.")
        try:
            current_reminders = data.get("Reminders", [])
            data["Reminders"] = [r for r in current_reminders if isinstance(r, dict) and r.get("id") != reminder_id]
            print(f"DEBUG [set_reminder]: Rolled back addition of reminder {reminder_id} from memory.")
        except Exception as remove_err:
             print(f"ERROR: Exception occurred while trying to roll back reminder {reminder_id} from in-memory list: {remove_err}")
        return f"Error: Failed to save the reminder persistently. Please try setting it again."


# --- Reminder Checking Logic (for background thread - CLI ONLY) ---

# --- !!! CORRECTED check_reminders Function !!! ---
def check_reminders():
    """Checks pending reminders, prints notifications if due (CLI), and updates status."""
    # print("DEBUG: Checking reminders...") # Can be verbose
    now = datetime.now()
    data = None
    try:
        data = _load_memory_data() # Load fresh data each time
    except Exception as load_err:
         print(f"ERROR [Reminder Check]: Failed to load memory data: {load_err}")
         return

    reminders = data.get("Reminders", [])
    updated = False # Flag to track if any reminder statuses were changed

    if not isinstance(reminders, list):
        print(f"ERROR [Reminder Check]: Reminders data in {MEMORY_FILE} is corrupted (not a list). Skipping check.")
        return

    new_reminders_list = []
    triggered_count = 0

    for reminder in reminders:
        try:
            if not isinstance(reminder, dict):
                 print(f"Warning [Reminder Check]: Skipping non-dictionary item in Reminders list: {str(reminder)[:100]}")
                 new_reminders_list.append(reminder)
                 continue

            reminder_id = reminder.get('id', 'Unknown ID')
            current_status = reminder.get("status")
            # <<< FIX 5: Look for the correct key "reminder_time" >>>
            reminder_time_str = reminder.get("reminder_time")

            # Process only 'pending' reminders for triggering
            if current_status == "pending":
                if not reminder_time_str:
                    # <<< FIX 6: Update error message to mention "reminder_time" >>>
                    print(f"Warning [Reminder Check]: Pending reminder {reminder_id} has missing 'reminder_time'. Setting status to error.")
                    reminder["status"] = "error_missing_time"
                    updated = True
                    new_reminders_list.append(reminder)
                    continue

                try:
                    # <<< FIX 7: Parse using strptime with the correct format >>>
                    reminder_dt = datetime.strptime(reminder_time_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                     # <<< FIX 8: Update error message and status for format error >>>
                     print(f"ERROR [Reminder Check]: Invalid time format '{reminder_time_str}' for reminder {reminder_id}. Expected YYYY-MM-DD HH:MM:SS. Setting status to error.")
                     reminder["status"] = "error_invalid_time_format" # More specific status
                     updated = True
                     new_reminders_list.append(reminder)
                     continue

                # Check if reminder time is past or present
                if reminder_dt <= now:
                    # --- CLI Notification ---
                    print("\n" + "="*10 + " REMINDER " + "="*10)
                    friendly_time = reminder_dt.strftime('%Y-%m-%d %I:%M %p')
                    print(f"Time: {friendly_time}")
                    print(f"Message: {reminder.get('message', '(No message)')}")
                    print("="*30 + "\n")
                    # Update status to 'triggered'
                    reminder["status"] = "triggered"
                    updated = True
                    triggered_count += 1
                    new_reminders_list.append(reminder)

                else:
                    # Reminder is pending but not yet due, keep it as is
                    new_reminders_list.append(reminder)

            else:
                # Reminder is not pending, keep it as is
                new_reminders_list.append(reminder)

        except Exception as e:
             r_id = "Unknown ID"
             if isinstance(reminder, dict): r_id = reminder.get('id', r_id)
             print(f"ERROR [Reminder Check]: Unexpected error processing reminder {r_id}: {e}")
             traceback.print_exc()
             if isinstance(reminder, dict):
                  reminder["status"] = "error_processing"
                  updated = True
             new_reminders_list.append(reminder)


    # If any statuses were updated, replace the list in the main data and save
    if updated:
        print(f"DEBUG [Reminder Check]: {triggered_count} reminder(s) triggered. Updating statuses in memory file.")
        data["Reminders"] = new_reminders_list # Replace the old list
        if not _save_memory_data(data):
            print(f"ERROR [Reminder Check]: Failed to save updated reminder statuses to {MEMORY_FILE}!")
        else:
            print("DEBUG [Reminder Check]: Reminder statuses saved successfully.")


def run_scheduler():
    """Sets up and runs the reminder check schedule loop (CLI)."""
    print("DEBUG: Reminder scheduler thread started. Checking every 60 seconds.")
    schedule.every(60).seconds.do(check_reminders)
    stop_event = threading.Event()

    def shutdown_gracefully():
        print("DEBUG: Scheduler received shutdown signal.")
        stop_event.set()

    while not stop_event.is_set():
        try:
            schedule.run_pending()
            stop_event.wait(1)
        except KeyboardInterrupt:
             print("DEBUG: Scheduler thread interrupted (KeyboardInterrupt). Signaling stop.")
             shutdown_gracefully()
             break
        except Exception as e:
            print(f"ERROR in scheduler loop: {e}")
            traceback.print_exc()
            stop_event.wait(10)

    print("DEBUG: Reminder scheduler thread finished.")


# --- Main Execution (CLI) ---
if __name__ == "__main__":
    print("\n--- Alpha Assistant (CLI Mode - v3 Reminders) ---")
    print("Initializing...")

    # --- Initial Memory File Check and Initialization ---
    initial_data = None
    memory_file_ok = False
    try:
         initial_data = _load_memory_data()
         if os.path.exists(MEMORY_FILE) and \
            isinstance(initial_data, dict) and \
            "Reminders" in initial_data and \
            isinstance(initial_data.get("Reminders"), list):
             print(f"DEBUG: Memory file {MEMORY_FILE} loaded successfully.")
             memory_file_ok = True
         else:
             print(f"DEBUG: Memory file {MEMORY_FILE} missing or invalid. Attempting to initialize/fix...")
             data_to_save = initial_data if isinstance(initial_data, dict) else DEFAULT_MEMORY_STRUCTURE.copy()
             if "Reminders" not in data_to_save or not isinstance(data_to_save.get("Reminders"), list):
                 data_to_save["Reminders"] = []

             if _save_memory_data(data_to_save):
                  print(f"DEBUG: Successfully initialized/saved default structure to {MEMORY_FILE}.")
                  memory_file_ok = True
             else:
                  print(f"FATAL ERROR: Could not write initial/default structure to {MEMORY_FILE}. Cannot continue.")
                  sys.exit(1)

    except Exception as load_err:
         print(f"FATAL ERROR: Could not load or initialize memory file {MEMORY_FILE}: {load_err}")
         traceback.print_exc()
         sys.exit(1)

    if not memory_file_ok:
        print(f"FATAL ERROR: Memory file {MEMORY_FILE} check failed. Exiting.")
        sys.exit(1)
    # --- End Memory File Check ---

    ai_instance = None
    scheduler_thread = None
    try:
        ai_instance = Alpha()

        # --- Start Reminder Scheduler Thread (CLI ONLY) ---
        print("Starting reminder scheduler background thread...")
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True, name="ReminderScheduler")
        scheduler_thread.start()

        time.sleep(0.5)
        if not scheduler_thread.is_alive():
             print("WARNING: Reminder scheduler thread failed to start or exited immediately. Reminders might not trigger.")
        else:
             print("Reminder scheduler is running.")
        # --- End Scheduler Start ---

        print("-" * 30)
        print(f"Using model: {MODEL_NAME}")
        print(f"Memory file: {MEMORY_FILE}")
        print(f"History file: {HISTORY_FILE}")
        registered_tool_names = list(Alpha.registered_functions.keys())
        print(f"Registered tools ({len(registered_tool_names)}): {', '.join(sorted(registered_tool_names))}")
        print("-" * 30 + "\nAlpha is ready. Type 'exit' or 'quit' to end.\n" + "-" * 30)

    except RuntimeError as init_error:
        print(f"\nFATAL ERROR during Alpha initialization: {init_error}")
        sys.exit(1)
    except Exception as init_error:
        print(f"\nFATAL ERROR during setup before main loop: {init_error}")
        traceback.print_exc()
        sys.exit(1)

    # --- Main Chat Loop (CLI Mode) ---
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                 print("\n\nAlpha: End of input detected. Shutting down...")
                 break
            except KeyboardInterrupt:
                print("\n\nAlpha: Detected interrupt (Ctrl+C). Shutting down...")
                break

            if not user_input: continue

            if user_input.lower() in ["exit", "quit", "goodbye", "bye", "stop", "q", "0"]:
                print("\nAlpha: Okay, shutting down...")
                break

            if ai_instance and ai_instance.chat:
                output = ai_instance.chat_with_gemini(user_input)
                ai_instance.speak(output)
            else:
                print("Alpha: [FATAL ERROR: Assistant instance or chat session is not available. Cannot process.]")
                break

    except KeyboardInterrupt:
        print("\n\nAlpha: Detected interrupt (Ctrl+C). Shutting down...")
    except Exception as main_loop_error:
        print(f"\nFATAL ERROR during main chat loop: {main_loop_error}")
        traceback.print_exc()

    finally:
        # Cleanup and final message
        print("\n" + "-" * 30 + "\nAlpha: Shutting down processes...")
        if scheduler_thread and scheduler_thread.is_alive():
             print("DEBUG: Reminder scheduler thread (daemon) will exit with the main program.")

        if ai_instance:
            try: ai_instance.speak("Goodbye!")
            except Exception: print("Alpha: Goodbye!")
        else: print("Alpha: Goodbye!")

        print(f"Chat history saved in: {HISTORY_FILE}")
        print(f"Memory data stored in: {MEMORY_FILE}")
        print("-" * 30)
        time.sleep(0.5)
        print("Exiting.")