"""
Alpha : My first Personal Assistant
Adapted for: Google Gemini (e.g., gemini-1.5-flash-latest)
Using: Python and Google Generative AI APIs
Handles memories in a date-keyed dictionary format.
"""

"""
Requirements:
Python-(winget install python) for windows
Google Generative AI Library-(pip install google-generativeai)
python-dotenv-(pip install python-dotenv)
"""
import sys
import google.generativeai as genai
# Importing necessary types, EXCLUDING Part
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Tool, FunctionDeclaration # Need Content for history

import json
from dotenv import load_dotenv
import os
import time
from datetime import datetime
# from google.protobuf.struct_pb2 import Struct # Use this if complex arguments needed
import re # Import regular expression module for checking date format

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("FATAL ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please create a .env file with GOOGLE_API_KEY=YOUR_API_KEY")
    sys.exit(1) # Exit if key is missing

genai.configure(api_key=GOOGLE_API_KEY)

# --- Select your model! ---
# Options: "gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro"
MODEL_NAME = "gemini-1.5-flash-latest"

# Safety settings - Adjust as needed
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- Files ---
MEMORY_FILE = "alpha_memory_daily.json" # Renamed for clarity of format
HISTORY_FILE = "alpha_chat_history.log" # For logging interactions

# --- Helper Functions for Daily Memory Format ---

def _load_daily_memories(filename=MEMORY_FILE):
    """Loads the date-keyed dictionary from the JSON file."""
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return {}
            data = json.loads(content)
            # Ensure it's a dictionary where keys *look like* dates (basic check)
            if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
                 # Basic check if keys resemble YYYY-MM-DD format
                 if all(re.match(r'^\d{4}-\d{2}-\d{2}$', k) for k in data.keys()):
                     return data
                 else:
                     print(f"Warning: Keys in {filename} do not all match YYYY-MM-DD format. Returning empty.")
                     return {}
            else:
                 print(f"Warning: Data in {filename} is not a dictionary of dictionaries. Returning empty.")
                 return {}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filename}. Returning empty.")
        return {}
    except Exception as e:
        print(f"An error occurred loading data from {filename}: {e}. Returning empty.")
        return {}

def _save_daily_memories(data, filename=MEMORY_FILE):
    """Saves the date-keyed dictionary to the JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Sort keys (dates) for consistent order
            json.dump(data, f, indent=4, sort_keys=True)
        return True
    except Exception as e:
        print(f"ERROR: Failed to write daily memories to {filename}: {e}")
        return False

def _validate_date_format(date_str):
    """Checks if a string is in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# --- Alpha Class ---
class Alpha:
    # Class variables to store tool definitions and function mappings
    tool_declarations = [] # Will hold FunctionDeclaration objects
    registered_functions = {} # Maps function names to actual Python functions

    def __init__(self):
        # self.history = [] # ChatSession handles history internally
        self.system_prompt = (
            "You are Alpha, my personal assistant. "
            "Be helpful, concise, and friendly. "
            "Use the available tools (functions) when appropriate to manage daily memories or get information like the date/time. "
            "Memories are stored by date (YYYY-MM-DD) and contain labeled details (e.g., 'Dinner': 'pizza'). "
            "Use 'add_daily_memory' to save details for a date. "
            "Use 'get_daily_memory' to retrieve details for a date (optionally by label). "
            "Use 'list_memory_dates' to see which dates have entries. "
            "Use 'delete_daily_memory' to remove details or entire dates."
        )
        # Initialize the Gemini model
        try:
            self.model = genai.GenerativeModel(
                MODEL_NAME,
                safety_settings=SAFETY_SETTINGS,
                system_instruction=self.system_prompt,
                # generation_config={"response_mime_type": "text/plain"} # Optional: Force text only if needed
            )
            # Start a chat session which maintains history automatically
            self.chat = self.model.start_chat(history=[])
            print(f"DEBUG: Chat session started with model {MODEL_NAME}.")
        except Exception as e:
            print(f"FATAL ERROR: Could not initialize GenerativeModel or ChatSession: {e}")
            print("Check your API key, model name, and network connection.")
            sys.exit(1)

    def _log_interaction(self, user_input, alpha_response):
        """Appends the user input and Alpha's response to the history log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} | You: {user_input}\n")
                f.write(f"{timestamp} | Alpha: {alpha_response}\n")
                f.write("-" * 20 + "\n") # Separator
        except Exception as e:
            print(f"\nWarning: Could not write to history file {HISTORY_FILE}: {e}")

    # REMOVED: list_memories (replaced by list_memory_dates tool)

    @classmethod
    def _create_function_declaration(cls, func):
        """Helper to create Gemini FunctionDeclaration from docstring."""
        if not func.__doc__:
             print(f"Warning: Function '{func.__name__}' has no docstring. Cannot generate API declaration.")
             return None

        doc_lines = [line.strip() for line in func.__doc__.strip().split('\n') if line.strip()]
        if not doc_lines:
            print(f"Warning: Function '{func.__name__}' has an empty docstring.")
            return None

        description = doc_lines[0]
        parameters_schema = {'type': 'object', 'properties': {}, 'required': []}

        # Simple parameter parsing (assumes "param_name: type: description" format)
        if len(doc_lines) > 1:
            for line in doc_lines[1:]:
                if ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) == 3:
                        param_name = parts[0].strip()
                        param_type_desc = parts[1].strip().lower()
                        param_desc = parts[2].strip()

                        # Basic type mapping
                        param_type = 'string' # Default
                        if 'int' in param_type_desc or 'integer' in param_type_desc: param_type = 'integer'
                        elif 'float' in param_type_desc or 'number' in param_type_desc: param_type = 'number'
                        elif 'bool' in param_type_desc or 'boolean' in param_type_desc: param_type = 'boolean'
                        elif 'list' in param_type_desc or 'array' in param_type_desc: param_type = 'array'

                        parameters_schema['properties'][param_name] = {'type': param_type, 'description': param_desc}
                        # Make parameter required unless explicitly marked [optional]
                        if '[optional]' not in param_desc.lower():
                             parameters_schema['required'].append(param_name)
                    # else: # Debugging line if needed
                    #     print(f"DEBUG: Skipping line in docstring for {func.__name__} (doesn't match param format): {line}")

        # Finalize schema
        if not parameters_schema['properties']:
             print(f"DEBUG: Function {func.__name__} declared with no parameters.")
             return FunctionDeclaration(name=func.__name__, description=description)
        else:
            # Remove 'required' if it's empty (all parameters were optional)
            if not parameters_schema['required']:
                del parameters_schema['required']
            print(f"DEBUG: Function {func.__name__} declared with parameters: {parameters_schema}")
            return FunctionDeclaration(name=func.__name__, description=description, parameters=parameters_schema)


    @classmethod
    def add_func(cls, func):
        """Decorator to register functions and create their Gemini tool declarations."""
        declaration = cls._create_function_declaration(func)
        if declaration:
            cls.registered_functions[func.__name__] = func
            cls.tool_declarations.append(declaration)
            print(f"DEBUG: Registered function '{func.__name__}' as a tool.")
        else:
             print(f"DEBUG: Failed to register function '{func.__name__}' (no valid declaration).")
        return func

    def _prepare_tools(self):
        """Returns the Tool object for the API call, if tools are declared."""
        if self.tool_declarations:
             valid_declarations = [decl for decl in self.tool_declarations if decl is not None]
             if valid_declarations:
                return Tool(function_declarations=valid_declarations)
        return None # Return None if no tools are available

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
                 print(f"Warning: Could not convert function arguments for {function_name} directly to dict: {e}. Trying iteration.")
                 try:
                     args_dict = {key: value for key, value in args.items()}
                     print(f"DEBUG: Calling function (via iteration): {function_name} with args: {args_dict}")
                 except Exception as e2:
                     print(f"ERROR: Failed to extract arguments for function {function_name}: {e2}. Calling with no args.")
                     args_dict = {}

            # Execute the function
            try:
                # Call the registered function (which is now bound to the instance 'self')
                # or remains a static method/regular function if defined outside the class initially
                # Check if the function expects 'self' (is it a method of Alpha?)
                # In this setup, the decorated functions are module-level but registered in the class.
                # They don't automatically get 'self'. If they needed instance state,
                # they'd need to be defined *inside* the class and potentially take `self`.
                # Since our new memory functions don't need 'self', this is fine.
                function_response = function_to_call(**args_dict)
                print(f"DEBUG: Function {function_name} returned: {type(function_response)} -> {str(function_response)[:100]}...")
                return function_name, function_response # Return name and result

            except TypeError as te:
                 print(f"ERROR: TypeError executing function '{function_name}' with args {args_dict}: {te}")
                 error_message = f"Error: Called function '{function_name}' with incompatible arguments. {te}"
                 return function_name, error_message
            except Exception as e:
                print(f"ERROR: Exception executing function '{function_name}' with args {args_dict}: {e}")
                error_message = f"Error executing function '{function_name}': {e}"
                return function_name, error_message
        else:
            print(f"Warning: Model tried to call unknown function: {function_name}")
            error_message = f"Error: Function '{function_name}' is not registered or available."
            return function_name, error_message

    def _prepare_function_response_dict(self, function_name, function_response_content):
         """Creates the dictionary needed for send_message to represent a function result."""
         if not isinstance(function_response_content, str):
             try:
                 serialized_content = json.dumps(function_response_content)
             except TypeError:
                 serialized_content = str(function_response_content)
         else:
             serialized_content = function_response_content

         response_dict = {
             "function_response": {
                 "name": function_name,
                 "response": {
                     "content": serialized_content,
                 }
             }
         }
         print(f"DEBUG: Prepared function response dictionary: {response_dict}")
         return response_dict


    def chat_with_gemini(self, user_input):
        """Sends user input to Gemini, handles history and function calls."""
        # 1. Prepare context (optional: list available dates if helpful)
        #    Can be computationally expensive if many dates exist.
        #    Consider only adding context if explicitly asked or if memory seems relevant.
        # memory_dates = list(_load_daily_memories().keys()) # Get dates from file directly
        # context_str = ""
        # if memory_dates:
        #     max_dates_display = 10
        #     displayed_dates = sorted(memory_dates, reverse=True)[:max_dates_display] # Show recent dates
        #     context_str = ', '.join(displayed_dates)
        #     if len(memory_dates) > max_dates_display:
        #         context_str += f", ... ({len(memory_dates) - max_dates_display} more)"
        #     context_str = f"\n(Hint: Dates with memories include: {context_str})"
        # else:
        #      context_str = "\n(Hint: No memory dates found yet.)"

        # Simplified: Let the user ask via list_memory_dates if needed
        prompt = f"{user_input}"
        # print(f"DEBUG: Sending prompt: {prompt}")

        # 2. Prepare tools for the API call
        current_tools = self._prepare_tools()
        tools_list = [current_tools] if current_tools else None

        try:
            # 3. Send the user message
            response = self.chat.send_message(prompt, tools=tools_list)

            # 4. Loop to handle potential function calls
            while True:
                function_call = None
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            function_call = part.function_call
                            print(f"DEBUG: Model requested function call: {function_call.name}")
                            break

                if function_call:
                    # 5. Execute the requested function
                    function_name, function_result = self._execute_function_call(function_call)

                    # 6. Prepare the response *dictionary*
                    function_response_dict = self._prepare_function_response_dict(
                        function_name, function_result
                    )

                    # 7. Send the function result back to the model
                    print(f"DEBUG: Sending function response for '{function_name}' back to model.")
                    response = self.chat.send_message(function_response_dict) # Send dictionary

                else:
                    # 8. If no function call, break the loop
                    break

            # 9. Extract the final text response
            final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')) if response.candidates else ""
            if not final_text and hasattr(response, 'text'): # Fallback
                 final_text = response.text

            if not final_text:
                 print("Warning: Received an empty final response from the model.")
                 # Log details for debugging
                 if response.prompt_feedback: print(f"Prompt Feedback: {response.prompt_feedback}")
                 if response.candidates and response.candidates[0].finish_reason: print(f"Finish Reason: {response.candidates[0].finish_reason}")
                 if response.candidates and response.candidates[0].safety_ratings: print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
                 return "[Alpha had no text response. Check logs for potential issues like safety blocking.]"

            self._log_interaction(user_input, final_text)
            return final_text

        except Exception as e:
            print(f"\nERROR: An unexpected error occurred during chat interaction: {type(e).__name__} - {e}")
            try: # Log extra details if possible
                if 'response' in locals() and response:
                    if response.prompt_feedback: print(f"Prompt Feedback: {response.prompt_feedback}")
                    if response.candidates:
                         if response.candidates[0].finish_reason: print(f"Finish Reason: {response.candidates[0].finish_reason}")
                         if response.candidates[0].safety_ratings: print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
            except Exception as inner_e: print(f"ERROR: Could not retrieve details after main error: {inner_e}")

            error_message = f"Sorry, an unexpected error occurred ({type(e).__name__}). Please check the console logs."
            self._log_interaction(user_input, f"[ERROR OCCURRED: {error_message}]")
            return error_message


    def speak(self, output):
        """Prints the output character by character."""
        print("\nAlpha: ", end='')
        output_text = output if output else "[No response received]"
        try:
            for char in output_text:
                print(char, end='', flush=True)
                time.sleep(0.025)
        except Exception as e:
            print(f"[Error during speak animation: {e}]", end='')
        print("\n")


# --- Tool Functions (NEW Daily Memory Management & Date/Time) ---

@Alpha.add_func
def add_daily_memory(date_str: str, label: str, value: str):
    """
    Save or update a specific labeled detail for a given date (YYYY-MM-DD).
    date_str: string: The date for the memory in YYYY-MM-DD format.
    label: string: The name or key for this piece of information (e.g., 'Dinner', 'Meeting Topic').
    value: string: The content or value associated with the label.
    """
    if not _validate_date_format(date_str):
        return f"Error: Invalid date format '{date_str}'. Please use YYYY-MM-DD."
    if not label.strip():
        return "Error: Memory label cannot be empty."

    data = _load_daily_memories()

    # Get the dictionary for the date, creating it if it doesn't exist
    date_entry = data.setdefault(date_str, {})

    # Add or update the label-value pair
    date_entry[label] = value
    print(f"DEBUG: Added/Updated memory for {date_str}: '{label}': '{value}'")

    # Save the updated data
    if _save_daily_memories(data):
        return f"Memory for {date_str} updated: '{label}' set to '{value}'."
    else:
        return f"Error: Failed to save memory update for {date_str}."

@Alpha.add_func
def get_daily_memory(date_str: str, label: str = None):
    """
    Retrieve memory details for a specific date (YYYY-MM-DD). Can get all details or just one specific labeled item.
    date_str: string: The date to retrieve memories for (YYYY-MM-DD).
    label: string [optional]: The specific label to retrieve. If omitted, returns all details for the date.
    """
    if not _validate_date_format(date_str):
        return f"Error: Invalid date format '{date_str}'. Please use YYYY-MM-DD."

    data = _load_daily_memories()

    if date_str not in data:
        return f"No memories found for the date {date_str}."

    date_entry = data[date_str]

    if label:
        # Retrieve a specific label
        label_strip = label.strip()
        if label_strip in date_entry:
            return f"For {date_str}, the value of '{label_strip}' is: {date_entry[label_strip]}"
        else:
            return f"For {date_str}, the label '{label_strip}' was not found."
    else:
        # Retrieve all details for the date
        if not date_entry:
             return f"No specific details found for {date_str} (entry exists but is empty)."

        details_list = [f"- {lbl}: {val}" for lbl, val in date_entry.items()]
        return f"Memories for {date_str}:\n" + "\n".join(details_list)

@Alpha.add_func
def list_memory_dates():
    """
    List all dates (YYYY-MM-DD) that currently have memory entries stored.
    """
    data = _load_daily_memories()
    dates = sorted(list(data.keys())) # Sort dates chronologically

    if not dates:
        return "No dates with memories found."
    else:
        return "Dates with memory entries:\n" + "\n".join(dates)


@Alpha.add_func
def delete_daily_memory(date_str: str, label: str = None):
    """
    Delete memory details for a specific date (YYYY-MM-DD). Can delete a specific label or the entire date entry.
    date_str: string: The date to delete memories from (YYYY-MM-DD).
    label: string [optional]: The specific label to delete. If omitted, deletes all entries for the entire date.
    """
    if not _validate_date_format(date_str):
        return f"Error: Invalid date format '{date_str}'. Please use YYYY-MM-DD."

    data = _load_daily_memories()

    if date_str not in data:
        return f"No memories found for the date {date_str} to delete."

    if label:
        # Delete a specific label
        label_strip = label.strip()
        if label_strip in data[date_str]:
            del data[date_str][label_strip]
            print(f"DEBUG: Deleted label '{label_strip}' for date {date_str}.")
            # If the date entry becomes empty after deletion, remove the date key itself
            if not data[date_str]:
                del data[date_str]
                print(f"DEBUG: Date entry {date_str} became empty and was removed.")
            if _save_daily_memories(data):
                 return f"Deleted label '{label_strip}' for date {date_str}."
            else:
                 # Attempt to revert change in memory (might be complex if file error persists)
                 return f"Error: Failed to save deletion of label '{label_strip}' for {date_str}."

        else:
            return f"Label '{label_strip}' not found for date {date_str}. Nothing deleted."
    else:
        # Delete the entire entry for the date
        del data[date_str]
        print(f"DEBUG: Deleted all entries for date {date_str}.")
        if _save_daily_memories(data):
            return f"Deleted all memory entries for date {date_str}."
        else:
            return f"Error: Failed to save deletion of date {date_str}."


@Alpha.add_func
def get_current_date_time():
    """
    Get the current local date and time, formatted for readability.
    """
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"The current date and time is {formatted_datetime}"


# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Alpha Assistant (Gemini Edition - Daily Memory) ---")
    print("Initializing...")

    try:
        # Initialize Alpha
        ai = Alpha()
        print(f"Using model: {MODEL_NAME}")
        print(f"Memory file: {MEMORY_FILE}")

        # List registered tools
        registered_tool_names = list(ai.registered_functions.keys())
        if registered_tool_names:
             print(f"Registered tools: {', '.join(registered_tool_names)}")
        else:
             print("No tools registered.")

        print("-" * 30)
        print("Alpha is ready. Type your prompts below.")
        print("Type 'exit' or 'quit' to end the session.")
        print("-" * 30)

    except Exception as init_error:
        print(f"\nFATAL ERROR during Alpha initialization: {init_error}")
        sys.exit(1)

    # Main interaction loop
    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "0"]:
                ai.speak("Goodbye!")
                break

            output = ai.chat_with_gemini(user_input)
            ai.speak(output)

    except KeyboardInterrupt:
        print("\n\nAlpha: Detected interrupt. Shutting down...")
        ai.speak("Okay, exiting now.")
    except EOFError:
         print("\n\nAlpha: End of input detected. Shutting down...")
         ai.speak("Goodbye!")
    except Exception as main_loop_error:
        print(f"\nFATAL ERROR during main loop: {main_loop_error}")
        try: ai.speak("A critical error occurred. I need to shut down.")
        except: pass

    finally:
        print("\n" + "-" * 30)
        print("Alpha: Session ended.")
        print(f"Conversation history saved to: {HISTORY_FILE}")
        print(f"Memories managed in: {MEMORY_FILE}")
        print("-" * 30)