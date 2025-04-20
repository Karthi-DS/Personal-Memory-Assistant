"""
Alpha : My first Personal Assistant
Adapted for: Google Gemini (gemini-1.5-flash-latest)
Using: Python and Google Generative AI APIs
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
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Tool, FunctionDeclaration

import json
from dotenv import load_dotenv
import os
import time
# from google.protobuf.struct_pb2 import Struct # Use this if complex arguments needed

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please create a .env file with GOOGLE_API_KEY=YOUR_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Select your model!
MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Or "gemini-1.5-pro-latest", "gemini-pro", etc.

# Safety settings - Adjust as needed
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- Memory File ---
MEMORY_FILE = "memory.json"

# --- Alpha Class ---
class Alpha:
    tool_declarations = [] # Will hold FunctionDeclaration objects
    registered_functions = {} # Maps function names to actual Python functions

    def __init__(self):
        self.history = [] # Manages conversation history (though Gemini chat handles it internally)
        self.system_prompt = "You are Alpha, my personal assistant. Be helpful and concise."
        # Initialize the Gemini model
        self.model = genai.GenerativeModel(
            MODEL_NAME,
            safety_settings=SAFETY_SETTINGS,
            system_instruction=self.system_prompt,
        )
        # Start a chat session which maintains history automatically
        self.chat = self.model.start_chat(history=[]) # Start with empty history

    def list_memories(self):
        """Loads memory titles from the JSON file."""
        if not os.path.isfile(MEMORY_FILE):
            return []
        try:
            with open(MEMORY_FILE, 'r') as f:
                content = f.read()
                if not content: return []
                data = json.loads(content)
            if isinstance(data, list):
                return [memory.get('title', 'Untitled') for memory in data if isinstance(memory, dict)]
            else:
                print(f"Warning: {MEMORY_FILE} does not contain a list.")
                return []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {MEMORY_FILE}. Is it empty or malformed?")
            return []
        except Exception as e:
            print(f"Error reading memories: {e}")
            return []

    @classmethod
    def _create_function_declaration(cls, func):
        """Helper to create Gemini FunctionDeclaration from docstring."""
        if not func.__doc__:
             print(f"Warning: Function '{func.__name__}' has no docstring. Cannot generate declaration.")
             return None
        doc_lines = func.__doc__.strip().split('\n')
        description = doc_lines[0].strip()
        parameters_schema = {'type': 'object', 'properties': {}, 'required': []}
        if len(doc_lines) > 1:
            for line in doc_lines[1:]:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) == 3:
                        param_name = parts[0].strip()
                        param_type_desc = parts[1].strip().lower()
                        param_desc = parts[2].strip()
                        param_type = 'string' # Default
                        if 'int' in param_type_desc or 'integer' in param_type_desc: param_type = 'integer'
                        elif 'float' in param_type_desc or 'number' in param_type_desc: param_type = 'number'
                        elif 'bool' in param_type_desc: param_type = 'boolean'
                        elif 'list' in param_type_desc or 'array' in param_type_desc: param_type = 'array'
                        parameters_schema['properties'][param_name] = {'type': param_type, 'description': param_desc}
                        if '[optional]' not in param_desc.lower():
                             parameters_schema['required'].append(param_name)
        if parameters_schema['properties']:
            if not parameters_schema['required']: del parameters_schema['required']
            return FunctionDeclaration(name=func.__name__, description=description, parameters=parameters_schema)
        else:
            return FunctionDeclaration(name=func.__name__, description=description)

    @classmethod
    def add_func(cls, func):
        """Decorator to register functions and create their Gemini tool declarations."""
        declaration = cls._create_function_declaration(func)
        if declaration:
            cls.registered_functions[func.__name__] = func
            cls.tool_declarations.append(declaration)
        return func

    def _prepare_tools(self):
        """Returns the Tool object for the API call, if tools are declared."""
        if self.tool_declarations:
             valid_declarations = [decl for decl in self.tool_declarations if decl is not None]
             if valid_declarations:
                return Tool(function_declarations=valid_declarations)
        return None

    # --- MODIFIED: Does NOT use Part ---
    def _handle_function_call(self, function_call):
        """Executes a function call requested by the model and returns the result dictionary."""
        function_name = function_call.name
        args = function_call.args

        try:
            args_dict = dict(args)
        except Exception as e:
            print(f"Warning: Could not convert function arguments for {function_name} to dict: {e}")
            args_dict = {}

        if function_name in self.registered_functions:
            function_to_call = self.registered_functions[function_name]
            print(f"DEBUG: Calling function: {function_name} with args: {args_dict}")
            try:
                function_response = function_to_call(**args_dict)
                print(f"DEBUG: Function {function_name} returned: {function_response}")

                # Ensure response is serializable (string is safest)
                if not isinstance(function_response, str):
                    try:
                        function_response_content = json.dumps(function_response)
                    except TypeError:
                         function_response_content = str(function_response) # Fallback to string
                else:
                    function_response_content = function_response

                # --- CONSTRUCT DICTIONARY (Instead of Part.from_function_response) ---
                return {
                    "function_response": {
                        "name": function_name,
                        "response": {
                            "content": function_response_content
                        }
                    }
                }
                # --- END DICTIONARY ---

            except Exception as e:
                print(f"Error executing function '{function_name}' with args {args_dict}: {e}")
                error_content = f"Error executing function: {e}"
                # --- CONSTRUCT ERROR DICTIONARY ---
                return {
                    "function_response": {
                        "name": function_name,
                        "response": {
                            "content": error_content
                        }
                    }
                }
                # --- END ERROR DICTIONARY ---
        else:
            print(f"Warning: Model tried to call unknown function: {function_name}")
            error_content = f"Error: Function '{function_name}' is not registered or available."
            # --- CONSTRUCT UNKNOWN FUNCTION DICTIONARY ---
            return {
                 "function_response": {
                    "name": function_name,
                    "response": {
                        "content": error_content
                    }
                }
            }
            # --- END UNKNOWN FUNCTION DICTIONARY ---

    def chat_with_gemini(self, user_input):
        """Sends user input to Gemini, handles history and function calls."""
        memory_titles = self.list_memories()
        memory_context = ""
        if memory_titles:
            max_titles = 10
            displayed_titles = memory_titles[:max_titles]
            context_str = ', '.join(displayed_titles)
            if len(memory_titles) > max_titles:
                context_str += f", ... ({len(memory_titles) - max_titles} more)"
            memory_context = f"\n(Available memory titles: {context_str})"

        prompt = f"{user_input}{memory_context}"
        current_tools = self._prepare_tools()
        tools_list = [current_tools] if current_tools else None

        try:
            # Send initial prompt
            response = self.chat.send_message(prompt, tools=tools_list)

            # Check for function call request
            function_call = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    # Check if the part IS a function call (not just contains one)
                    if hasattr(part, 'function_call') and part.function_call:
                         function_call = part.function_call
                         break

            # Handle function calls if requested
            while function_call:
                print(f"DEBUG: Model requested function call: {function_call.name}")

                # Execute the function and get the result DICTIONARY
                function_response_dict = self._handle_function_call(function_call)

                print(f"DEBUG: Sending function response dict: {function_response_dict}")

                # Send the function response dictionary back to the model
                # The send_message method should handle this dictionary correctly
                # for the function result turn.
                response = self.chat.send_message(
                    function_response_dict,
                    # tools=None # Don't send tools when responding to function call
                )

                # Check if the *new* response asks for *another* function call
                function_call = None
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                         if hasattr(part, 'function_call') and part.function_call:
                             function_call = part.function_call
                             break

            # Extract final text response after all function calls (if any)
            final_text = ""
            if response.candidates and response.candidates[0].content.parts:
                 final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            if not final_text and hasattr(response, 'text'): # Fallback
                 final_text = response.text

            return final_text

        except Exception as e:
            print(f"ERROR: An error occurred during chat interaction: {e}")
            # Attempt to log additional details if possible
            try:
                if 'response' in locals() and response:
                    if response.prompt_feedback: print(f"Prompt Feedback: {response.prompt_feedback}")
                    if response.candidates and response.candidates[0].finish_reason: print(f"Finish Reason: {response.candidates[0].finish_reason}")
                    if response.candidates and response.candidates[0].safety_ratings: print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
                    if hasattr(response, 'text'): return f"[Error occurred, partial response]: {response.text}"
            except Exception as inner_e: print(f"ERROR: Further error retrieving details after main error: {inner_e}")
            return f"Sorry, an unexpected error occurred. Check logs. ({type(e).__name__})"


    def speak(self, output):
        """Prints the output character by character."""
        print("\nAlpha: ", end='')
        output_text = output if output else "[No response received]"
        for char in output_text:
            print(char, end='', flush=True)
            time.sleep(0.03)
        print("\n")


# --- Tool Functions (Memory Management & Date/Time) ---

@Alpha.add_func
def save_as_memory(title: str, content: str):
    """
    Save a piece of data as a memory for future reference. Overwrites if title exists.
    title: string: A short, descriptive title for the memory.
    content: string: The actual content or text of the memory to save.
    """
    data = []
    if os.path.isfile(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as file:
                content_read = file.read()
                if content_read: data = json.loads(content_read)
            if not isinstance(data, list):
                 print(f"Warning: {MEMORY_FILE} content was not a list. Resetting."); data = []
        except json.JSONDecodeError: print(f"Warning: Could not decode JSON from {MEMORY_FILE}. Resetting."); data = []
        except FileNotFoundError: pass
    memory_updated = False
    for item in data:
        if isinstance(item, dict) and item.get('title') == title:
            item['memory'] = content; memory_updated = True; break
    if not memory_updated: data.append({"title": title, "memory": content})
    try:
        with open(MEMORY_FILE, 'w') as file: json.dump(data, file, indent=4)
        status = "updated" if memory_updated else "saved"
        return f"Memory '{title}' {status} successfully."
    except Exception as e: print(f"ERROR: Failed to write to {MEMORY_FILE}: {e}"); return f"Error saving/updating memory: {e}"

@Alpha.add_func
def delete_memory(title: str):
    """
    Delete a memory permanently using its exact title.
    title: string: The exact title of the memory to delete.
    """
    if not os.path.isfile(MEMORY_FILE): return f"No memory file ({MEMORY_FILE}) found. Cannot delete."
    data = []
    try:
        with open(MEMORY_FILE, 'r') as file:
            content = file.read()
            if not content: return f"Memory file ({MEMORY_FILE}) is empty. Cannot delete."
            data = json.loads(content)
        if not isinstance(data, list): return f"Memory file ({MEMORY_FILE}) format is invalid (not a list)."
    except json.JSONDecodeError: return f"Memory file ({MEMORY_FILE}) is corrupted."
    except FileNotFoundError: return f"No memory file ({MEMORY_FILE}) found."
    original_length = len(data)
    updated_data = [memory for memory in data if not (isinstance(memory, dict) and memory.get("title") == title)]
    if len(updated_data) == original_length: return f"Memory with title '{title}' not found."
    else:
        try:
            with open(MEMORY_FILE, 'w') as file: json.dump(updated_data, file, indent=4)
            return f"Memory with title '{title}' deleted successfully."
        except Exception as e: print(f"ERROR: Failed to write {MEMORY_FILE}: {e}"); return f"Error writing updated memory file: {e}"

@Alpha.add_func
def get_memory(title: str):
    """
    Retrieve the content of a specific memory using its exact title.
    title: string: The exact title of the memory to retrieve.
    """
    if not os.path.isfile(MEMORY_FILE): return "No memory file found."
    try:
        with open(MEMORY_FILE, 'r') as file:
            content = file.read()
            if not content: return "Memory file is empty."
            data = json.loads(content)
        if not isinstance(data, list): return "Memory file format is invalid (not a list)."
        for memory in data:
            if isinstance(memory, dict) and memory.get('title') == title:
                return memory.get('memory', '[Error: Memory content missing for this title]')
        return f"Memory with title '{title}' not found."
    except json.JSONDecodeError: return "Memory file is corrupted or has invalid JSON."
    except FileNotFoundError: return "Memory file not found."
    except Exception as e: print(f"ERROR: Failed to read {MEMORY_FILE}: {e}"); return f"Error reading memory file: {e}"

@Alpha.add_func
def update_memory(title: str, new_content: str):
    """
    Update the content of an existing memory identified by its title. (Acts like save_as_memory).
    title: string: The exact title of the memory to update.
    new_content: string: The new text or content for the memory.
    """
    return save_as_memory(title=title, content=new_content)

@Alpha.add_func
def get_current_date_time():
    """
    Get the current date and time. Example: 25/07/2024, 16:30:55
    """
    from datetime import datetime
    now = datetime.now()
    formatted_datetime = now.strftime("%d/%m/%Y, %H:%M:%S")
    return formatted_datetime

# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Alpha Assistant (Gemini Edition) ---")
    print("Initializing...")
    try:
        ai = Alpha()
        print(f"Using model: {MODEL_NAME}")
        registered_tool_names = [f.name for f in ai.tool_declarations if f is not None]
        print(f"Registered tools: {registered_tool_names}")
        print("Enter your prompts below. Type '0' or 'exit' to quit.")
        print("-" * 20)
    except Exception as init_error:
        print(f"\nFATAL ERROR during initialization: {init_error}")
        print("Please check your API key and network connection.")
        sys.exit(1)

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower().strip() in ["0", "exit", "quit"]:
                print("\nAlpha: Goodbye!")
                break
            elif not user_input.strip(): continue
            else: prompt = user_input.strip()
            output = ai.chat_with_gemini(prompt)
            ai.speak(output)
    except KeyboardInterrupt: print("\nAlpha: Exiting due to user interruption.")
    except Exception as main_loop_error: print(f"\nFATAL ERROR during main loop: {main_loop_error}")
    finally:
        print("\nAlpha: Session ended.")
        # Optional history saving can be added here if needed