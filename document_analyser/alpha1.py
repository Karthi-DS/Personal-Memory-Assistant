# alpha1.py

import os
import json
import time
import traceback
from threading import Thread
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
import re
import inspect
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Tool, FunctionDeclaration, File

# Configuration
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file")

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Or "gemini-1.5-flash-latest"

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Define MEMORY_FILE and TEMP_UPLOAD_DIR
MEMORY_FILE = "memory.json"
TEMP_UPLOAD_DIR = "temp_uploads_alpha"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True) # Ensure temp dir exists

class TimeoutException(Exception):
    pass

# Timeout decorator
def timeout(seconds=60):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]; exception = [None]
            def target():
                try: result[0] = func(*args, **kwargs)
                except Exception as e: exception[0] = e
            thread = Thread(target=target); thread.start(); thread.join(seconds)
            if thread.is_alive():
                # TODO: Consider how to properly terminate the underlying API call if possible
                print(f"Warning: Timeout in '{func.__name__}' after {seconds}s.")
                raise TimeoutException(f"Operation timed out after {seconds} seconds")
            if exception[0]: raise exception[0]
            return result[0]
        return wrapper
    return decorator

class Alpha:
    tool_declarations: List[FunctionDeclaration] = []
    registered_functions: Dict[str, callable] = {}

    def __init__(self):
        self.history = [] # History managed by the ChatSession object internally now
        # --- REFINED SYSTEM PROMPT ---
        self.system_prompt = """You are Alpha, an advanced personal assistant specializing in document analysis and information recall.
You will be provided with context from uploaded files directly *for the current query*.
Your tasks are to:
1. Understand the content of the provided file(s) for the current turn.
2. Identify key sections and important information.
3. Provide concise summaries when asked.
4. Answer specific questions about the file content accurately *when provided*.
5. Use the 'save_as_memory' tool *if requested* to save summaries or key findings with a specific title for later reference. Provide informative titles.
6. Use the 'retrieve_memory' tool *if requested* to fetch information previously saved under a specific title.
7. If asked a question unrelated to the file context provided *in the current turn*, state that your current focus is the document *if one was just provided*. Otherwise, check your conversation history or use the 'retrieve_memory' tool if the question refers to saved information. If you cannot answer, indicate you need the file, more information, or the correct memory title."""
        try:
            self.model = genai.GenerativeModel(
                MODEL_NAME,
                safety_settings=SAFETY_SETTINGS,
                system_instruction=self.system_prompt
            )
            # Initialize chat history (empty for new session)
            self.chat = self.model.start_chat(history=[])
            print(f"DEBUG: Alpha initialized with model {MODEL_NAME}. Chat session started.")
            self._ensure_memory_file_exists()
        except Exception as e:
             print(f"FATAL ERROR during Alpha initialization: {e}")
             traceback.print_exc()
             raise

    def _ensure_memory_file_exists(self):
        # Creates or validates the memory JSON file.
        if not os.path.exists(MEMORY_FILE):
            print(f"DEBUG: Memory file '{MEMORY_FILE}' not found. Creating.")
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
            return

        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            # If file exists but is empty, initialize with []
            if not content.strip():
                print(f"DEBUG: Memory file '{MEMORY_FILE}' is empty. Initializing.")
                with open(MEMORY_FILE, 'w', encoding='utf-8') as wf:
                    json.dump([], wf)
                return

            # Try loading JSON to check validity
            data = json.loads(content)
            if not isinstance(data, list):
                print(f"Warning: Memory file '{MEMORY_FILE}' does not contain a list. Overwriting with empty list.")
                with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump([], f)

        except json.JSONDecodeError:
            print(f"Warning: Memory file '{MEMORY_FILE}' contains invalid JSON. Overwriting with empty list.")
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
        except Exception as e:
            print(f"Error checking memory file {MEMORY_FILE}: {e}. Attempting to reset.")
            try:
                with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            except Exception as write_e:
                print(f"FATAL: Could not reset memory file {MEMORY_FILE}: {write_e}")
                # Optionally re-raise or handle more gracefully
                raise write_e

    @timeout(120) # Increased timeout slightly for potentially complex file analysis + function calls
    def chat_with_gemini(self, prompt: str, file_context: Optional[List[File]] = None) -> str:
        """
        Sends a prompt and optional file context to the Gemini model and handles responses,
        including function calls.

        Args:
            prompt: The user's text prompt.
            file_context: An optional list of genai.File objects previously uploaded.

        Returns:
            The model's final text response.
        """
        try:
            contents = []
            # Add file context first if provided
            if file_context:
                contents.extend(file_context)
                print(f"DEBUG: Sending {len(file_context)} file(s) with prompt.")
            # Then add the text prompt
            contents.append(prompt)

            tools = self._prepare_tools() # This returns None or a list like [Tool(...)]
            print(f"DEBUG: Sending contents (types): {[type(c).__name__ for c in contents]}")
            if tools:
                # Access the first element of the list (the Tool object) then its declarations
                print(f"DEBUG: Sending with tools: {[d.name for d in tools[0].function_declarations]}")
            else:
                 print("DEBUG: Sending without tools.")

            # --- Send message to the chat session ---
            # Passing 'tools' (the list) here is correct for the API
            # The timeout decorator wraps this call
            response = self.chat.send_message(contents, tools=tools)

            # --- Handle potential function calls ---
            max_func_calls = 5
            call_count = 0
            while call_count < max_func_calls:
                # Check the latest response part for a function call
                # Sometimes the function call might not be the *very* last part if the model adds text after it.
                # Let's check all parts in the last content block.
                fc_part = None
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in reversed(response.candidates[0].content.parts): # Check from end
                        if hasattr(part, 'function_call') and part.function_call:
                            fc_part = part
                            # print(f"DEBUG: Found FunctionCall part: {fc_part}") # Verbose debug
                            break # Found the call

                if fc_part and fc_part.function_call:
                    function_call = fc_part.function_call
                    print(f"DEBUG: Handling function call: {function_call.name}")

                    # Execute the function call
                    api_response_content = self._handle_function_call(function_call)

                    if api_response_content:
                        print("DEBUG: Sending function response back to model.")
                        # Send the function response back to the model via the chat session
                        # The timeout decorator wraps this call too if needed
                        response = self.chat.send_message(api_response_content, tools=tools)
                        call_count += 1
                    else:
                        print("ERROR: _handle_function_call did not return valid content.")
                        # Constructing an error response to send back
                        error_response_for_api = {
                            "function_response": {
                                "name": function_call.name,
                                "response": {"content": f"Error: Tool '{function_call.name}' failed internally without specific error message."}
                            }
                        }
                        # The timeout decorator wraps this call too
                        response = self.chat.send_message(error_response_for_api, tools=tools)
                        # Don't increment call_count here, let the model react to the error
                        break # Exit loop after sending error response

                else:
                    # No function call in the latest response part, break the loop
                    # print("DEBUG: No function call found in the latest response part.") # Verbose debug
                    break # Exit the function call loop

            if call_count >= max_func_calls:
                print("WARN: Reached maximum tool interaction limit.")
                return "Reached maximum tool interaction limit. Please try simplifying your request."

            # --- Extract final text response ---
            final_text = ""
            # Ensure we look at the *final* response after potential function calls
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text'): # Fallback for simpler responses
                 final_text = response.text

            # Update internal history representation IF NEEDED (ChatSession does this)
            # self.history = list(self.chat.history) # Uncomment if you need manual access outside

            return final_text or "Received an empty or non-text response from the assistant."

        except TimeoutException as e:
            print(f"ERROR: Timeout during chat. {e}")
            return f"Processing took too long and was interrupted. ({e})"
        except Exception as e:
            print(f"ERROR: Unexpected error in chat_with_gemini: {type(e).__name__}")
            traceback.print_exc()
            # Attempt to get specific API error details if available
            api_error_details = ""
            if hasattr(e, 'response') and hasattr(e.response, 'status_code') and hasattr(e.response, 'text'): # For google.api_core.exceptions
                try: api_error_details = f" (API Status: {e.response.status_code}, Details: {e.response.text})"
                except: pass # Ignore if accessing response details fails
            elif hasattr(e, 'message'): # General exception message
                api_error_details = f" ({e.message})"
            elif isinstance(e, genai.types.BlockedPromptException):
                 api_error_details = " (Blocked due to safety settings or prompt issues)"
            elif isinstance(e, genai.types.StopCandidateException):
                 api_error_details = " (Stopped due to safety settings)"


            return f"An unexpected error occurred: {str(e)}{api_error_details}."

    def _handle_function_call(self, function_call) -> Optional[Dict[str, Any]]:
        """
        Executes registered function calls based on the model's request and
        returns a dictionary structured for the API's function response.

        Args:
            function_call: The FunctionCall object from the model response.

        Returns:
            A dictionary formatted for genai.ChatSession.send_message's
            FunctionResponse part, or None on critical failure before formatting.
        """
        func_name = function_call.name
        args = function_call.args

        print(f"DEBUG: Handling function call '{func_name}' with raw args: {args}")

        content = f"Error: Tool '{func_name}' encountered an unknown issue during execution." # Default error

        if func_name not in self.registered_functions:
            print(f"Warning: Function '{func_name}' not registered.")
            content = f"Error: Tool '{func_name}' is not available or not registered."
        else:
            function_to_call = self.registered_functions[func_name]
            args_dict = {} # Initialize args_dict here
            try:
                # Convert FunctionCall arguments (often dict-like) to a standard Python dictionary
                try:
                    args_dict = dict(args)
                except Exception as conversion_err:
                    print(f"DEBUG: Could not directly convert args to dict ({conversion_err}). Trying item iteration.")
                    # Fallback if direct dict conversion fails
                    args_dict = {key: value for key, value in args.items()} if hasattr(args, 'items') else {}
                    if not args_dict and args: # If still empty but args existed
                         print(f"Warning: Failed to convert args to dict: {args}")
                         content = f"Error: Could not process arguments for tool '{func_name}'."
                         # Skip further processing if args conversion failed badly
                         # Format and return the error immediately
                         return {
                             "function_response": {
                                 "name": func_name,
                                 "response": {"content": content}
                             }
                         }

                print(f"DEBUG: Converted Args for validation: {args_dict}")

                # --- Argument Validation and Filtering ---
                sig = inspect.signature(function_to_call)
                valid_param_names = set(sig.parameters.keys())

                # Filter the args received from LLM to only include valid parameter names
                filtered_args = {
                    k: v for k, v in args_dict.items() if k in valid_param_names
                }
                print(f"DEBUG: Filtered Args for execution: {filtered_args}")

                # Check for missing required arguments (those without default values)
                missing_required = []
                for param_name, param in sig.parameters.items():
                    if param.default is inspect.Parameter.empty and param_name not in filtered_args:
                        missing_required.append(param_name)

                if missing_required:
                    print(f"ERROR: Missing required arguments for '{func_name}': {missing_required}. Provided args: {args_dict}")
                    content = f"Error: Tool '{func_name}' was called without required arguments: {', '.join(missing_required)}. Please provide values for these arguments."
                else:
                    # --- Execute the function with filtered arguments ---
                    print(f"DEBUG: Executing function '{func_name}' with args: {filtered_args}")
                    # The timeout decorator is applied to chat_with_gemini, not individual tool calls here.
                    # If a tool needs its own timeout, it should be implemented within the tool function or via a separate decorator there.
                    result = function_to_call(**filtered_args)

                    # Convert result to string if it's not already
                    if not isinstance(result, str):
                        try:
                            # Use JSON dumps for structured data, fallback to str()
                            content = json.dumps(result)
                        except TypeError:
                            content = str(result)
                    else:
                        content = result

                    print(f"DEBUG: Function '{func_name}' executed successfully. Result length: {len(content)} chars. Type: {type(result).__name__}")
                # --- End Argument Validation ---

            except Exception as e:
                print(f"ERROR: Exception executing function '{func_name}' with args {filtered_args if 'filtered_args' in locals() else args_dict}.")
                traceback.print_exc()
                # Provide a more specific error message if possible
                content = f"Error executing tool '{func_name}': {type(e).__name__} - {str(e)}"

        # --- Format the response for the API ---
        # This structure is required by send_message when responding to a function call
        function_response_payload = {
            "function_response": {
                "name": func_name,
                "response": {
                    # The 'content' field within 'response' should contain the result
                    "content": content,
                }
            }
        }
        # print(f"DEBUG: Prepared function response payload: {function_response_payload}") # Verbose debug
        return function_response_payload


    @classmethod
    def _create_function_declaration(cls, func: callable) -> Optional[FunctionDeclaration]:
        """
        Generates a FunctionDeclaration object from a function's docstring.
        Format:
        First line: Description.
        Following lines (optional): Args:, Params:, Parameters: section.
        Each arg line: param_name (type_hint): Description [optional]
        """
        if not func.__doc__:
            print(f"DEBUG: Function '{func.__name__}' has no docstring. Cannot create declaration.")
            return None

        docstring = inspect.getdoc(func)
        if not docstring:
             print(f"DEBUG: Function '{func.__name__}' has empty docstring after inspect.getdoc. Cannot create declaration.")
             return None

        lines = docstring.strip().split('\n')
        description = lines[0].strip()
        if not description:
            print(f"DEBUG: Function '{func.__name__}' docstring lacks a description on the first line.")
            return None

        parameters = {"type": "object", "properties": {}, "required": []}
        param_section_found = False
        # Simple pattern: "param_name (type): description" or "param_name: description"
        # Allows optional type hint in parentheses. Captures description.
        param_pattern = re.compile(r"^\s*(\w+)\s*(?:\(([^)]+)\))?:\s*(.*)$")

        param_lines = []
        for line in lines[1:]:
             line_stripped = line.strip()
             if not param_section_found:
                 if line_stripped.lower().startswith(('args:', 'params:', 'parameters:')):
                     param_section_found = True
                 continue # Skip lines before the parameter section header or blank lines within header section
             if line_stripped: # Only process non-empty lines after section start
                param_lines.append(line_stripped)

        if not param_section_found and len(lines) > 1:
             # If no explicit section, assume lines after description *might* be params if they match pattern
             print(f"DEBUG: No explicit Args/Params section for '{func.__name__}'. Trying to parse subsequent lines.")
             param_lines = [l.strip() for l in lines[1:] if l.strip()]


        for line in param_lines:
            match = param_pattern.match(line)
            if match:
                param_name, type_hint, desc = match.groups()
                param_name = param_name.strip()
                desc = desc.strip()

                # Infer basic types from hint or keywords in description
                param_type = 'string' # Default type
                if type_hint:
                    type_hint_lower = type_hint.strip().lower()
                    if 'int' in type_hint_lower or 'integer' in type_hint_lower: param_type = 'integer'
                    elif 'float' in type_hint_lower or 'number' in type_hint_lower: param_type = 'number'
                    elif 'bool' in type_hint_lower or 'boolean' in type_hint_lower: param_type = 'boolean'
                    elif 'list' in type_hint_lower or 'array' in type_hint_lower: param_type = 'array' # Basic array support
                    # Add more complex type mappings if needed (e.g., object)
                elif 'integer' in desc.lower() or 'number' in desc.lower() and 'float' not in desc.lower():
                    param_type = 'integer' # Guess based on description keyword
                elif 'float' in desc.lower() or 'decimal' in desc.lower():
                    param_type = 'number'
                elif 'boolean' in desc.lower() or 'true/false' in desc.lower():
                     param_type = 'boolean'


                parameters['properties'][param_name] = {
                    "type": param_type,
                    "description": desc
                }

                # Determine if required: Parameter is required if '[optional]' is NOT in the description
                # AND if the function signature doesn't specify a default value
                sig = inspect.signature(func)
                is_required_by_sig = (param_name in sig.parameters and
                                    sig.parameters[param_name].default is inspect.Parameter.empty)

                if '[optional]' not in desc.lower() and is_required_by_sig:
                    parameters['required'].append(param_name)
            else:
                print(f"DEBUG: Line in docstring for '{func.__name__}' did not match param pattern: '{line}'")


        # Clean up parameters dict if no properties were found or no params are required
        if not parameters['properties']:
            parameters = None # No parameters defined
            print(f"DEBUG: No parameters extracted for '{func.__name__}'.")
        elif not parameters.get('required'): # Use .get() in case 'required' was never added
             if 'required' in parameters: del parameters['required'] # 'required' field should not be empty list
             print(f"DEBUG: Parameters extracted for '{func.__name__}' (all optional): {list(parameters['properties'].keys())}")
        else:
             # Ensure required list has unique entries and exists in properties
             parameters['required'] = sorted(list(set(p for p in parameters['required'] if p in parameters['properties'])))
             if not parameters['required']: # If filtering removed all required
                  del parameters['required']
                  print(f"DEBUG: Parameters extracted for '{func.__name__}' (all optional after validation): {list(parameters['properties'].keys())}")
             else:
                 print(f"DEBUG: Parameters extracted for '{func.__name__}': {list(parameters['properties'].keys())}, Required: {parameters['required']}")


        # Create the FunctionDeclaration
        try:
             declaration = FunctionDeclaration(
                 name=func.__name__,
                 description=description,
                 parameters=parameters # Pass None if no parameters
             )
             print(f"DEBUG: Successfully created FunctionDeclaration for '{func.__name__}'.")
             return declaration
        except Exception as e:
             print(f"ERROR: Failed to create FunctionDeclaration for '{func.__name__}': {e}")
             traceback.print_exc()
             return None


    @classmethod
    def add_func(cls, func: callable) -> callable:
        """Class method decorator to register a function as a tool."""
        declaration = cls._create_function_declaration(func)
        if declaration:
            cls.registered_functions[func.__name__] = func
            # Avoid adding duplicates if script reloads in some environments
            if declaration.name not in [d.name for d in cls.tool_declarations]:
                 cls.tool_declarations.append(declaration)
                 print(f"DEBUG: Registered function '{func.__name__}' as tool.")
            else:
                 print(f"DEBUG: Function '{func.__name__}' already registered. Overwriting declaration and function reference.")
                 # Remove old one and add new one to handle potential updates during development
                 cls.tool_declarations = [d for d in cls.tool_declarations if d.name != declaration.name]
                 cls.tool_declarations.append(declaration)

        else:
            print(f"Warning: Could not create declaration for function '{func.__name__}'. It will not be available as a tool.")
        return func # Return the original function so it can still be called directly

    def _prepare_tools(self) -> Optional[List[Tool]]:
        """Prepares the list of Tool objects for the API call."""
        if not self.tool_declarations:
            return None
        # Filter out any None entries that might have occurred during declaration creation
        valid_declarations = [d for d in self.tool_declarations if d is not None]
        if not valid_declarations:
            return None
        # Gemini API expects Tools argument as a list containing one Tool object
        # which itself contains the list of function declarations.
        return [Tool(function_declarations=valid_declarations)]

# --- Tool Function Definitions ---

@Alpha.add_func
def save_as_memory(title: str, content: str) -> str:
    """Saves or updates information under a given title in the memory file (memory.json).

    Args:
        title (string): The unique title or key for this memory entry. Cannot be empty.
        content (string): The text content to store (e.g., a summary, notes, key findings). Cannot be empty.

    Returns:
        string: A status message indicating success or failure of the save operation.
    """
    print(f"DEBUG: Tool 'save_as_memory' called with title='{title}', content='{content[:50]}...'")
    # Basic input validation
    if not title or not isinstance(title, str) or not title.strip():
        return "Error: Memory title cannot be empty or invalid. Please provide a specific, non-empty string title."
    if not content or not isinstance(content, str) or not content.strip():
        return "Error: Memory content cannot be empty or invalid. Please provide the non-empty string content to save."

    title = title.strip() # Normalize title
    content = content.strip() # Normalize content

    try:
        memories = []
        # Read existing memories, handling potential file issues
        try:
            # Ensure file exists before reading
            if not os.path.exists(MEMORY_FILE):
                 print(f"DEBUG: {MEMORY_FILE} not found during save attempt. Will create new.")
                 # _ensure_memory_file_exists should have created it, but double check / handle race condition possibility
                 with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump([], f)

            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                file_content = f.read()
            if file_content.strip(): # Check if file is not empty
                loaded_data = json.loads(file_content)
                if isinstance(loaded_data, list):
                    memories = loaded_data
                else:
                    print(f"Warning: Content of {MEMORY_FILE} is not a list. Resetting memory before save.")
                    memories = [] # Reset if format is wrong
            # If file is empty or just whitespace, memories remains []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {MEMORY_FILE}. Resetting memory before save.")
            memories = [] # Reset if JSON is invalid
        except FileNotFoundError:
             print(f"Warning: {MEMORY_FILE} not found even after check. Creating.")
             with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                 json.dump([], f)
             memories = [] # Start fresh
        except Exception as read_err:
            print(f"Error reading {MEMORY_FILE}, cannot guarantee save integrity: {read_err}")
            # Decide on behavior: Proceed with potentially incomplete data? Or fail? Let's fail for safety.
            return f"Error: Could not reliably read existing memory from {MEMORY_FILE}. Save aborted."

        # Check if title exists and update, otherwise append
        updated = False
        for i, mem in enumerate(memories):
            # Ensure we are checking dictionaries with a 'title' key
            if isinstance(mem, dict) and mem.get('title') == title:
                memories[i]['content'] = content # Update content
                memories[i]['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S %Z") # Add/Update timestamp
                updated = True
                print(f"DEBUG: Updating existing memory entry: '{title}'")
                break

        if not updated:
            memories.append({
                'title': title,
                'content': content,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S %Z") # Add timestamp for new entry
                })
            print(f"DEBUG: Adding new memory entry: '{title}'")

        # Write the updated list back to the file
        try:
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(memories, f, indent=2, ensure_ascii=False)
            return f"Success: Memory entry '{title}' has been saved/updated."
        except Exception as write_e:
             print(f"ERROR: Failed to write updated memory to {MEMORY_FILE}: {write_e}")
             traceback.print_exc()
             return f"Error: Failed to write memory to file for title '{title}' due to a system error."

    except Exception as e:
        print(f"ERROR: Unexpected error in 'save_as_memory' for title '{title}'.")
        traceback.print_exc()
        return f"Error: An unexpected system error occurred while trying to save memory for '{title}': {str(e)}"

@Alpha.add_func
def retrieve_memory(title: str) -> str:
    """Retrieves saved information from memory using its title.

    Args:
        title (string): The unique title of the memory entry to retrieve. Cannot be empty.

    Returns:
        string: The content of the memory entry, or a message indicating it wasn't found or an error occurred.
    """
    print(f"DEBUG: Tool 'retrieve_memory' called with title='{title}'")
    if not title or not isinstance(title, str) or not title.strip():
        return "Error: Memory title cannot be empty or invalid. Please provide the specific title to retrieve."

    title = title.strip() # Normalize title

    try:
        memories = []
        # Read existing memories, handling potential file issues
        try:
            if not os.path.exists(MEMORY_FILE):
                 print(f"DEBUG: {MEMORY_FILE} not found during retrieve attempt. Cannot retrieve.")
                 return f"Error: Memory file '{MEMORY_FILE}' does not exist. Nothing to retrieve."

            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                file_content = f.read()
            if file_content.strip(): # Check if file is not empty
                loaded_data = json.loads(file_content)
                if isinstance(loaded_data, list):
                    memories = loaded_data
                else:
                    print(f"Warning: Content of {MEMORY_FILE} is not a list. Cannot retrieve accurately.")
                    return f"Error: Memory file '{MEMORY_FILE}' has an invalid format. Cannot retrieve."
            else:
                # File exists but is empty
                print(f"DEBUG: {MEMORY_FILE} is empty. Cannot retrieve.")
                return f"No memory entry found with the title '{title}' (memory is empty)."

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {MEMORY_FILE}. Cannot retrieve.")
            return f"Error: Memory file '{MEMORY_FILE}' is corrupted or has invalid JSON. Cannot retrieve."
        except FileNotFoundError:
             print(f"Error: {MEMORY_FILE} not found unexpectedly during read.")
             return f"Error: Memory file '{MEMORY_FILE}' could not be accessed."
        except Exception as read_err:
            print(f"Error reading {MEMORY_FILE}: {read_err}")
            return f"Error: Could not read memory from {MEMORY_FILE} due to a system error."

        # Search for the title
        found_content = None
        for mem in memories:
            # Ensure we are checking dictionaries with a 'title' key
            if isinstance(mem, dict) and mem.get('title') == title:
                found_content = mem.get('content') # Get the content
                if found_content is None: # Handle case where title exists but content is missing/null
                     print(f"Warning: Memory entry '{title}' found but has missing or null content.")
                     return f"Error: Memory entry '{title}' exists but its content is missing or empty."
                print(f"DEBUG: Found memory entry: '{title}'")
                break # Stop searching once found

        if found_content is not None:
            # Maybe add timestamp info if useful for the LLM?
            # timestamp = mem.get('timestamp', 'unknown time')
            # return f"Retrieved content for '{title}' (saved {timestamp}):\n{found_content}"
            return found_content # Return only the content as requested by prompt usually
        else:
            print(f"DEBUG: Memory entry not found for title: '{title}'")
            return f"No memory entry found with the title '{title}'. Please check the title or save it first."

    except Exception as e:
        print(f"ERROR: Unexpected error in 'retrieve_memory' for title '{title}'.")
        traceback.print_exc()
        return f"Error: An unexpected system error occurred while trying to retrieve memory for '{title}': {str(e)}"


# --- Example Usage (Optional - usually run from a separate script) ---
if __name__ == "__main__":
    print("Initializing Alpha...")
    alpha_instance = Alpha()
    print("Alpha initialized. Available tools:", [d.name for d in alpha_instance.tool_declarations])

    # Example Interaction Flow:
    print("\n--- Interaction 1: Analyze and Save ---")
    # Simulate uploading a file (replace with actual genai.upload_file if needed)
    # For this example, we'll skip actual file upload and just pretend.
    # fake_file_context = [...]
    prompt1 = "Please summarize the main points of the provided document (imagine one was provided) and save it as 'Meeting Notes Summary'."
    response1 = alpha_instance.chat_with_gemini(prompt1) # , file_context=fake_file_context)
    print("\nAlpha Response 1:")
    print(response1)

    # Let's check memory.json manually or assume the above worked.
    # Manually create/update memory.json for testing retrieval if the LLM call failed:
    # {"title": "Meeting Notes Summary", "content": "This is a simulated summary of meeting notes.", "timestamp": "..."}

    print("\n--- Interaction 2: Retrieve ---")
    prompt2 = "Can you remind me what was in the 'Meeting Notes Summary'?"
    response2 = alpha_instance.chat_with_gemini(prompt2)
    print("\nAlpha Response 2:")
    print(response2)

    print("\n--- Interaction 3: Retrieve Non-existent ---")
    prompt3 = "What did I save under 'Project Plan Draft'?"
    response3 = alpha_instance.chat_with_gemini(prompt3)
    print("\nAlpha Response 3:")
    print(response3)

    print("\n--- Interaction 4: Save again (Update) ---")
    prompt4 = "Actually, update the 'Meeting Notes Summary'. Add that 'Action item: John to follow up'."
    response4 = alpha_instance.chat_with_gemini(prompt4)
    print("\nAlpha Response 4:")
    print(response4)

    print("\n--- Interaction 5: Retrieve Updated ---")
    prompt5 = "Show me the latest 'Meeting Notes Summary'."
    response5 = alpha_instance.chat_with_gemini(prompt5)
    print("\nAlpha Response 5:")
    print(response5)