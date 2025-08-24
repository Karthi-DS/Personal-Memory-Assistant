# app.py (Merged Application with Mode-Switching Sidebar & Calendar)
import streamlit as st
import os
import sys
import traceback # For detailed error logging
from datetime import datetime, timedelta # Need timedelta for calendar view potentially
import time
import json
from streamlit_autorefresh import st_autorefresh
import google.generativeai as genai
import base64 # For embedding alarm sound

# Check if File type exists, handle potential import variations
# Ensure google.generativeai is installed: pip install google-generativeai
try:
    from google.generativeai.types import File
    # Check if File type is usable (simple heuristic)
    if not hasattr(File, 'to_dict') and not hasattr(File, 'name'):
         print("Warning: Imported 'File' from google.generativeai.types does not seem like the expected type.")
         File = None # Treat as not found if it doesn't look right
except ImportError:
    print("Warning: google.generativeai.types.File not found. Document upload might be affected. Please ensure 'google-generativeai' library is installed and up-to-date.")
    File = None # Placeholder if import fails

from typing import Optional, List

# --- Add Calendar Import ---
# Ensure streamlit-calendar is installed: pip install streamlit-calendar
try:
    from streamlit_calendar import calendar # Import the calendar component
except ImportError:
    st.error("ERROR: `streamlit-calendar` library not found. Please install it: `pip install streamlit-calendar`")
    st.stop() # Stop execution if calendar is essential and missing

# === 1. Streamlit Page Configuration ===
st.set_page_config(
    page_title="Unified Assistant",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Title will be set dynamically based on mode

# === 2. Locate and Import Helper Modules ===
# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add this directory to the Python path if it's not already there
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
print(f"DEBUG: Current directory '{current_dir}' added to sys.path.")
print(f"DEBUG: sys.path = {sys.path}")

# --- Attempt to Import Alpha (General Assistant) ---
alpha_file_path = os.path.join(current_dir, 'alpha1.py')
alpha_available = False
alpha_init_error = None
Alpha, _load_memory_data, _save_memory_data, MEMORY_FILE = None, None, None, None
try:
    print(f"DEBUG: Looking for alpha1.py at: {alpha_file_path}")
    if not os.path.isfile(alpha_file_path):
        raise FileNotFoundError(f"'alpha1.py' not found in the directory: {current_dir}")

    # Import necessary components from alpha1
    from alpha1 import Alpha, _load_memory_data, _save_memory_data, MEMORY_FILE
    alpha_available = True
    print("DEBUG: Successfully imported Alpha, _load_memory_data, _save_memory_data, MEMORY_FILE from alpha1.py")
    # Check if imported objects are callable/valid where expected
    if not callable(Alpha): alpha_init_error = "'Alpha' imported from alpha1.py is not callable (not a class?)."
    if not callable(_load_memory_data): alpha_init_error = "'_load_memory_data' imported from alpha1.py is not callable."
    if not callable(_save_memory_data): alpha_init_error = "'_save_memory_data' imported from alpha1.py is not callable."
    if not isinstance(MEMORY_FILE, str) and MEMORY_FILE is not None: alpha_init_error = "'MEMORY_FILE' imported from alpha1.py is not a string or None."
    if alpha_init_error: alpha_available = False # Mark as unavailable if checks fail

    # Optional: Check for schedule if alpha1 uses it internally
    try: import schedule # noqa: F401
    except ImportError: print("DEBUG: Optional 'schedule' library not found (may be used by alpha1).")

except (ImportError, FileNotFoundError, ModuleNotFoundError) as e:
    alpha_init_error = f"Failed to import `Alpha` or helpers from `alpha1.py`. Reason: `{e}`. Check file existence, path, and dependencies."
    print(f"ERROR: {alpha_init_error}")
except Exception as e:
    alpha_init_error = f"An unexpected error occurred importing from `alpha1.py`: {e}"
    print(f"ERROR: {alpha_init_error}")
    traceback.print_exc()

# --- Attempt to Import DocumentAlpha (Document Analysis) ---
doc_alpha_file_path = os.path.join(current_dir, 'DocumentAlpha.py')
doc_alpha_available = False
doc_alpha_init_error = None
DocumentAlpha, TEMP_UPLOAD_DIR, TimeoutException = None, None, None
try:
    print(f"DEBUG: Looking for DocumentAlpha.py at: {doc_alpha_file_path}")
    if not os.path.isfile(doc_alpha_file_path):
        raise FileNotFoundError(f"'DocumentAlpha.py' not found in the directory: {current_dir}")

    # Import necessary components from DocumentAlpha
    from DocumentAlpha import DocumentAlpha, TEMP_UPLOAD_DIR, TimeoutException
    doc_alpha_available = True
    print("DEBUG: Successfully imported DocumentAlpha, TEMP_UPLOAD_DIR, TimeoutException from DocumentAlpha.py")
     # Check if imported objects are callable/valid where expected
    if not callable(DocumentAlpha): doc_alpha_init_error = "'DocumentAlpha' imported from DocumentAlpha.py is not callable (not a class?)."
    if not isinstance(TEMP_UPLOAD_DIR, str): doc_alpha_init_error = "'TEMP_UPLOAD_DIR' imported from DocumentAlpha.py is not a string."
    if not isinstance(TimeoutException, type) or not issubclass(TimeoutException, Exception): doc_alpha_init_error = "'TimeoutException' imported from DocumentAlpha.py is not a valid Exception class."
    if doc_alpha_init_error: doc_alpha_available = False # Mark as unavailable if checks fail

except (ImportError, FileNotFoundError, ModuleNotFoundError) as e:
    doc_alpha_init_error = f"Failed to import `DocumentAlpha` from `DocumentAlpha.py`. Reason: `{e}`. Check file existence, path, and dependencies (like google-generativeai)."
    print(f"ERROR: {doc_alpha_init_error}")
except Exception as e:
    doc_alpha_init_error = f"An unexpected error occurred importing from `DocumentAlpha.py`: {e}"
    print(f"ERROR: {doc_alpha_init_error}")
    traceback.print_exc()

# === 3. Styling (Optional) ===
st.markdown("""
<style>
    /* General Chat Bubbles */
    .stChatMessage {
        border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 1rem;
        border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        word-wrap: break-word; /* Ensure long words break */
        overflow-wrap: break-word;
    }
    /* --- Calendar Specific Styling --- */
    .fc-view-harness { /* Limit calendar height */
        height: 350px !important;
    }
    .fc-daygrid-day:hover { /* Indicate clickable days */
        cursor: pointer;
        background-color: #f0f0f0; /* Light grey hover */
    }
    .fc-daygrid-day.fc-day-today { /* Highlight today's date */
        background-color: rgba(255, 220, 40, 0.15) !important; /* Subtle yellow highlight */
    }
    /* --- End Calendar Styling --- */

    /* --- Optional: Hide default audio controls --- */
    /*
    audio {
        display: none;
    }
    */
</style>
""", unsafe_allow_html=True)


# === 4. Reminder Checking Function (Depends on Alpha) ===

# --- CONFIGURATION FOR ALARM SOUND ---
# Place your alarm sound file (e.g., "alarm.mp3", "alarm.wav") in the same directory as app.py
# Ensure the file *actually exists* at this location relative to app.py
ALARM_SOUND_FILE = "alarm.wav" # MODIFY THIS if your file name is different
# -----------------------------------

def play_alarm_sound():
    """Reads, encodes, and embeds the alarm sound for autoplay using HTML5 audio."""
    try:
        sound_file_path = os.path.join(current_dir, ALARM_SOUND_FILE)
        if not os.path.exists(sound_file_path):
            print(f"Warning: Alarm sound file not found at {sound_file_path}")
            st.warning(f"Alarm sound '{ALARM_SOUND_FILE}' not found.", icon="🔊")
            return

        with open(sound_file_path, "rb") as f:
            sound_bytes = f.read()
        sound_b64 = base64.b64encode(sound_bytes).decode()

        # Determine mime type based on file extension
        file_ext = os.path.splitext(ALARM_SOUND_FILE)[1].lower()
        if file_ext == ".mp3":
            mime_type = "audio/mpeg"
        elif file_ext == ".wav":
            mime_type = "audio/wav"
        elif file_ext == ".ogg":
            mime_type = "audio/ogg"
        else:
            # Default or guess common types if extension is unknown
            mime_type = "audio/mpeg"
            print(f"Warning: Unknown audio file extension '{file_ext}'. Assuming '{mime_type}'. Supported types depend on browser.")

        sound_data_uri = f"data:{mime_type};base64,{sound_b64}"

        # Embed HTML5 audio player with autoplay
        # Note: Autoplay might be blocked by browser policies until the user interacts with the page.
        # It usually works in Streamlit after the first interaction.
        st.markdown(
            f'<audio autoplay="true" src="{sound_data_uri}"></audio>',
            unsafe_allow_html=True,
        )
        print(f"DEBUG: Attempting to play alarm sound ({ALARM_SOUND_FILE}) via embedded HTML audio.") # Log attempt

    except FileNotFoundError:
        # This case should be caught by os.path.exists above, but kept for robustness
        print(f"ERROR: Alarm sound file '{ALARM_SOUND_FILE}' not found at path: {sound_file_path}")
        st.error(f"Error finding alarm sound file: {ALARM_SOUND_FILE}")
    except Exception as audio_e:
        print(f"ERROR: Failed to load or embed alarm sound: {audio_e}")
        traceback.print_exc()
        st.error(f"Error playing alarm sound: {audio_e}")


def check_and_display_reminders():
    """Checks memory for pending reminders and displays/triggers them."""
    if not alpha_available or not callable(_load_memory_data) or not callable(_save_memory_data) or not MEMORY_FILE:
        # Silently return if prerequisites aren't met
        # print("DEBUG (Reminder Check): Alpha or memory functions/file unavailable.") # Can be noisy
        return

    try:
        now = datetime.now()
        # print(f"DEBUG: Checking reminders at {now.strftime('%Y-%m-%d %H:%M:%S')}") # Optional: Log checks
        data = _load_memory_data() # Load current data

        # Validate data structure
        if not isinstance(data, dict):
            print(f"ERROR: Memory data loaded from '{MEMORY_FILE}' is not a dictionary. Cannot process reminders.")
            return
        reminders = data.get("Reminders", [])
        if not isinstance(reminders, list):
            print("Warning: 'Reminders' key in memory does not contain a list. Resetting to empty list.")
            data["Reminders"] = [] # Attempt to fix by resetting
            _save_memory_data(data) # Save the fix
            return

        reminders_updated = False
        played_sound_this_run = False # Flag to prevent multiple sounds in one refresh cycle
        processed_ids_this_run = set() # Track reminders processed in this specific run

        # Use list comprehension for potential modification safety if needed, but direct iteration is okay here
        for index, reminder in enumerate(list(reminders)): # Iterate over a copy if modifying list size
            try:
                reminder_id = reminder.get('id', f"index_{index}") # Use index as fallback ID
                if reminder_id in processed_ids_this_run: continue # Skip if already processed this run

                # Basic validation: Ensure it's a dict and has 'pending' status
                if not isinstance(reminder, dict):
                    print(f"Warning: Item at index {index} in Reminders is not a dictionary: {reminder}")
                    processed_ids_this_run.add(reminder_id) # Mark as processed (invalid)
                    continue
                if reminder.get("status") != "pending":
                    processed_ids_this_run.add(reminder_id) # Mark as processed (not pending)
                    continue

                # --- Check Time ---
                reminder_time_str = reminder.get("reminder_time")
                if not reminder_time_str or not isinstance(reminder_time_str, str):
                    print(f"Warning: Missing or invalid 'reminder_time' for reminder ID {reminder_id}: {reminder_time_str}")
                    if reminder.get('status') != 'error_missing_time':
                       reminder['status'] = 'error_missing_time'; reminders_updated = True
                    processed_ids_this_run.add(reminder_id); continue

                # Validate and parse reminder time (using consistent format)
                try:
                    # Ensure the format matches how reminders are saved by Alpha
                    reminder_dt = datetime.strptime(reminder_time_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    print(f"Warning: Invalid date format '{reminder_time_str}' for reminder ID {reminder_id}. Expected 'YYYY-MM-DD HH:MM:SS'.")
                    if reminder.get('status') != 'error_bad_format':
                         reminder['status'] = 'error_bad_format'; reminders_updated = True
                    processed_ids_this_run.add(reminder_id); continue

                # --- Trigger Condition ---
                if reminder_dt <= now:
                    message = reminder.get('message', 'Reminder!')
                    print(f"DEBUG: Triggering reminder: ID {reminder_id} - Time {reminder_time_str} - Message: {message}")

                    # Use st.toast for non-blocking notification
                    st.toast(f"🔔 Reminder: {message}", icon="⏰")

                    # --- Play Sound and Update Status (Only Once per reminder trigger) ---
                    # Change status from 'pending' to 'triggered'
                    if reminder.get('status') == 'pending': # Double-check status before changing
                        reminder['status'] = 'triggered'
                        reminders_updated = True
                        print(f"DEBUG: Updated status to 'triggered' for reminder ID {reminder_id}")

                        # Play the alarm sound only if not already played during this specific check cycle
                        if not played_sound_this_run:
                            play_alarm_sound()
                            played_sound_this_run = True # Set flag for this cycle

                    processed_ids_this_run.add(reminder_id) # Mark as processed for this run

            except Exception as e:
                # Handle errors within the loop for a specific reminder gracefully
                reminder_id_err = reminder.get('id', f"error_index_{index}") if isinstance(reminder, dict) else f"error_index_{index}"
                print(f"ERROR: Unexpected error processing reminder {reminder_id_err}: {e}")
                traceback.print_exc() # Log the full traceback for debugging
                # Update status to avoid repeated errors on the same reminder (if possible)
                if isinstance(reminder, dict) and reminder.get('status') != 'error_unexpected':
                    try:
                         reminder['status'] = 'error_unexpected'; reminders_updated = True
                    except Exception: pass # Ignore if reminder object itself is malformed
                processed_ids_this_run.add(reminder_id_err)

        # Save changes back to memory file *only if* any reminder statuses were updated
        if reminders_updated:
            print("DEBUG: Reminders updated, attempting to save memory data...")
            if not _save_memory_data(data):
                print("ERROR (Reminder Check): _save_memory_data failed after updating reminder statuses.")
            else:
                print("DEBUG: Successfully saved updated reminder statuses to memory.")

    except FileNotFoundError:
         # Handle case where memory file might disappear between checks (unlikely but possible)
         print(f"ERROR: Memory file '{MEMORY_FILE}' not found during reminder check.")
    except json.JSONDecodeError:
         print(f"ERROR: Failed to decode JSON from memory file '{MEMORY_FILE}' during reminder check. File might be corrupt.")
    except Exception as e:
        # Catch-all for errors during loading or overall processing
        print(f"ERROR (Reminder Check): General failure in check_and_display_reminders: {e}")
        traceback.print_exc()


# === 5. Initialize Session State ===
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    # --- App Wide ---
    if 'app_mode' not in st.session_state: st.session_state.app_mode = '🤖 General Assistant' # Default mode

    # --- Alpha (General Assistant) ---
    if 'alpha_instance' not in st.session_state: st.session_state.alpha_instance = None
    if 'alpha_model_name' not in st.session_state: st.session_state.alpha_model_name = "Unavailable"
    if 'alpha_init_done' not in st.session_state: st.session_state.alpha_init_done = False
    if 'alpha_init_error_msg' not in st.session_state: st.session_state.alpha_init_error_msg = alpha_init_error # Store import error if any
    if 'general_messages' not in st.session_state: st.session_state.general_messages = [] # Chat history for Alpha
    if 'selected_calendar_date' not in st.session_state: st.session_state.selected_calendar_date = None # State for calendar interaction

    # --- DocumentAlpha ---
    if 'doc_alpha_instance' not in st.session_state: st.session_state.doc_alpha_instance = None
    if 'doc_alpha_init_done' not in st.session_state: st.session_state.doc_alpha_init_done = False
    if 'doc_alpha_init_error_msg' not in st.session_state: st.session_state.doc_alpha_init_error_msg = doc_alpha_init_error # Store import error if any
    if 'document_messages' not in st.session_state: st.session_state.document_messages = [] # Chat history for DocumentAlpha
    if 'current_file_context' not in st.session_state: st.session_state.current_file_context = None # Holds the Google AI File object(s)
    if 'current_file_display_name' not in st.session_state: st.session_state.current_file_display_name = None # For display in UI
    if 'file_being_processed' not in st.session_state: st.session_state.file_being_processed = None # Lock to prevent race conditions during upload

initialize_session_state() # Call initialization function


# === 6. Initialize Assistants (Run only once per session) ===
def initialize_assistants():
    """Initializes Alpha and DocumentAlpha instances if available and not already done."""

    # Initialize Alpha (General Assistant)
    # Condition: alpha1.py was found and imported, and init hasn't been attempted yet
    if alpha_available and not st.session_state.alpha_init_done:
        st.session_state.alpha_init_done = True # Mark as attempted, even if it fails
        try:
            print("DEBUG: Attempting to initialize Alpha (General Assistant)...")
            with st.spinner("Waking up Alpha (General Assistant)..."):
                # Check/Create Memory File (if paths and functions are valid)
                if callable(_load_memory_data) and callable(_save_memory_data) and MEMORY_FILE:
                    initial_data = _load_memory_data() # Load first to see if it exists/is valid
                    if not os.path.exists(MEMORY_FILE):
                        print(f"DEBUG (App Init): Memory file '{MEMORY_FILE}' not found, saving default structure.")
                        # Ensure default structure includes keys expected by reminder/calendar features
                        default_structure = {"Reminders": [], "Schedule": {}}
                        if not _save_memory_data(initial_data or default_structure):
                            st.warning("Could not create initial memory file for Alpha.")
                    elif initial_data is None: # Handle case where file exists but load fails
                         print(f"Warning: Failed to load data from existing memory file '{MEMORY_FILE}'. Check file content/permissions.")
                         # Optionally try saving default structure again? Or just warn.
                elif not (callable(_load_memory_data) and callable(_save_memory_data)):
                    st.warning("Alpha memory functions (_load/_save) not available/callable. Reminders/Schedule might not persist.")
                elif not MEMORY_FILE:
                     st.warning("MEMORY_FILE path not defined in alpha1.py. Cannot check/create memory file.")

                # Instantiate Alpha - This is where Alpha's __init__ is called
                alpha_init_instance = Alpha()

            # Basic check if initialization seemed successful (e.g., model attribute exists)
            if alpha_init_instance and hasattr(alpha_init_instance, 'model'):
                st.session_state.alpha_instance = alpha_init_instance
                st.toast("Alpha (General Assistant) initialized!", icon="🤖")
                print("DEBUG: Alpha initialized successfully.")

                # Get model name safely
                try:
                    # Access model info (might vary based on how Alpha stores it)
                    model_obj = getattr(st.session_state.alpha_instance, 'model', None)
                    # Gemini models often store full name in _model_name or similar private attr
                    full_model_path = getattr(model_obj, '_model_name', "Unknown Model Path")
                    # Extract the common model name part
                    st.session_state.alpha_model_name = full_model_path.split('/')[-1]
                except Exception as e:
                    print(f"Warning: Could not retrieve Alpha model name: {e}")
                    st.session_state.alpha_model_name = "Model name N/A"

                # Add initial message only if chat history is empty
                if not st.session_state.general_messages:
                     st.session_state.general_messages.append({
                        "role": "assistant",
                        "content": "Hello! I'm Alpha. How can I help you today? Ask questions, set reminders, or use the calendar in the sidebar."})
            else:
                # Instance created but seems incomplete or invalid
                raise RuntimeError("Alpha instance created but appears incomplete (e.g., missing 'model' attribute?). Check Alpha's __init__ method.")

        except Exception as init_error:
            print(f"ERROR: Failed to initialize Alpha: {init_error}", file=sys.stderr)
            traceback.print_exc()
            # Store the runtime error message
            st.session_state.alpha_init_error_msg = f"Runtime Error Initializing Alpha: {init_error}"
            st.session_state.alpha_instance = None # Ensure it's None on failure
            st.session_state.alpha_model_name = "Initialization Failed"
            # Add error message to chat if empty
            if not st.session_state.general_messages:
                 st.session_state.general_messages = [{"role": "assistant", "content": f"Error: Could not initialize Alpha. {st.session_state.alpha_init_error_msg}"}]

    # Initialize DocumentAlpha
    # Condition: DocumentAlpha.py was found and imported, and init hasn't been attempted yet
    if doc_alpha_available and not st.session_state.doc_alpha_init_done:
        st.session_state.doc_alpha_init_done = True # Mark as attempted
        try:
            print("DEBUG: Attempting to initialize DocumentAlpha...")
            # Basic check for Google API key (essential for DocumentAlpha)
            # This relies on genai being configured externally (e.g., env var) before this point.
            google_api_key = os.getenv('GOOGLE_API_KEY')
            genai_configured = False
            if google_api_key:
                try:
                    # Try configuring explicitly if key found (might already be done)
                    genai.configure(api_key=google_api_key)
                    print("DEBUG: genai configured with GOOGLE_API_KEY from environment.")
                    genai_configured = True
                except Exception as config_err:
                    print(f"Warning: Found GOOGLE_API_KEY but failed to configure genai: {config_err}")
            elif getattr(genai, 'api_key', None): # Check if already configured somehow
                 print("DEBUG: genai appears to be already configured (api_key attribute exists).")
                 genai_configured = True

            if not genai_configured:
                 raise EnvironmentError("Google API Key (GOOGLE_API_KEY) not found or genai configuration failed. Document analysis requires API access.")

            with st.spinner("Waking up DocumentAlpha..."):
                # Instantiate DocumentAlpha - This calls DocumentAlpha's __init__
                doc_alpha_init_instance = DocumentAlpha()

            # Perform a basic check on the instance if needed (e.g., check for expected methods)
            if doc_alpha_init_instance and callable(getattr(doc_alpha_init_instance, 'chat_with_gemini', None)):
                st.session_state.doc_alpha_instance = doc_alpha_init_instance
                st.toast("DocumentAlpha initialized!", icon="📄")
                print("DEBUG: DocumentAlpha initialized successfully.")

                # Add initial message only if chat history is empty
                if not st.session_state.document_messages:
                     st.session_state.document_messages.append({
                         "role": "assistant", "content": "DocumentAlpha ready. Please upload a document using the sidebar to start."})
            else:
                raise RuntimeError("DocumentAlpha instance created but seems incomplete (e.g., missing 'chat_with_gemini' method?). Check DocumentAlpha's class definition.")

        except EnvironmentError as env_err: # Catch specific API key error
             print(f"ERROR: {env_err}", file=sys.stderr)
             st.session_state.doc_alpha_init_error_msg = f"Configuration Error: {env_err}"
             st.session_state.doc_alpha_instance = None
             if not st.session_state.document_messages:
                 st.session_state.document_messages = [{"role": "assistant", "content": f"Error: Could not initialize DocumentAlpha. {st.session_state.doc_alpha_init_error_msg}"}]

        except Exception as init_error:
            print(f"ERROR: Failed to initialize DocumentAlpha: {init_error}", file=sys.stderr)
            traceback.print_exc()
            # Store the runtime error message
            st.session_state.doc_alpha_init_error_msg = f"Runtime Error Initializing DocumentAlpha: {init_error}"
            st.session_state.doc_alpha_instance = None # Ensure it's None on failure
            # Add error message to chat if empty
            if not st.session_state.document_messages:
                 st.session_state.document_messages = [{"role": "assistant", "content": f"Error: Could not initialize DocumentAlpha. {st.session_state.doc_alpha_init_error_msg}"}]

initialize_assistants() # Call initialization function


# === 7. Setup Auto-Refresh and Check Reminders (If Alpha is Ready) ===
# Determine if Alpha is fully ready for reminder checking
is_alpha_ready_for_reminders = (
    alpha_available
    and st.session_state.alpha_instance is not None
    and callable(_load_memory_data)
    and callable(_save_memory_data)
    and MEMORY_FILE is not None # Ensure memory file path is defined
)

refresh_interval_seconds = 20 # Check every 20 seconds (adjust as needed)
if is_alpha_ready_for_reminders and st.session_state.app_mode == '🤖 General Assistant':
    # Only setup autorefresh if needed and in the right mode
    st_autorefresh(interval=refresh_interval_seconds * 1000, key="reminder_refresher")
    # print(f"DEBUG: Autorefresh active ({refresh_interval_seconds}s) for reminders.") # Can be noisy
    # Check reminders on *every* script run when Alpha is ready (includes refreshes and interactions)
    check_and_display_reminders()
# else:
#     if st.session_state.app_mode == '🤖 General Assistant': # Log only if in relevant mode
#         print("DEBUG: Autorefresh inactive (Alpha not ready, memory functions missing, or MEMORY_FILE not set).")


# === 8. Sidebar Rendering Logic ===

# --- Memory Overview and Calendar (for Alpha mode) ---
def display_sidebar_memory_overview():
    """Displays pending reminders and the memory calendar in the sidebar."""
    st.subheader("🧠 Alpha Memory")
    if not alpha_available or not st.session_state.alpha_instance or not callable(_load_memory_data) or not MEMORY_FILE:
        st.warning("Memory features unavailable.")
        if st.session_state.alpha_init_error_msg:
             st.error(f"Init Error: {st.session_state.alpha_init_error_msg}")
        elif not callable(_load_memory_data) or not callable(_save_memory_data):
             st.error("Memory loading/saving functions (`_load_memory_data`, `_save_memory_data`) not found or callable in `alpha1.py`.")
        elif not MEMORY_FILE:
            st.error("`MEMORY_FILE` path not defined in `alpha1.py`.")
        return

    try:
        memory_data = _load_memory_data()
        if not isinstance(memory_data, dict):
            st.error("Memory data is not in the expected dictionary format. Cannot display overview.")
            if MEMORY_FILE: print(f"ERROR: Invalid memory data structure loaded from {MEMORY_FILE}")
            return

        # --- Display Pending Reminders ---
        st.markdown("**🔔 Pending Reminders**")
        reminders_list = memory_data.get("Reminders", [])
        if isinstance(reminders_list, list):
            pending_reminders = [
                r for r in reminders_list
                if isinstance(r, dict) and r.get("status") == "pending" and r.get("reminder_time")
            ]
            if pending_reminders:
                try:
                    # Sort by reminder time (datetime objects), handle potential errors during conversion
                    def get_sort_key(r):
                        try: return datetime.strptime(r["reminder_time"], '%Y-%m-%d %H:%M:%S')
                        except (ValueError, KeyError): return datetime.max # Put invalid/missing times last
                    pending_reminders.sort(key=get_sort_key)

                except Exception as sort_e: # Broad catch for any sorting issue
                    print(f"Warning: Error sorting reminders: {sort_e}")
                    st.warning("Could not sort reminders by time.")

                # Display top N pending reminders
                max_reminders_to_show = 5
                for reminder in pending_reminders[:max_reminders_to_show]:
                    msg = reminder.get('message', 'No Message')[:40] # Truncate long messages
                    time_str = reminder.get('reminder_time', 'Invalid Time') # Already checked format
                    st.markdown(f"- `{time_str}`: {msg}{'...' if len(reminder.get('message', '')) > 40 else ''}")
                if len(pending_reminders) > max_reminders_to_show:
                    st.caption(f"... and {len(pending_reminders) - max_reminders_to_show} more.")
            else:
                st.caption("No pending reminders.")
        else:
            st.warning("Reminders data in memory is invalid (not a list).")
            # Attempt to fix memory structure?
            # memory_data['Reminders'] = []
            # _save_memory_data(memory_data)
        st.divider()

        # --- Calendar for Daily Tasks/Memories ---
        st.markdown("**📅 Memory Calendar**")
        schedule_data = memory_data.get("Schedule", {})
        calendar_events = []
        if isinstance(schedule_data, dict):
            for date_str, tasks in schedule_data.items():
                # Validate date format and ensure tasks exist
                if isinstance(tasks, dict) and tasks: # Ensure there are tasks for the date
                    try:
                        datetime.strptime(date_str, '%Y-%m-%d') # Validate date format strictly
                        calendar_events.append({
                            "title": f"🗒️ {len(tasks)}", # Show number of items
                            "start": date_str,       # Event covers the whole day
                            "end": date_str,
                            "color": "#3498db",      # Blue marker for days with entries
                            "allDay": True,          # Mark as background event for the day
                            # Optional: Add custom property to store date if needed elsewhere
                            # "extendedProps": {"date_str": date_str}
                        })
                    except ValueError:
                        print(f"Warning: Invalid date format '{date_str}' found in schedule data, skipping for calendar.")
        else:
            st.warning("Schedule data in memory is not a dictionary. Cannot display calendar events.")

        # Calendar Configuration
        calendar_options = {
            "headerToolbar": {
                "left": "prev,next today",
                "center": "title",
                "right": "dayGridMonth", # Keep it simple for overview
            },
            "initialView": "dayGridMonth",
            "selectable": True,         # Allow clicking on dates
            "selectMirror": True,       # Show placeholder on selection drag (visual feedback)
            "eventColor": '#808080',    # Default color (unused if event has own color)
            "height": "auto",          # Adjust height automatically (can clash with CSS)
            # Use CSS for fixed height if preferred: "height": "350px",
            "contentHeight": "auto",
            "editable": False,          # Don't allow dragging events
            "dayMaxEvents": True,       # Show "+more" link if too many events
            # "locale": "en",          # Optional: Set locale if needed
        }

        # Render the Calendar Component
        # It returns details about user interaction (like clicks)
        calendar_result = calendar(
            events=calendar_events,
            options=calendar_options,
            # IMPORTANT: Specify ONLY the callbacks you want to handle.
            # 'dateClick' is for clicking the background of a date cell.
            # 'eventClick' is for clicking an actual event marker.
            callbacks=["dateClick"],
            key="memory_calendar"       # Unique key is crucial for state preservation
        )

        # --- Handle Calendar Date Click ---
        clicked_date_str = None
        date_extracted = False

        # Check if the calendar returned a result and if it's the 'dateClick' callback
        if calendar_result and isinstance(calendar_result, dict) and calendar_result.get('callback') == 'dateClick':
            # Extract the date information from the callback data
            clicked_data = calendar_result.get('dateClick')
            if clicked_data and isinstance(clicked_data, dict):
                 # Prefer 'dateStr' as it's usually the clean 'YYYY-MM-DD'
                if 'dateStr' in clicked_data and isinstance(clicked_data['dateStr'], str):
                    raw_date_str = clicked_data['dateStr']
                    try:
                        # Validate the format before using
                        datetime.strptime(raw_date_str[:10], '%Y-%m-%d')
                        clicked_date_str = raw_date_str[:10] # Take only the date part
                        date_extracted = True
                        print(f"DEBUG: Extracted date via ['dateClick']['dateStr']: {clicked_date_str}")
                    except ValueError:
                        print(f"DEBUG: Value in ['dateClick']['dateStr'] ('{raw_date_str}') not in YYYY-MM-DD format.")

                # Fallback to 'date' if 'dateStr' is missing or invalid
                elif not date_extracted and 'date' in clicked_data and isinstance(clicked_data['date'], str):
                    raw_date = clicked_data['date']
                    # Handle potentially longer ISO strings like '2024-04-24T00:00:00.000Z'
                    if len(raw_date) >= 10:
                        potential_date = raw_date[:10]
                        try:
                            datetime.strptime(potential_date, '%Y-%m-%d')
                            clicked_date_str = potential_date
                            clicked_date_str = clicked_date_str[:-2] + str(int(clicked_date_str.split('-')[2])+1)
                            date_extracted = True
                            # Note: The previous fix adding +1 day seemed incorrect based on standard calendar behavior.
                            # dateClick usually returns the start of the clicked day.
                            # clicked_date_str = clicked_date_str[:-2] + str(int(clicked_date_str.split('-')[2])+1) # Removed this
                            print(f"DEBUG: Extracted date via ['dateClick']['date'] (fallback): {clicked_date_str}")
                        except ValueError:
                            print(f"DEBUG: Sliced date from ['dateClick']['date'] ('{potential_date}') is invalid format.")
                    else:
                        print(f"DEBUG: Value in ['dateClick']['date'] ('{raw_date}') is too short.")

                # Log if callback occurred but no valid date was extracted
                if not date_extracted:
                    print(f"DEBUG: ['dateClick'] received, but failed to extract valid YYYY-MM-DD from data: {clicked_data}")
            else:
                print(f"DEBUG: Calendar 'dateClick' callback structure invalid or missing data: {clicked_data}")

        # --- Update Session State and Rerun if a New Date Was Clicked ---
        if date_extracted and clicked_date_str:
            current_selected = st.session_state.get('selected_calendar_date')
            print(f"DEBUG: Date extracted: {clicked_date_str}. Current selection in state: {current_selected}")
            # Update state and rerun *only if* the clicked date is different from the current selection
            if current_selected != clicked_date_str:
                print(f"DEBUG: Updating selected_calendar_date from '{current_selected}' to '{clicked_date_str}' and triggering rerun...")
                st.session_state.selected_calendar_date = clicked_date_str
                st.rerun() # Force rerun to reflect the selection and update displayed activities
            else:
                # Avoid unnecessary reruns if the same date is clicked again
                print(f"DEBUG: Clicked date {clicked_date_str} is already selected. No state change or rerun triggered.")
        # If a dateClick happened but we failed to extract a date, we just log it (handled above)

        # --- Display Activities for the selected date ---
        st.divider() # Separator before showing activities
        display_date = st.session_state.get('selected_calendar_date')

        if display_date:
            print(f"DEBUG: Displaying activities for selected date: {display_date}")
            st.markdown(f"**Activities for {display_date}:**")
            # Retrieve activities safely, defaulting to an empty dict
            activities = schedule_data.get(display_date, {})

            if isinstance(activities, dict) and activities:
                 # Use a container with fixed height for scrollable list if many activities
                with st.container(height=150): # Adjust height as needed
                    # Sort items alphabetically by label (key) for consistent display
                    for label, value in sorted(activities.items()):
                        # Basic display, improve formatting as needed
                        display_value = str(value)
                        # Truncate long values for display
                        if len(display_value) > 100: display_value = display_value[:100] + '...'
                        # Format label nicely (replace underscores, title case)
                        formatted_label = label.replace('_', ' ').title()
                        st.markdown(f"- **{formatted_label}:** {display_value}")
            elif isinstance(activities, dict):
                st.caption(f"No activities recorded for {display_date}.")
            else:
                # This indicates potentially corrupt data in the schedule for this date
                st.warning(f"Activity data for {display_date} is invalid (expected a dictionary, found {type(activities)}).")
                print(f"ERROR: Invalid schedule data for date '{display_date}': {activities}")
        else:
             st.caption("Click a date on the calendar to view recorded activities for that day.")

    except FileNotFoundError:
        st.error(f"Memory file '{MEMORY_FILE}' not found. Cannot load schedule/reminders.")
    except json.JSONDecodeError:
         st.error(f"Error decoding memory file '{MEMORY_FILE}'. It might be corrupted. Check the file content.")
         if MEMORY_FILE: print(f"ERROR: JSONDecodeError loading {MEMORY_FILE}", file=sys.stderr)
    except Exception as e:
        st.error(f"An unexpected error occurred rendering the memory overview/calendar:")
        st.exception(e) # Show full traceback in the UI for easier debugging
        print("ERROR rendering memory overview/calendar:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


# --- Document Upload Sidebar Logic (for Document Analysis mode) ---
def display_sidebar_document_upload():
    """Handles file uploads and context management for DocumentAlpha."""
    st.subheader("📄 Document Upload & Context")

    # Check if DocumentAlpha is available and initialized
    if not doc_alpha_available:
        error_msg = st.session_state.doc_alpha_init_error_msg or 'Import failed (check DocumentAlpha.py).'
        st.error(f"Doc Analysis Unavailable: {error_msg}")
        return
    # Check for runtime initialization errors
    if st.session_state.doc_alpha_init_error_msg and not st.session_state.doc_alpha_instance:
         st.error(f"DocAlpha Init Error: {st.session_state.doc_alpha_init_error_msg}")
         return
    # Check if instance is simply not ready yet (should be handled by init logic, but good backup)
    if not st.session_state.doc_alpha_instance:
         st.info("Document uploader unavailable (DocumentAlpha not initialized or failed).")
         return
    # Check if necessary types/constants were imported correctly
    if File is None:
        st.error("Google AI File type not available (import failed). Document upload is disabled.")
        return
    if not TEMP_UPLOAD_DIR or not isinstance(TEMP_UPLOAD_DIR, str):
        st.error("Configuration Error: `TEMP_UPLOAD_DIR` is not correctly set in `DocumentAlpha.py`.")
        return

    # --- File Uploader Widget ---
    # Allowed types can be customized, e.g., ['pdf', 'txt', 'md', 'docx', 'png', 'jpg']
    # Check Gemini API documentation for currently supported file types for multimodal analysis.
    allowed_types = None # Allow any type for now, rely on Gemini API to handle/reject
    uploaded_file = st.file_uploader(
        "Upload a document for analysis",
        type=allowed_types,
        key="file_uploader", # Unique key for the widget
        accept_multiple_files=False, # Keep single file for simplicity; multi-file needs more complex state management
        help="Upload a file (PDF, TXT, images, etc.). Gemini API support varies."
    )

    # --- File Processing Logic ---
    # Triggered when a file is selected in the uploader widget
    if uploaded_file is not None:
        # Check if it's genuinely a *new* file upload event, compared to the one already processed
        # Also check the processing lock to prevent parallel processing on rapid reruns
        is_new_file_event = (st.session_state.current_file_display_name != uploaded_file.name)
        can_process = is_new_file_event and (st.session_state.file_being_processed != uploaded_file.name)

        if can_process:
            # Set lock *before* starting processing
            st.session_state.file_being_processed = uploaded_file.name
            print(f"\nDEBUG: New file detected: '{uploaded_file.name}'. Starting processing.")
            # Clear old context immediately before processing new file
            st.session_state.current_file_context = None
            st.session_state.current_file_display_name = None # Clear display name too

            with st.spinner(f"Processing '{uploaded_file.name}'... This may take time depending on file size and API speed."):
                temp_file_path = None
                gemini_file_object = None
                try:
                    # 1. Create TEMP_UPLOAD_DIR if it doesn't exist
                    if not os.path.exists(TEMP_UPLOAD_DIR):
                         os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
                         print(f"DEBUG: Created temporary upload directory: {TEMP_UPLOAD_DIR}")

                    # 2. Save uploaded file to a temporary location
                    # Create a unique-ish temp filename to avoid clashes
                    # Consider using tempfile module for more robust temp file creation if needed
                    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"upload_{int(time.time())}_{uploaded_file.name}")
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue()) # Get file content from memory
                    print(f"DEBUG: Temp file saved at '{temp_file_path}' (Size: {os.path.getsize(temp_file_path)} bytes)")

                    # 3. Upload to Google AI using genai.upload_file
                    print(f"DEBUG: Uploading '{temp_file_path}' to Google AI...")
                    # The genai configuration (API key) should have happened during DocumentAlpha init
                    # Perform the upload via SDK
                    # display_name is optional but helpful for identifying files in Google AI console
                    gemini_file_object = genai.upload_file(
                        path=temp_file_path,
                        display_name=uploaded_file.name,
                        # mime_type can often be inferred, but specifying can help:
                        # mime_type=uploaded_file.type
                    )
                    print(f"DEBUG: Google AI File upload initiated. API Name/ID: {gemini_file_object.name}, Display Name: {gemini_file_object.display_name}")

                    # --- 4. IMPORTANT: Wait for Google AI to finish processing ---
                    # This is crucial as the file can't be used until ACTIVE state.
                    file_state_timeout = 180 # Increased timeout to 3 minutes
                    poll_interval = 5       # Check every 5 seconds
                    start_time = time.time()
                    while True:
                         current_state = genai.get_file(gemini_file_object.name).state.name
                         print(f"DEBUG: Polling Google AI File state ({gemini_file_object.name}): {current_state}")
                         if current_state == "ACTIVE":
                             print(f"DEBUG: Google AI File is now ACTIVE (ID: {gemini_file_object.name}).")
                             break # Success! Exit the loop.
                         elif current_state == "FAILED":
                              st.error(f"❌ Google AI processing failed for file '{uploaded_file.name}'. State: {current_state}.")
                              raise RuntimeError(f"Google AI File processing failed. Final state: {current_state}")
                         elif current_state != "PROCESSING":
                             # Catch any other unexpected states
                              st.warning(f"❓ Unexpected Google AI file state: {current_state} for '{uploaded_file.name}'.")
                              # Decide whether to treat as error or continue polling? For now, treat as error.
                              raise RuntimeError(f"Google AI File entered unexpected state: {current_state}")

                         # Check timeout
                         if time.time() - start_time > file_state_timeout:
                              st.error(f"⏳ Google AI File processing timed out after {file_state_timeout}s for '{uploaded_file.name}'. Last state: {current_state}.")
                              # Attempt to delete the timed-out file? Optional.
                              # try: genai.delete_file(gemini_file_object.name); print("DEBUG: Deleted timed-out file from Google AI.")
                              # except Exception as del_err: print(f"Warning: Failed to delete timed-out file {gemini_file_object.name}: {del_err}")
                              raise TimeoutException(f"Google AI File processing timed out after {file_state_timeout}s.") # Use custom exception

                         # Wait before polling again
                         time.sleep(poll_interval)

                    # 5. Update session state with the *processed* and *active* file object
                    # Store as a list for potential future multi-file support
                    st.session_state.current_file_context = [gemini_file_object]
                    st.session_state.current_file_display_name = uploaded_file.name # Use original name for UI consistency
                    st.success(f"✅ Ready: '{uploaded_file.name}'")
                    print(f"DEBUG: Successfully processed and set file context for '{uploaded_file.name}'")

                    # Reset chat history for the new document to avoid confusion
                    st.session_state.document_messages = [{"role": "assistant", "content": f"File '{uploaded_file.name}' is loaded and ready. Ask me questions about its content."}]

                except (TimeoutException, RuntimeError, genai.types.google_types.PermissionDenied, genai.types.google_types.NotFoundError) as process_err:
                     # Catch specific known errors during processing/polling
                     st.error(f"❌ Error during file preparation: {process_err}")
                     print(f"ERROR during file preparation for '{uploaded_file.name}': {process_err}", file=sys.stderr)
                     # Ensure context is cleared on failure
                     st.session_state.current_file_context = None
                     st.session_state.current_file_display_name = None
                     # Add error message to the chat
                     st.session_state.document_messages.append({"role": "assistant", "content": f"Error processing '{uploaded_file.name}': {process_err}. Please check console logs, API key permissions, or try a different file."})
                     # If the file object was created but failed processing, try to delete it from Google AI
                     if gemini_file_object and gemini_file_object.name:
                         try:
                            print(f"DEBUG: Attempting to delete failed/problematic Google AI file: {gemini_file_object.name}")
                            # genai.delete_file(gemini_file_object.name) # Uncomment carefully - deletes file from Google!
                            print(f"INFO: Deletion of failed Google AI file '{gemini_file_object.name}' is currently disabled in code.")
                         except Exception as del_err:
                            print(f"Warning: Failed to delete Google AI file '{gemini_file_object.name}' after processing error: {del_err}")

                except Exception as e:
                    # Catch any other unexpected errors during the process
                    st.error(f"❌ An unexpected error occurred while processing file '{uploaded_file.name}': {e}")
                    print(f"ERROR processing file '{uploaded_file.name}':", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    # Ensure context is cleared
                    st.session_state.current_file_context = None
                    st.session_state.current_file_display_name = None
                    # Add generic error to chat
                    st.session_state.document_messages.append({"role": "assistant", "content": f"An unexpected error occurred while processing '{uploaded_file.name}'. Please check logs or try again."})

                finally:
                    # 6. Clean up the temporary file from local disk regardless of success/failure
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            print(f"DEBUG: Removed temp file: {temp_file_path}")
                        except Exception as e_rem:
                            print(f"Warning: Could not remove temp file '{temp_file_path}': {e_rem}")

                    # Release the processing lock *after* all operations (including cleanup)
                    if st.session_state.file_being_processed == uploaded_file.name:
                         st.session_state.file_being_processed = None
                         print(f"DEBUG: Released file processing lock for '{uploaded_file.name}'.")
                    # Rerun Streamlit to update the UI immediately after processing attempt
                    st.rerun()

    # --- Display Current Context or Clear Button ---
    # This part runs if no *new* file upload event was handled in this script run
    elif st.session_state.current_file_display_name:
         # Display the name of the currently loaded file
         st.markdown(f"**Current File Context:**")
         st.success(f"`{st.session_state.current_file_display_name}` is loaded and active.")

         # --- Button to Clear Context ---
         st.warning("Clearing context might permanently delete the file from Google AI storage if enabled.") # Add warning
         if st.button("Clear File Context & Chat", key="clear_context", help="Removes the current file context, clears the document chat history, and potentially deletes the file from Google AI."):
             file_to_delete_name = None
             file_to_delete_id = None
             file_display_name_cleared = st.session_state.current_file_display_name # Store name before clearing

             # --- Optional: Delete the corresponding file from Google AI Storage ---
             # !! WARNING: This is permanent deletion. Use with caution. !!
             # Consider adding a confirmation dialog if implementing deletion.
             DELETE_FROM_GOOGLE_AI = False # <<< SET TO True TO ENABLE DELETION >>>

             if DELETE_FROM_GOOGLE_AI and st.session_state.current_file_context and isinstance(st.session_state.current_file_context, list):
                 # Assuming single file context for now
                 if len(st.session_state.current_file_context) > 0 and File and isinstance(st.session_state.current_file_context[0], File):
                    file_obj = st.session_state.current_file_context[0]
                    # The unique ID is stored in the 'name' attribute (e.g., 'files/abc123xyz')
                    file_to_delete_id = getattr(file_obj, 'name', None)
                    file_to_delete_name = getattr(file_obj, 'display_name', file_display_name_cleared) # Use display name for messages

                    if file_to_delete_id:
                        try:
                            print(f"DEBUG: Attempting to delete Google AI file: ID {file_to_delete_id} (Display Name: {file_to_delete_name})")
                            # THE ACTUAL DELETION CALL:
                            genai.delete_file(file_to_delete_id)
                            print(f"DEBUG: Successfully deleted Google AI file: {file_to_delete_id}")
                            st.toast(f"Google AI file '{file_to_delete_name}' deleted.", icon="🗑️")
                        except Exception as del_err:
                            print(f"Warning: Failed to delete Google AI file {file_to_delete_id}: {del_err}")
                            st.warning(f"Could not delete file '{file_to_delete_name}' from Google AI storage: {del_err}", icon="⚠️")
                    else:
                         print("Warning: Could not get Google AI file ID to delete.")
             elif DELETE_FROM_GOOGLE_AI:
                  print("INFO: Google AI file deletion enabled, but no valid file context found to delete.")
             else:
                  print(f"INFO: File context for '{file_display_name_cleared}' cleared locally. Deletion from Google AI is disabled (DELETE_FROM_GOOGLE_AI is False).")
                  st.info("Note: File context cleared from app. File deletion on Google AI is currently disabled in the code.", icon="ℹ️")

             # Clear session state variables regardless of deletion success/failure
             st.session_state.current_file_context = None
             st.session_state.current_file_display_name = None
             # Clear document chat history as well
             st.session_state.document_messages = [{"role": "assistant", "content": "File context cleared. Upload a new document to begin analysis."}]
             print(f"DEBUG: File context for '{file_display_name_cleared}' cleared by user.")
             st.toast("File context and chat cleared!", icon="🧹")
             # Rerun to reflect the cleared state in the UI
             st.rerun()
    else:
         # If no file has been uploaded yet or context was cleared
         st.info("Upload a file using the button above to start document analysis.")


# === 9. Sidebar Definition ===
with st.sidebar:
    # Optional: Add a logo
    try:
        st.image("logo.png", width=100) # Assumes logo.png is in the same directory
    except Exception:
        st.write("Unified Assistant") # Fallback text if logo fails

    st.header("Assistant Controls")

    # --- Mode Selection ---
    # Use index based on current session state for consistent selection after reruns
    available_modes = ['🤖 General Assistant', '📄 Document Analysis']
    # Dynamically disable modes if their respective modules failed to import/init
    disabled_modes_indices = []
    if not alpha_available or st.session_state.alpha_init_error_msg:
        disabled_modes_indices.append(0)
    if not doc_alpha_available or st.session_state.doc_alpha_init_error_msg:
         disabled_modes_indices.append(1)

    # Determine current mode index, default to 0 if invalid
    try:
        current_mode_index = available_modes.index(st.session_state.app_mode)
    except ValueError:
        current_mode_index = 0 # Default to first mode if current state is invalid
        st.session_state.app_mode = available_modes[0] # Reset state

    app_mode = st.radio(
        "Select Mode:",
        options=available_modes,
        key='app_mode_radio',       # Unique key for the radio widget
        index=current_mode_index,
        horizontal=True,           # Display options side-by-side if space allows
        help="Switch between the general chatbot and the document analysis tool.",
        # Disable options based on availability (Streamlit > 1.28?)
        # disabled=disabled_modes_indices # This needs newer Streamlit versions
        # Manual disabling (show message if trying to select disabled)
        # Note: Radio button disabling isn't directly supported well across all versions
    )

    # Display warning if a mode is selected but unavailable
    if app_mode == '🤖 General Assistant' and not alpha_available:
         st.warning("General Assistant module (alpha1.py) failed to load.", icon="⚠️")
    if app_mode == '📄 Document Analysis' and not doc_alpha_available:
         st.warning("Document Analysis module (DocumentAlpha.py) failed to load.", icon="⚠️")
    if app_mode == '📄 Document Analysis' and doc_alpha_available and st.session_state.doc_alpha_init_error_msg:
         st.warning(f"Document Analysis init failed: {st.session_state.doc_alpha_init_error_msg}", icon="⚠️")

    # Update app_mode in session state *only if* the radio button value has changed
    if st.session_state.app_mode != app_mode:
        print(f"DEBUG: Mode changed from '{st.session_state.app_mode}' to '{app_mode}' via radio button. Rerunning.")
        st.session_state.app_mode = app_mode
        # Optional: Clear states specific to the *previous* mode when switching?
        # e.g., clear calendar selection when switching away from General mode
        # if app_mode == '📄 Document Analysis':
        #    st.session_state.selected_calendar_date = None
        # e.g., clear file context when switching away from Document mode? (Maybe not desired)
        # if app_mode == '🤖 General Assistant':
        #    # Decide whether to clear st.session_state.current_file_context etc.
        st.rerun() # Rerun immediately to load the correct UI and sidebar sections

    st.divider() # Visual separator

    # --- Conditional Sidebar Content Based on Mode ---
    if st.session_state.app_mode == '🤖 General Assistant':
        # Display memory overview (reminders, calendar) if Alpha is available
        if alpha_available:
            display_sidebar_memory_overview()
        else:
            st.error("General Assistant module not loaded. Memory features unavailable.")
        st.divider()
        st.subheader("Chat Controls")
        if st.button("Clear General Chat", key="clear_general", help="Clears the chat history for the General Assistant."):
            # Reset chat history, keeping an initial message
            st.session_state.general_messages = [{"role": "assistant", "content": "Chat history cleared. How can I help?"}]
            # Also clear the calendar selection when clearing chat
            st.session_state.selected_calendar_date = None
            st.toast("General chat history cleared!", icon="🧹")
            print("DEBUG: General chat cleared.")
            st.rerun()

    elif st.session_state.app_mode == '📄 Document Analysis':
        # Display document uploader and context manager if DocAlpha is available
        if doc_alpha_available:
            display_sidebar_document_upload() # This function handles its own internal error states
        else:
             st.error("Document Analysis module not loaded. Upload features unavailable.")
        st.divider()
        st.subheader("Chat Controls")
        # Button to clear only the chat history for Document Analysis
        # The button to clear context *and* chat is within display_sidebar_document_upload
        if st.button("Clear Document Chat Only", key="clear_document_chat", help="Clears only the chat history for Document Analysis, keeping the loaded file context."):
             # Keep an introductory message based on whether a file is currently loaded
             if st.session_state.current_file_display_name:
                  st.session_state.document_messages = [{"role": "assistant", "content": f"Document chat history cleared. File '{st.session_state.current_file_display_name}' is still loaded. Ask questions or clear the file context using the button above."}]
             else:
                  st.session_state.document_messages = [{"role": "assistant", "content": "Document chat history cleared. Upload a document to begin."}]
             st.toast("Document chat history cleared!", icon="🧹")
             print("DEBUG: Document chat cleared (context kept).")
             st.rerun()


# === 10. Main Area Rendering Logic ===

# --- General Assistant UI ---
def render_general_assistant_ui():
    """Renders the chat interface and controls for the General Assistant."""
    st.title("🤖 General Assistant (Alpha)")

    # --- Status Bar ---
    is_alpha_ready = alpha_available and st.session_state.alpha_instance is not None
    # Determine status text and color based on availability and initialization state
    if is_alpha_ready:
        status_color = "green"
        status_text = "Ready"
    elif alpha_available and st.session_state.alpha_init_error_msg:
        status_color = "orange"
        status_text = "Initialization Error"
    elif not alpha_available and alpha_init_error: # Import failed
        status_color = "red"
        status_text = "Import Failed"
    else: # Catch-all for other unavailable states
        status_color = "red"
        status_text = "Unavailable"

    model_name_display = st.session_state.get('alpha_model_name', 'N/A')
    # Reminder status depends on Alpha *and* memory functions/file
    if is_alpha_ready_for_reminders:
        reminder_status = 'Active'
    elif is_alpha_ready: # Alpha ready, but memory functions/file missing
        reminder_status = 'Inactive (Memory Error)'
    else: # Alpha itself not ready
        reminder_status = 'Inactive (Alpha Not Ready)'

    st.caption(f"Status: :{status_color}[{status_text}] | Model: `{model_name_display}` | Reminders: `{reminder_status}`")

    # --- Display Initialization Errors prominently if Alpha isn't ready ---
    if not is_alpha_ready:
        init_err = st.session_state.alpha_init_error_msg or alpha_init_error or "Assistant module (alpha1.py) could not be loaded or initialized."
        st.error(f"Alpha Assistant is currently unavailable. Reason: {init_err}")
        # Provide more specific guidance based on common errors
        if "FileNotFoundError" in str(init_err):
             st.warning(f"Action: Ensure `alpha1.py` exists in the directory `{current_dir}`.")
        elif "ImportError" in str(init_err) or "ModuleNotFoundError" in str(init_err):
             st.warning("Action: Check `alpha1.py` for the required `Alpha` class, helper functions (`_load_memory_data`, `_save_memory_data`), `MEMORY_FILE` constant. Ensure all required Python packages (like `google-generativeai`, potentially `schedule`) are installed (`pip install -r requirements.txt`).")
        elif "Runtime Error" in str(init_err):
             st.warning("Action: An error occurred during Alpha's initialization (`__init__`). Check the console logs (where you ran `streamlit run app.py`) for detailed tracebacks.")
        return # Stop rendering the chat UI if Alpha isn't ready

    st.divider() # Separator before chat messages

    # --- Chat Message Display ---
    # Use a container to potentially limit height and make it scrollable if it gets very long
    # chat_container = st.container(height=500) # Example: Limit height
    # with chat_container:
    if not st.session_state.general_messages:
         st.info("Start chatting with Alpha below!") # Show if history is empty
    else:
        for message in st.session_state.general_messages:
            role = message.get("role", "assistant") # Default to assistant if role missing
            content = message.get("content", "[empty message]")
            avatar = "👤" if role == "user" else "🤖"
            try:
                with st.chat_message(role, avatar=avatar):
                    st.markdown(content) # Render markdown for potential formatting
            except Exception as display_err:
                 print(f"Error displaying chat message: {display_err} - Message: {message}")
                 st.error(f"Error displaying a message: {display_err}")

    # --- Chat Input ---
    prompt_alpha = st.chat_input(
        "Ask Alpha anything (e.g., 'set a reminder for 10pm', 'what's the weather?')...",
        key="general_chat_input",
        disabled=not is_alpha_ready # Disable input if Alpha isn't ready
    )

    if prompt_alpha:
        # Append user message to state immediately
        st.session_state.general_messages.append({"role": "user", "content": prompt_alpha})
        # Display user message (will be shown on the automatic rerun)
        # No need to manually display here, rerun handles it

        # --- Process Input ---
        # Simple exit condition check (optional)
        if prompt_alpha.lower().strip() in ["exit", "quit", "bye", "stop", "goodbye", "0"]:
            response = "Okay, goodbye! Let me know if you need anything else."
            st.session_state.general_messages.append({"role": "assistant", "content": response})
            # Rerun to show the goodbye message
            st.rerun()
        else:
            # Process the input with the Alpha instance
            try:
                ai = st.session_state.alpha_instance # Get the initialized instance
                with st.spinner("Alpha is thinking... 🤔"):
                    # Ensure prompt is a non-empty string and stripped of whitespace
                    clean_prompt = str(prompt_alpha).strip()
                    if not clean_prompt:
                         ai_response = "Please provide some input." # Handle empty input case
                    else:
                         # Call the backend chat function in alpha1.py
                         # This function should handle the actual interaction with the Gemini API
                         # and potentially update memory (reminders/schedule)
                         print(f"DEBUG: Sending to Alpha.chat_with_gemini: '{clean_prompt}'")
                         ai_response = ai.chat_with_gemini(clean_prompt)
                         print(f"DEBUG: Received from Alpha.chat_with_gemini: '{ai_response}'")

                # Handle potential None or unexpected response types from the backend
                if ai_response is None:
                     # The backend might have performed an action (like setting reminder) without a text response
                     ai_response_str = "[Alpha processed the request (no text response)]"
                     st.warning("Alpha processed the request but didn't provide a text reply. Check memory or ask again if needed.")
                else:
                    ai_response_str = str(ai_response) # Ensure it's a string

                # Append assistant's response to history
                st.session_state.general_messages.append({"role": "assistant", "content": ai_response_str})

            except Exception as chat_error:
                print(f"ERROR during Alpha chat interaction: {chat_error}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                error_message = f"Sorry, an error occurred while processing your request with Alpha: ({type(chat_error).__name__}). Please check the console logs for details or try phrasing your request differently."
                st.session_state.general_messages.append({"role": "assistant", "content": error_message})
                # Display error immediately in the chat area for user feedback (optional, as rerun will show it too)
                # st.error(f"An error occurred interacting with Alpha: {chat_error}")
            finally:
                 # Rerun to display the new user message and the assistant's response/error
                 st.rerun()


# --- Document Analysis UI ---
def render_document_analysis_ui():
    """Renders the chat interface and controls for Document Analysis."""
    st.title("📄 Document Analysis (DocumentAlpha)")

    # --- Status Bar ---
    is_doc_alpha_ready = doc_alpha_available and st.session_state.doc_alpha_instance is not None
    # Determine status text and color
    if is_doc_alpha_ready:
        status_color = "green"
        status_text = "Ready"
    elif doc_alpha_available and st.session_state.doc_alpha_init_error_msg:
        status_color = "orange"
        status_text = "Initialization Error"
    elif not doc_alpha_available and doc_alpha_init_error: # Import failed
        status_color = "red"
        status_text = "Import Failed"
    else: # Catch-all unavailable
        status_color = "red"
        status_text = "Unavailable"

    # File context status
    file_context_obj_list = st.session_state.get('current_file_context') # Expecting a list
    file_display_name = st.session_state.get('current_file_display_name')
    file_status = "File Context: None Loaded"
    if file_display_name and file_context_obj_list and isinstance(file_context_obj_list, list) and len(file_context_obj_list) > 0:
        # Optional: Could add a check here to re-verify the file state via genai.get_file if needed, but might be slow.
        # file_state = getattr(file_context_obj_list[0], 'state', None) # Assuming single file
        # file_state_name = getattr(file_state, 'name', 'UNKNOWN')
        # file_status = f"File Context: `{file_display_name}` (State: {file_state_name})" # More detailed status
        file_status = f"File Context: `{file_display_name}`" # Simplified status
    elif file_display_name and not file_context_obj_list:
         # This might happen if upload failed after filename was set, or context cleared unexpectedly
         file_status = f"File Context: Error with '{file_display_name}' (Context missing)"
    elif st.session_state.file_being_processed:
        # Show processing status if a file upload is currently underway
        file_status = f"File Context: Processing '{st.session_state.file_being_processed}'..."

    st.caption(f"Status: :{status_color}[{status_text}] | {file_status}")

    # --- Display Initialization Errors ---
    if not is_doc_alpha_ready:
        init_err = st.session_state.doc_alpha_init_error_msg or doc_alpha_init_error or "Document Analysis module (DocumentAlpha.py) could not be loaded or initialized."
        st.error(f"Document Analysis Assistant is unavailable. Reason: {init_err}")
        # Provide guidance
        if "FileNotFoundError" in str(init_err):
             st.warning(f"Action: Ensure `DocumentAlpha.py` exists in the directory `{current_dir}`.")
        elif "ImportError" in str(init_err) or "ModuleNotFoundError" in str(init_err):
             st.warning("Action: Check `DocumentAlpha.py` for the `DocumentAlpha` class, `TEMP_UPLOAD_DIR`, `TimeoutException`. Ensure `google-generativeai` is installed. Check API key setup.")
        elif "EnvironmentError" in str(init_err) or "GOOGLE_API_KEY" in str(init_err):
            st.warning("Action: Ensure the `GOOGLE_API_KEY` environment variable is set correctly *before* starting Streamlit, and the key is valid with the Generative Language API enabled.")
        elif "Runtime Error" in str(init_err):
             st.warning("Action: An error occurred during DocumentAlpha's initialization (`__init__`). Check console logs for details.")
        return # Stop rendering chat UI

    st.divider()

    # --- Chat Message Display ---
    # chat_container_doc = st.container(height=500) # Optional height limit
    # with chat_container_doc:
    if not st.session_state.document_messages:
         if st.session_state.current_file_display_name:
              st.info("Ask questions about the loaded document below.")
         else:
              st.info("Upload a document using the sidebar to begin analysis.")
    else:
        for message in st.session_state.document_messages:
            role = message.get("role", "assistant")
            content = message.get("content", "[empty message]")
            avatar = "👤" if role == "user" else "📄" # Document icon for assistant
            try:
                with st.chat_message(role, avatar=avatar):
                    st.markdown(content)
            except Exception as display_err:
                 print(f"Error displaying document chat message: {display_err} - Message: {message}")
                 st.error(f"Error displaying a message: {display_err}")

    # --- Chat Input ---
    # Input should be disabled if DocAlpha isn't ready OR if no valid file context is loaded
    is_input_disabled = (
        not is_doc_alpha_ready
        or not st.session_state.current_file_context
        or not isinstance(st.session_state.current_file_context, list)
        or len(st.session_state.current_file_context) == 0
        or st.session_state.file_being_processed is not None # Disable while processing
    )
    input_placeholder = "Ask about the document..." if st.session_state.current_file_context else "Upload a document via the sidebar to enable chat"
    if st.session_state.file_being_processed:
        input_placeholder = f"Processing '{st.session_state.file_being_processed}'..." # Show processing state in placeholder

    prompt_doc = st.chat_input(
        input_placeholder,
        key="document_chat_input",
        disabled=is_input_disabled
    )

    if prompt_doc:
        # This block should only be reached if input wasn't disabled, implying context exists and is ready
        if not st.session_state.current_file_context or not isinstance(st.session_state.current_file_context, list) or len(st.session_state.current_file_context) == 0:
             # This is a safeguard, should ideally not happen if disabled logic is correct
             st.warning("Cannot process request: No valid document context is currently loaded.", icon="⚠️")
        else:
            # Add user message to state
            st.session_state.document_messages.append({"role": "user", "content": prompt_doc})
            # Display user message immediately (will be shown on rerun anyway)
            # Let rerun handle display

            # --- Get Assistant Response ---
            with st.spinner("DocumentAlpha is analyzing... 🧐"):
                try:
                    # 1. Get the current file context from session state (should be a list of File objects)
                    file_context_list = st.session_state.get('current_file_context')

                    # 2. Validate the context format again (robustness check)
                    if not isinstance(file_context_list, list) or len(file_context_list) == 0:
                        raise ValueError("Document context is missing or empty.")
                    # Check if File type exists and if items are of that type (if File imported correctly)
                    if File and not all(isinstance(f, File) for f in file_context_list):
                        raise TypeError("Invalid item type found in file context list. Expected Google AI File objects.")
                    # Context seems valid (it's a non-empty list, potentially containing File objects)
                    valid_context = file_context_list # Pass the list

                    # 3. Get the DocumentAlpha instance
                    doc_ai = st.session_state.doc_alpha_instance
                    if not doc_ai:
                        raise RuntimeError("DocumentAlpha instance not found in session state.")

                    # 4. Call the chat method with the prompt and validated context list
                    clean_prompt = str(prompt_doc).strip()
                    if not clean_prompt:
                         response = "Please ask a specific question about the document."
                    else:
                         print(f"DEBUG: Sending to DocumentAlpha.chat_with_gemini: '{clean_prompt}' with context: {[getattr(f, 'name', 'Unknown File') for f in valid_context]}")
                         # The backend function should handle the actual API call with the file(s)
                         response = doc_ai.chat_with_gemini(clean_prompt, file_context=valid_context)
                         print(f"DEBUG: Received from DocumentAlpha.chat_with_gemini: '{response}'")

                    # 5. Process response
                    if response is None:
                        response_text = "[DocumentAlpha provided no text response]"
                        st.warning("DocumentAlpha processed the request but didn't return a text reply.")
                    else:
                         response_text = str(response) # Ensure string conversion

                    # Append response to message history
                    st.session_state.document_messages.append({"role": "assistant", "content": response_text})

                except (TimeoutException, genai.types.google_types.DeadlineExceeded) as e:
                     error_msg = f"⚠️ Analysis timed out: {e}. The document might be too complex, or the request took too long. Try asking something simpler or breaking down the query."
                     st.session_state.document_messages.append({"role": "assistant", "content": error_msg})
                     print(f"ERROR during DocumentAlpha chat (Timeout): {e}", file=sys.stderr)
                except (TypeError, ValueError, RuntimeError) as context_err: # Catch context validation or runtime errors
                     error_msg = f"An internal error occurred: {context_err}. If related to context, try clearing and re-uploading the file."
                     st.session_state.document_messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                     print(f"ERROR during DocumentAlpha chat (Internal/Context): {context_err}", file=sys.stderr)
                     traceback.print_exc(file=sys.stderr)
                except Exception as e:
                    # Catch-all for other unexpected errors during chat
                    print(f"ERROR during DocumentAlpha chat interaction: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    error_msg = f"An unexpected error occurred during analysis: ({type(e).__name__}). Please check the console logs or try again."
                    st.session_state.document_messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                finally:
                    # Rerun to display user message and assistant response/error
                    st.rerun()


# === 11. Main App Logic ===
# Select which UI function to call based on the current mode stored in session state
try:
    if st.session_state.app_mode == '🤖 General Assistant':
        render_general_assistant_ui()
    elif st.session_state.app_mode == '📄 Document Analysis':
        render_document_analysis_ui()
    else:
        # Fallback in case app_mode state gets corrupted somehow
        st.error(f"Error: Invalid application mode '{st.session_state.app_mode}' found in session state. Resetting to default.")
        print(f"ERROR: Invalid app_mode '{st.session_state.app_mode}', resetting.")
        st.session_state.app_mode = '🤖 General Assistant' # Reset to default
        time.sleep(0.5) # Brief pause before rerunning
        st.rerun()
except Exception as main_render_err:
     # Catch unexpected errors in the main rendering logic itself
     st.error("A critical error occurred in the main application rendering.")
     st.exception(main_render_err)
     print("FATAL ERROR in main rendering logic:", file=sys.stderr)
     traceback.print_exc(file=sys.stderr)

# === End of app.py ===