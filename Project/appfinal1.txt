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
import base64 # <--- IMPORT ADDED

# Check if File type exists, handle potential import variations
try:
    from google.generativeai.types import File
except ImportError:
    # Handle older versions or potential future changes if necessary
    print("Warning: google.generativeai.types.File not found. Document upload might be affected.")
    File = None # Placeholder

from typing import Optional, List

# --- Add Calendar Import ---
from streamlit_calendar import calendar # Import the calendar component

# === 1. Streamlit Page Configuration ===
st.set_page_config(
    page_title="Unified Assistant",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Title will be set dynamically based on mode

# === 2. Locate and Import Helper Modules ===
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- Attempt to Import Alpha (General Assistant) ---
alpha_file_path = os.path.join(current_dir, 'alpha1.py')
alpha_available = False
alpha_init_error = None
Alpha, _load_memory_data, _save_memory_data, MEMORY_FILE = None, None, None, None
try:
    if not os.path.isfile(alpha_file_path):
        raise FileNotFoundError(f"'alpha1.py' not found in the directory: {current_dir}")
    from alpha1 import Alpha, _load_memory_data, _save_memory_data, MEMORY_FILE
    alpha_available = True
    try: import schedule # Optional dependency within alpha1?
    except ImportError: pass
except (ImportError, FileNotFoundError, ModuleNotFoundError) as e:
    alpha_init_error = f"Failed to import `Alpha` or helpers from `alpha1.py`. Reason: `{e}`."
except Exception as e:
    alpha_init_error = f"An unexpected error occurred importing from `alpha1.py`: {e}"
    traceback.print_exc()

# --- Attempt to Import DocumentAlpha (Document Analysis) ---
doc_alpha_file_path = os.path.join(current_dir, 'DocumentAlpha.py')
doc_alpha_available = False
doc_alpha_init_error = None
DocumentAlpha, TEMP_UPLOAD_DIR, TimeoutException = None, None, None
try:
    if not os.path.isfile(doc_alpha_file_path):
        raise FileNotFoundError(f"'DocumentAlpha.py' not found in the directory: {current_dir}")
    from DocumentAlpha import DocumentAlpha, TEMP_UPLOAD_DIR, TimeoutException
    doc_alpha_available = True
except (ImportError, FileNotFoundError, ModuleNotFoundError) as e:
    doc_alpha_init_error = f"Failed to import `DocumentAlpha` from `DocumentAlpha.py`. Reason: `{e}`."
except Exception as e:
    doc_alpha_init_error = f"An unexpected error occurred importing from `DocumentAlpha.py`: {e}"
    traceback.print_exc()

# === 3. Styling (Optional) ===
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 1rem;
        border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
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

    /* Hide the default audio player controls if desired (optional) */
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
ALARM_SOUND_FILE = "alarm.wav" # MODIFY THIS if your file name is different
# -----------------------------------

def play_alarm_sound():
    """Reads, encodes, and embeds the alarm sound for autoplay."""
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
            mime_type = "audio/mpeg" # Default guess
            print(f"Warning: Unknown audio file extension '{file_ext}'. Assuming 'audio/mpeg'.")

        sound_data_uri = f"data:{mime_type};base64,{sound_b64}"

        # Embed HTML5 audio player with autoplay
        # Note: Autoplay may be blocked by browser policies until user interacts with the page.
        st.markdown(
            f'<audio autoplay="true" src="{sound_data_uri}"></audio>',
            unsafe_allow_html=True,
        )
        print(f"DEBUG: Attempting to play alarm sound ({ALARM_SOUND_FILE})") # Log attempt
    except FileNotFoundError:
        # This case should be caught by os.path.exists above, but kept for safety
        print(f"ERROR: Alarm sound file '{ALARM_SOUND_FILE}' not found at path: {sound_file_path}")
        st.error(f"Error finding alarm sound file: {ALARM_SOUND_FILE}")
    except Exception as audio_e:
        print(f"ERROR: Failed to load or play alarm sound: {audio_e}")
        traceback.print_exc()
        st.error(f"Error playing alarm sound: {audio_e}")


def check_and_display_reminders():
    if not alpha_available or not callable(_load_memory_data) or not callable(_save_memory_data):
        # print("DEBUG (Reminder Check): Alpha or memory functions unavailable.") # Too noisy for console
        return
    try:
        now = datetime.now()
        data = _load_memory_data()
        reminders = data.get("Reminders", [])
        if not isinstance(reminders, list):
            print("Warning: 'Reminders' in memory is not a list.")
            return # Avoid processing invalid data

        reminders_updated = False
        played_sound_this_run = False # Flag to prevent multiple sounds in one refresh cycle
        processed_ids_this_run = set() # Track reminders processed in this specific run

        for index, reminder in enumerate(reminders):
            try:
                reminder_id = reminder.get('id', f"index_{index}") # Use index as fallback ID
                if reminder_id in processed_ids_this_run: continue # Skip if already processed this run

                # Basic validation: Ensure it's a dict and has 'pending' status
                if not isinstance(reminder, dict) or reminder.get("status") != "pending":
                     processed_ids_this_run.add(reminder_id); continue # Mark as processed (not pending)

                reminder_time_str = reminder.get("reminder_time")
                if not reminder_time_str:
                    if reminder.get('status') != 'error_missing_time':
                       reminder['status'] = 'error_missing_time'; reminders_updated = True
                    processed_ids_this_run.add(reminder_id); continue

                # Validate and parse reminder time
                try:
                    reminder_dt = datetime.strptime(reminder_time_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    print(f"Warning: Invalid date format '{reminder_time_str}' for reminder ID {reminder_id}")
                    if reminder.get('status') != 'error_processing':
                         reminder['status'] = 'error_processing'; reminders_updated = True
                    processed_ids_this_run.add(reminder_id); continue

                # --- Trigger Condition ---
                if reminder_dt <= now:
                    message = reminder.get('message', 'Reminder!')
                    print(f"DEBUG: Triggering reminder: {reminder_id} - {message}")
                    st.toast(f"🔔 Reminder: {message}", icon="⏰")

                    # --- Play Sound and Update Status (Only Once per reminder) ---
                    # The 'triggered' status prevents re-triggering on subsequent checks
                    if reminder.get('status') != 'triggered':
                        reminder['status'] = 'triggered'
                        reminders_updated = True
                        # Play the alarm sound only if not already played during this check cycle
                        if not played_sound_this_run:
                            play_alarm_sound()
                            played_sound_this_run = True # Set flag for this cycle

                    processed_ids_this_run.add(reminder_id) # Mark as processed for this run

            except Exception as e:
                reminder_id_err = reminder.get('id', f"error_index_{index}") if isinstance(reminder, dict) else f"error_index_{index}"
                print(f"ERROR: Unexpected error checking reminder {reminder_id_err}: {e}")
                traceback.print_exc() # Log the full traceback for debugging
                # Update status to avoid repeated errors on the same reminder (if possible)
                if isinstance(reminder, dict) and reminder.get('status') != 'error_unexpected':
                    try:
                         reminder['status'] = 'error_unexpected'; reminders_updated = True
                    except Exception: pass # Ignore if reminder object itself is malformed
                processed_ids_this_run.add(reminder_id_err)

        # Save changes back to memory file if any reminder statuses were updated
        if reminders_updated:
            if not _save_memory_data(data):
                print("ERROR (autorefresh): _save_memory_data failed after updating reminder statuses.")
            else:
                print("DEBUG: Saved updated reminder statuses to memory.")
    except Exception as e:
        print(f"ERROR (autorefresh): General failure in check_and_display_reminders: {e}")
        traceback.print_exc()


# === 5. Initialize Session State ===
def initialize_session_state():
    # --- App Wide ---
    if 'app_mode' not in st.session_state: st.session_state.app_mode = '🤖 General Assistant'

    # --- Alpha (General Assistant) ---
    if 'alpha_instance' not in st.session_state: st.session_state.alpha_instance = None
    if 'alpha_model_name' not in st.session_state: st.session_state.alpha_model_name = "Unavailable"
    if 'alpha_init_done' not in st.session_state: st.session_state.alpha_init_done = False
    if 'alpha_init_error_msg' not in st.session_state: st.session_state.alpha_init_error_msg = alpha_init_error
    if 'general_messages' not in st.session_state: st.session_state.general_messages = []
    # Calendar state for Alpha mode
    if 'selected_calendar_date' not in st.session_state: st.session_state.selected_calendar_date = None

    # --- DocumentAlpha ---
    if 'doc_alpha_instance' not in st.session_state: st.session_state.doc_alpha_instance = None
    if 'doc_alpha_init_done' not in st.session_state: st.session_state.doc_alpha_init_done = False
    if 'doc_alpha_init_error_msg' not in st.session_state: st.session_state.doc_alpha_init_error_msg = doc_alpha_init_error
    if 'document_messages' not in st.session_state: st.session_state.document_messages = []
    if 'current_file_context' not in st.session_state: st.session_state.current_file_context = None # Holds the Google AI File object(s)
    if 'current_file_display_name' not in st.session_state: st.session_state.current_file_display_name = None # For display in UI

initialize_session_state()


# === 6. Initialize Assistants (Run only once per session) ===
def initialize_assistants():
    # Initialize Alpha (General Assistant)
    if alpha_available and not st.session_state.alpha_init_done:
        st.session_state.alpha_init_done = True # Mark as attempted
        try:
            print("DEBUG: Attempting to initialize Alpha...")
            with st.spinner("Waking up Alpha (General Assistant)..."):
                # Check/Create Memory File (if paths are valid)
                if callable(_load_memory_data) and callable(_save_memory_data):
                    initial_data = _load_memory_data() # Load first to see if it exists/is valid
                    if MEMORY_FILE and not os.path.exists(MEMORY_FILE):
                        print(f"DEBUG (App Init): Memory file '{MEMORY_FILE}' not found, saving default structure.")
                        if not _save_memory_data(initial_data or {"Reminders": [], "Schedule": {}}): # Ensure default structure
                            st.warning("Could not create initial memory file for Alpha.")
                    elif not MEMORY_FILE:
                         st.warning("MEMORY_FILE path not available in alpha1.py. Cannot check/create memory file.")
                else:
                    st.warning("Alpha memory functions (_load/_save) not available. Reminders/Schedule might not persist.")

                # Instantiate Alpha
                alpha_init_instance = Alpha()

            if alpha_init_instance and hasattr(alpha_init_instance, 'model'): # Basic check
                st.session_state.alpha_instance = alpha_init_instance
                st.toast("Alpha (General Assistant) initialized!", icon="🤖")
                print("DEBUG: Alpha initialized successfully.")
                # Get model name safely
                try:
                    model_obj = getattr(st.session_state.alpha_instance, 'model', None)
                    # Gemini models often store name in _model_name or similar private attr
                    full_model_path = getattr(model_obj, '_model_name', "Unknown Model")
                    st.session_state.alpha_model_name = full_model_path.split('/')[-1] # Get last part
                except Exception as e:
                    print(f"Warning: Could not retrieve Alpha model name: {e}")
                    st.session_state.alpha_model_name = "Model name N/A"

                # Add initial message only if chat history is empty
                if not st.session_state.general_messages:
                     st.session_state.general_messages.append({
                        "role": "assistant",
                        "content": "Hello! I'm Alpha. How can I help you today? Ask questions, set reminders, or use the calendar."})
            else:
                # Instance created but seems incomplete or invalid
                raise RuntimeError("Alpha instance created but appears incomplete (missing 'model' attribute?).")

        except Exception as init_error:
            print(f"ERROR: Failed to initialize Alpha: {init_error}", file=sys.stderr)
            traceback.print_exc()
            st.session_state.alpha_init_error_msg = f"Runtime Error Initializing Alpha: {init_error}"
            st.session_state.alpha_instance = None # Ensure it's None on failure
            st.session_state.alpha_model_name = "Initialization Failed"
            # Add error message to chat if empty
            if not st.session_state.general_messages:
                 st.session_state.general_messages = [{"role": "assistant", "content": f"Error: Could not initialize Alpha. {st.session_state.alpha_init_error_msg}"}]

    # Initialize DocumentAlpha
    if doc_alpha_available and not st.session_state.doc_alpha_init_done:
        st.session_state.doc_alpha_init_done = True # Mark as attempted
        try:
            print("DEBUG: Attempting to initialize DocumentAlpha...")
            with st.spinner("Waking up DocumentAlpha..."):
                # Instantiate DocumentAlpha
                doc_alpha_init_instance = DocumentAlpha()

            st.session_state.doc_alpha_instance = doc_alpha_init_instance
            st.toast("DocumentAlpha initialized!", icon="📄")
            print("DEBUG: DocumentAlpha initialized successfully.")
            # Add initial message only if chat history is empty
            if not st.session_state.document_messages:
                 st.session_state.document_messages.append({
                     "role": "assistant", "content": "DocumentAlpha ready. Please upload a document using the sidebar."})

        except Exception as init_error:
            print(f"ERROR: Failed to initialize DocumentAlpha: {init_error}", file=sys.stderr)
            traceback.print_exc()
            st.session_state.doc_alpha_init_error_msg = f"Runtime Error Initializing DocumentAlpha: {init_error}"
            st.session_state.doc_alpha_instance = None # Ensure it's None on failure
            # Add error message to chat if empty
            if not st.session_state.document_messages:
                 st.session_state.document_messages = [{"role": "assistant", "content": f"Error: Could not initialize DocumentAlpha. {st.session_state.doc_alpha_init_error_msg}"}]

initialize_assistants()

# === 7. Setup Auto-Refresh and Check Reminders (If Alpha is Ready) ===
# Determine if Alpha is ready for reminder checking (instance exists AND memory functions are callable)
is_alpha_ready_for_reminders = (
    alpha_available
    and st.session_state.alpha_instance is not None
    and callable(_load_memory_data)
    and callable(_save_memory_data)
)

refresh_interval_seconds = 10 # Check every 20 seconds
if is_alpha_ready_for_reminders:
    # Only setup autorefresh if needed
    st_autorefresh(interval=refresh_interval_seconds * 1000, key="reminder_refresher")
    # print("DEBUG: Autorefresh active for reminders.") # Can be noisy
    check_and_display_reminders() # Check reminders on *every* script run (including refreshes and interactions)
# else:
#     print("DEBUG: Autorefresh inactive (Alpha not ready or memory functions missing).")


# === 8. Sidebar Rendering Logic ===

# --- Memory Overview and Calendar (for Alpha mode) ---
def display_sidebar_memory_overview():
    st.subheader("🧠 Alpha Memory")
    if not alpha_available or not st.session_state.alpha_instance or not callable(_load_memory_data):
        st.warning("Memory features unavailable.")
        if st.session_state.alpha_init_error_msg:
             st.error(f"Init Error: {st.session_state.alpha_init_error_msg}")
        elif not callable(_load_memory_data):
             st.error("Memory loading function (`_load_memory_data`) not found or callable in `alpha1.py`.")
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
            pending_reminders = [r for r in reminders_list if isinstance(r, dict) and r.get("status") == "pending"]
            if pending_reminders:
                try:
                    # Sort by reminder time, putting invalid times last
                    pending_reminders.sort(key=lambda r: datetime.strptime(r.get("reminder_time", "9999-12-31 23:59:59"), '%Y-%m-%d %H:%M:%S'))
                except ValueError:
                    st.warning("Some reminder times have invalid format, sorting might be affected.")
                # Display top N pending reminders
                max_reminders_to_show = 5
                for reminder in pending_reminders[:max_reminders_to_show]:
                    msg = reminder.get('message', 'No Message')[:40] # Truncate long messages for display
                    time_str = reminder.get('reminder_time', 'Invalid Time')
                    st.markdown(f"- `{time_str}`: {msg}{'...' if len(reminder.get('message', '')) > 40 else ''}")
                if len(pending_reminders) > max_reminders_to_show:
                    st.caption(f"... and {len(pending_reminders) - max_reminders_to_show} more.")
            else:
                st.caption("No pending reminders.")
        else:
            st.warning("Reminders data in memory is invalid (not a list).")
        st.divider()

        # --- Calendar for Daily Tasks/Memories ---
        st.markdown("**📅 Memory Calendar**")
        schedule_data = memory_data.get("Schedule", {})
        calendar_events = []
        if isinstance(schedule_data, dict):
            for date_str, tasks in schedule_data.items():
                # Check if tasks exist for the date and it's a valid date format
                if isinstance(tasks, dict) and tasks:
                    try:
                        datetime.strptime(date_str, '%Y-%m-%d') # Validate date format strictly
                        calendar_events.append({
                            "title": f"🗒️ {len(tasks)}", # Show number of items as event title
                            "start": date_str,       # Start of the day
                            "end": date_str,         # End of the day (needed for allDay)
                            "color": "#3498db",      # Blue marker for days with entries
                            "allDay": True,          # Makes it a background event for the day
                            # Optional: Add custom property to store date if needed later
                            # "extendedProps": {"date_str": date_str}
                        })
                    except ValueError:
                        print(f"Warning: Invalid date format '{date_str}' found in schedule data, skipping for calendar.")
        else:
            st.warning("Schedule data in memory is not a dictionary.")

        # Calendar Configuration
        calendar_options = {
            "headerToolbar": {
                "left": "prev,next today",
                "center": "title",
                "right": "dayGridMonth", # Keep it simple, only month view needed here
            },
            "initialView": "dayGridMonth",
            "selectable": True,         # Allow clicking on dates
            "eventColor": '#808080',    # Default color (unused if event has own color)
            "height": "350px",          # Limit calendar height (sync with CSS if needed)
            "contentHeight": "auto",    # Adjust content height automatically
            "editable": False,          # Don't allow dragging events
        }

        # Render the Calendar
        calendar_result = calendar(
            events=calendar_events,
            options=calendar_options,
            callbacks=["dateClick"],    # Trigger callback ONLY when a date background is clicked
            key="memory_calendar"       # IMPORTANT: Key prevents state issues on reruns
        )

        # --- Handle Calendar Click --- # <<< MODIFIED SECTION START >>>
        clicked_date_str = None
        date_extracted = False

        # Added more robust debugging
        if calendar_result and isinstance(calendar_result, dict):
            # Log the full result for easier debugging if needed
            # print(f"\nDEBUG (Calendar Click): Received calendar_result: {json.dumps(calendar_result, indent=2)}")

            # Check if it's the 'dateClick' callback we requested
            if calendar_result.get('callback') == 'dateClick':
                clicked_data = calendar_result.get('dateClick')
                if clicked_data and isinstance(clicked_data, dict):
                    # --- PRIORITIZE 'dateStr' --- It usually contains just 'YYYY-MM-DD'
                    if 'dateStr' in clicked_data and isinstance(clicked_data['dateStr'], str):
                        raw_date_str = clicked_data['dateStr']
                        # Validate format just in case (optional but safer)
                        try:
                            datetime.strptime(raw_date_str[:10], '%Y-%m-%d') # Check if first 10 chars are valid date
                            clicked_date_str = raw_date_str[:10]
                            date_extracted = True
                            print(f"DEBUG: Extracted date via ['dateClick']['dateStr']: {clicked_date_str}")
                        except ValueError:
                            print(f"DEBUG: Value in ['dateClick']['dateStr'] ('{raw_date_str}') not in YYYY-MM-DD format.")

                    # Fallback to 'date' if 'dateStr' isn't present or invalid format
                    elif not date_extracted and 'date' in clicked_data and isinstance(clicked_data['date'], str):
                        raw_date = clicked_data['date']
                        # Slice the potentially longer ISO string (e.g., '2024-04-24T00:00:00.000Z')
                        if len(raw_date) >= 10:
                            potential_date = raw_date[:10]
                            # Add validation here too
                            try:
                                datetime.strptime(potential_date, '%Y-%m-%d')
                                clicked_date_str = potential_date
                                clicked_date_str = clicked_date_str[:-2] + str(int(clicked_date_str.split('-')[2])+1)
                                date_extracted = True
                                print(f"DEBUG: Extracted date via ['dateClick']['date'] (fallback): {clicked_date_str}")
                            except ValueError:
                                print(f"DEBUG: Sliced date from ['dateClick']['date'] ('{potential_date}') is invalid format.")
                        else:
                            print(f"DEBUG: Value in ['dateClick']['date'] ('{raw_date}') is too short.")

                    # Log if callback happened but no suitable date field was found
                    if not date_extracted:
                        print(f"DEBUG: ['dateClick'] structure present, but valid 'dateStr' or 'date' not found: {clicked_data}")

                else: print(f"DEBUG: Expected 'dateClick' data missing or invalid: {clicked_data}")
            else:
                # Log if the callback name doesn't match (shouldn't happen with current config)
                print(f"DEBUG: Received callback '{calendar_result.get('callback')}', expected 'dateClick'. Full result: {calendar_result}")

        # --- State Update Logic ---
        if date_extracted and clicked_date_str:
            current_selected = st.session_state.get('selected_calendar_date')
            print(f"DEBUG: Date extracted: {clicked_date_str}. Current selection in state: {current_selected}")
            # Only update state and rerun if the selected date has actually changed
            if current_selected != clicked_date_str:
                print(f"DEBUG: Updating selected_calendar_date from '{current_selected}' to '{clicked_date_str}' and rerunning...")
                st.session_state.selected_calendar_date = clicked_date_str
                st.rerun() # Rerun necessary to reflect selection and update displayed activities
            else:
                # Avoid unnecessary reruns if the same date is clicked again
                print(f"DEBUG: Clicked date {clicked_date_str} is already selected. No state change or rerun triggered.")
        elif calendar_result and calendar_result.get('callback') == 'dateClick':
            # If it was a dateClick event but we failed to extract a date, log it
            print("DEBUG: A dateClick event occurred, but failed to extract a valid YYYY-MM-DD date.")
        # <<< MODIFIED SECTION END >>>

        # --- Display Activities for the selected date ---
        st.divider() # Separator before showing activities
        display_date = st.session_state.get('selected_calendar_date')

        if display_date:
            print("display_date",display_date)
            st.markdown(f"**Activities for {display_date}:**")
            # Retrieve activities safely, defaulting to an empty dict
            activities = schedule_data.get(display_date, {})

            if isinstance(activities, dict) and activities:
                # Use a container with fixed height for scrollable list if needed
                with st.container(height=150):
                    # Sort items alphabetically by label for consistent display
                    for label, value in sorted(activities.items()):
                        # Basic display, improve formatting as needed (e.g., handle long values)
                        display_value = str(value)[:100] + ('...' if len(str(value)) > 100 else '') # Truncate long values
                        st.markdown(f"- **{label.replace('_', ' ').title()}**")
            elif isinstance(activities, dict):
                st.caption(f"No activities recorded for {display_date}.")
            else:
                st.warning(f"Activity data for {display_date} is invalid (not a dictionary).")
        else:
             st.caption("Click a date on the calendar to view recorded activities.")

    except FileNotFoundError:
        st.error(f"Memory file '{MEMORY_FILE}' not found. Cannot load schedule/reminders.")
    except json.JSONDecodeError:
         st.error(f"Error decoding memory file '{MEMORY_FILE}'. It might be corrupted. Check the file content.")
         if MEMORY_FILE: print(f"ERROR: JSONDecodeError for {MEMORY_FILE}", file=sys.stderr)
    except Exception as e:
        st.error(f"An unexpected error occurred rendering the memory overview/calendar:")
        st.exception(e) # Show full traceback in the UI for easier debugging
        print("ERROR rendering memory overview/calendar:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


# --- Document Upload Sidebar Logic (for Document Analysis mode) ---
def display_sidebar_document_upload():
    st.subheader("📄 Document Upload & Context")
    if not doc_alpha_available:
        st.error(f"Doc Analysis Unavailable: {st.session_state.doc_alpha_init_error_msg or 'Import failed.'}")
        return
    if st.session_state.doc_alpha_init_error_msg and not st.session_state.doc_alpha_instance:
         # Show error if initialization failed specifically
         st.error(f"DocAlpha Init Error: {st.session_state.doc_alpha_init_error_msg}")
         return
    if not st.session_state.doc_alpha_instance:
         # Generic message if instance just isn't ready yet
         st.info("Document uploader unavailable (DocAlpha not initialized).")
         return

    # --- File Uploader Widget ---
    uploaded_file = st.file_uploader(
        "Upload a document for analysis",
        type=None, # Allow any type for now, add specific types if needed e.g., ['pdf', 'txt', 'md']
        key="file_uploader",
        accept_multiple_files=False, # Stick to single file for simplicity
        help="Upload a file (PDF, TXT, etc.) to analyze its content."
    )

    # --- File Processing Logic ---
    if uploaded_file is not None:
        # Check if it's a new file compared to the one currently loaded
        is_new_file = (st.session_state.current_file_display_name != uploaded_file.name)

        # Process only if it's genuinely a new file upload (prevents reprocessing on unrelated reruns)
        # Use a temporary state variable as a lock to avoid race conditions during processing
        if is_new_file and "file_being_processed" not in st.session_state:
            st.session_state["file_being_processed"] = uploaded_file.name # Lock with filename
            st.session_state.current_file_context = None # Clear old context immediately
            st.session_state.current_file_display_name = None

            with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
                temp_file_path = None
                try:
                    # 1. Validate TEMP_UPLOAD_DIR configuration
                    if not TEMP_UPLOAD_DIR or not isinstance(TEMP_UPLOAD_DIR, str):
                        st.error("Configuration Error: TEMP_UPLOAD_DIR is not set correctly in DocumentAlpha.py.")
                        raise ValueError("TEMP_UPLOAD_DIR invalid")
                    if not os.path.exists(TEMP_UPLOAD_DIR):
                         os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
                         print(f"DEBUG: Created temporary upload directory: {TEMP_UPLOAD_DIR}")

                    # 2. Save uploaded file to a temporary location
                    # Use a safe filename approach if needed (e.g., unique IDs)
                    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"upload_{int(time.time())}_{uploaded_file.name}")
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    print(f"DEBUG: Temp file saved at '{temp_file_path}'")

                    # 3. Upload to Google AI using genai.upload_file
                    print(f"DEBUG: Uploading '{temp_file_path}' to Google AI...")
                    # Ensure genai is configured (API key should be set via genai.configure or env vars)
                    # Basic check - relies on prior configuration or env var
                    if not getattr(genai, 'api_key', None) and not os.getenv('GOOGLE_API_KEY'):
                          print("Warning: Cannot definitively confirm genai configuration. Attempting upload...")
                          # Consider adding a more robust check if the API allows without making a full call

                    # Check if the File type was imported successfully
                    if File is None:
                        st.error("Google AI File type not available (import failed). Cannot process document.")
                        raise ImportError("Required type `google.generativeai.types.File` not found.")

                    # Perform the upload via SDK
                    gemini_file_object = genai.upload_file(
                        path=temp_file_path,
                        display_name=uploaded_file.name # Use original filename for display in Gemini UI
                        # mime_type can often be inferred, or set explicitly: mime_type=uploaded_file.type
                    )
                    print(f"DEBUG: Google AI File uploaded successfully. ID: {gemini_file_object.name}, Display Name: {gemini_file_object.display_name}")

                    # --- IMPORTANT: Check File State ---
                    # Wait for the file to be processed by Google AI
                    file_state_timeout = 120 # seconds to wait
                    start_time = time.time()
                    while gemini_file_object.state.name == "PROCESSING":
                         print("DEBUG: Google AI File state is PROCESSING, waiting...")
                         time.sleep(5) # Wait 5 seconds before checking again
                         gemini_file_object = genai.get_file(gemini_file_object.name) # Re-fetch file state
                         if time.time() - start_time > file_state_timeout:
                              raise TimeoutError(f"Google AI File processing timed out after {file_state_timeout}s.")

                    if gemini_file_object.state.name != "ACTIVE":
                         st.error(f"❌ Google AI File is not active. State: {gemini_file_object.state.name}. Cannot use for analysis.")
                         raise RuntimeError(f"Google AI File failed processing. Final state: {gemini_file_object.state.name}")

                    print(f"DEBUG: Google AI File is now ACTIVE (ID: {gemini_file_object.name}).")

                    # 4. Update session state with the processed file object
                    st.session_state.current_file_context = [gemini_file_object] # Store as a list (future-proof for multi-file)
                    st.session_state.current_file_display_name = uploaded_file.name
                    st.success(f"✅ Ready: '{uploaded_file.name}'")

                    # Reset chat history for the new document to avoid confusion
                    st.session_state.document_messages = [{"role": "assistant", "content": f"File '{uploaded_file.name}' is loaded and ready. Ask me questions about its content."}]

                except (TimeoutError, RuntimeError) as file_state_err:
                     st.error(f"❌ Error during file preparation: {file_state_err}")
                     print(f"ERROR during file preparation: {file_state_err}", file=sys.stderr)
                     st.session_state.current_file_context = None
                     st.session_state.current_file_display_name = None
                     # Add error to chat
                     st.session_state.document_messages.append({"role": "assistant", "content": f"Error processing {uploaded_file.name}: {file_state_err}. Please try again or upload a different file."})

                except Exception as e:
                    st.error(f"❌ Error processing file '{uploaded_file.name}': {e}")
                    print(f"ERROR processing file '{uploaded_file.name}':", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    st.session_state.current_file_context = None
                    st.session_state.current_file_display_name = None
                    # Add generic error to chat
                    st.session_state.document_messages.append({"role": "assistant", "content": f"An unexpected error occurred while processing {uploaded_file.name}. Please check logs or try again."})

                finally:
                    # 5. Clean up temporary file regardless of success/failure
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            print(f"DEBUG: Removed temp file: {temp_file_path}")
                        except Exception as e_rem:
                            print(f"Warning: Could not remove temp file {temp_file_path}: {e_rem}")

                    # Release the lock and rerun to update UI immediately
                    if "file_being_processed" in st.session_state:
                        del st.session_state["file_being_processed"]
                        print("DEBUG: Released file processing lock.")
                    st.rerun()

    # --- Display Current Context or Clear Button ---
    # This part runs if no *new* file was uploaded in this script run
    elif st.session_state.current_file_display_name:
         st.markdown(f"**Current File Context:**")
         # Use success box for better visibility of the active file
         st.success(f"`{st.session_state.current_file_display_name}` is loaded.")

         # --- Button to Clear Context ---
         if st.button("Clear File Context & Chat", key="clear_context", help="Removes the current file context and clears the document chat history."):
             file_to_delete_id = None
             file_display_name_cleared = st.session_state.current_file_display_name

             # --- Optional: Delete from Google AI Storage ---
             # Be cautious with this - ensure users want permanent deletion.
             # Add a confirmation step if implementing deletion.
             if st.session_state.current_file_context and isinstance(st.session_state.current_file_context, list):
                 # Assuming single file context for now
                 if len(st.session_state.current_file_context) > 0 and File and isinstance(st.session_state.current_file_context[0], File):
                    file_obj = st.session_state.current_file_context[0]
                    file_to_delete_id = file_obj.name # The ID is stored in the 'name' attribute
                    try:
                        print(f"DEBUG: Attempting to delete Google AI file: ID {file_to_delete_id}")
                        # --- UNCOMMENT TO ENABLE DELETION ---
                        # genai.delete_file(file_to_delete_id)
                        # print(f"DEBUG: Successfully deleted Google AI file: {file_to_delete_id}")
                        # st.toast(f"Google AI file '{file_display_name_cleared}' deleted.", icon="🗑️")
                        # --- KEEP COMMENTED IF DELETION IS NOT DESIRED ---
                        print(f"INFO: Deletion of Google AI file '{file_to_delete_id}' is currently disabled in the code.")
                        st.info("Note: File context cleared from app, but file deletion on Google AI is disabled.", icon="ℹ️")

                    except Exception as del_err:
                        print(f"Warning: Failed to delete Google AI file {file_to_delete_id}: {del_err}")
                        st.warning(f"Could not delete file '{file_display_name_cleared}' from Google AI storage: {del_err}", icon="⚠️")

             # Clear session state regardless of deletion success/failure
             st.session_state.current_file_context = None
             st.session_state.current_file_display_name = None
             # Clear document chat history as well
             st.session_state.document_messages = [{"role": "assistant", "content": "File context cleared. Upload a new document to begin analysis."}]
             print(f"DEBUG: File context for '{file_display_name_cleared}' cleared by user.")
             st.toast("File context and chat cleared!", icon="🧹")
             st.rerun() # Rerun to reflect the cleared state
    else:
         # If no file has been uploaded yet or context was cleared
         st.info("Upload a file using the button above to start analysis.")


# === 9. Sidebar Definition ===
with st.sidebar:
    st.image("logo.png", width=100)  # Optional: Add a logo if you have one
    st.header("Unified Assistant")

    # --- Mode Selection ---
    # Use index based on current session state for consistent selection after reruns
    current_mode_index = 0 if st.session_state.app_mode == '🤖 General Assistant' else 1
    app_mode = st.radio(
        "Select Mode:",
        ('🤖 General Assistant', '📄 Document Analysis'),
        key='app_mode_radio',       # Unique key for the radio widget
        index=current_mode_index,
        horizontal=True,           # Display options side-by-side
        help="Switch between the general chatbot and the document analysis tool."
    )

    # Update app_mode in session state *only if* the radio button value has changed
    if st.session_state.app_mode != app_mode:
        st.session_state.app_mode = app_mode
        print(f"DEBUG: Mode changed to '{app_mode}' via radio button. Rerunning.")
        # Clear specific states when switching modes? (Optional)
        # e.g., clear calendar selection when switching away from General
        # if app_mode == '📄 Document Analysis':
        #    st.session_state.selected_calendar_date = None
        st.rerun() # Rerun immediately to load the correct UI and sidebar sections

    st.divider()

    # --- Conditional Sidebar Content ---
    if st.session_state.app_mode == '🤖 General Assistant':
        # Display memory overview (reminders, calendar)
        display_sidebar_memory_overview()
        st.divider()
        st.subheader("Chat Controls")
        if st.button("Clear General Chat", key="clear_general", help="Clears the chat history for the General Assistant."):
            st.session_state.general_messages = [{"role": "assistant", "content": "Chat history cleared."}]
            # Also clear the calendar selection when clearing chat
            st.session_state.selected_calendar_date = None
            st.toast("General chat history cleared!", icon="🧹")
            print("DEBUG: General chat cleared.")
            st.rerun()

    elif st.session_state.app_mode == '📄 Document Analysis':
        # Display document uploader and context manager
        display_sidebar_document_upload()
        st.divider()
        st.subheader("Chat Controls")
        # Separate clear button for document chat - note: context clear is above
        if st.button("Clear Document Chat Only", key="clear_document_chat", help="Clears only the chat history for Document Analysis, keeping the loaded file context."):
             # Keep the introductory message based on whether a file is loaded
             if st.session_state.current_file_display_name:
                  st.session_state.document_messages = [{"role": "assistant", "content": f"Document chat history cleared. File '{st.session_state.current_file_display_name}' is still loaded. Ask questions or clear the file context."}]
             else:
                  st.session_state.document_messages = [{"role": "assistant", "content": "Document chat history cleared. Upload a document to begin."}]
             st.toast("Document chat history cleared!", icon="🧹")
             print("DEBUG: Document chat cleared (context kept).")
             st.rerun()


# === 10. Main Area Rendering Logic ===

# --- General Assistant UI ---
def render_general_assistant_ui():
    st.title("🤖 General Assistant (Alpha)")

    # --- Status Bar ---
    is_alpha_ready = alpha_available and st.session_state.alpha_instance is not None
    status_color = "green" if is_alpha_ready else ("orange" if alpha_init_error and not is_alpha_ready else "red")
    status_text = "Ready" if is_alpha_ready else ("Import Error" if alpha_init_error and not st.session_state.alpha_instance else "Initialization Failed")
    model_name = st.session_state.get('alpha_model_name', 'N/A')
    # Reminder status depends on Alpha *and* memory functions
    reminder_status = 'Active' if is_alpha_ready_for_reminders else ('Inactive (Alpha Not Ready)' if not is_alpha_ready else 'Inactive (Memory Error)')
    st.caption(f"Status: :{status_color}[{status_text}] | Model: `{model_name}` | Reminders: `{reminder_status}`")

    # --- Display Initialization Errors prominently if needed ---
    if not is_alpha_ready:
        init_err = st.session_state.alpha_init_error_msg or "Import failed or Alpha object could not be created."
        st.error(f"Alpha Assistant is currently unavailable. Reason: {init_err}")
        # Provide more specific guidance based on common errors
        if "FileNotFoundError" in init_err:
             st.warning("Check that `alpha1.py` exists in the same directory as `app.py`.")
        elif "ImportError" in init_err or "ModuleNotFoundError" in init_err:
             st.warning("Check that `alpha1.py` has the necessary `Alpha` class, helper functions (`_load_memory_data`, `_save_memory_data`), `MEMORY_FILE` constant, and that all required Python packages (like `google-generativeai`, potentially `schedule`) are installed.")
        elif "Runtime Error" in init_err:
             st.warning("An error occurred during Alpha's initialization. Check the console logs (where you ran `streamlit run app.py`) for detailed tracebacks.")
        return # Stop rendering the chat UI if Alpha isn't ready

    st.divider() # Separator before chat messages

    # --- Chat Message Display ---
    # Use a container to potentially limit height and make it scrollable if it gets very long
    # chat_container = st.container(height=500) # Example: Limit height
    # with chat_container:
    for message in st.session_state.general_messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"]) # Render markdown for potential formatting

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
        # with st.chat_message("user", avatar="👤"): st.markdown(prompt_alpha) # No need, rerun handles display

        # --- Process Input ---
        # Simple exit condition check
        if prompt_alpha.lower().strip() in ["exit", "quit", "bye", "stop", "goodbye", "0"]:
            response = "Okay, goodbye! Let me know if you need anything else."
            st.session_state.general_messages.append({"role": "assistant", "content": response})
            # No rerun needed here, just display the goodbye message on the next natural run.
            # Or force rerun if immediate display is desired:
            # st.rerun()
        else:
            # Process the input with the Alpha instance
            try:
                ai = st.session_state.alpha_instance
                with st.spinner("Alpha is thinking... 🤔"):
                    # Ensure prompt is a non-empty string and stripped of whitespace
                    clean_prompt = str(prompt_alpha).strip()
                    if not clean_prompt:
                         ai_response = "Please provide some input." # Handle empty input
                    else:
                         # Call the backend chat function
                         ai_response = ai.chat_with_gemini(clean_prompt)

                # Handle potential None or unexpected response types from the backend
                ai_response_str = str(ai_response) if ai_response is not None else "[Alpha did not provide a text response]"
                if ai_response is None:
                    st.warning("Alpha returned an empty response. The request might have been processed (e.g., reminder set), but no confirmation message was generated.")
                    # Still append the placeholder to history
                    st.session_state.general_messages.append({"role": "assistant", "content": ai_response_str})
                else:
                    st.session_state.general_messages.append({"role": "assistant", "content": ai_response_str})

            except Exception as chat_error:
                print(f"ERROR during Alpha chat interaction: {chat_error}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                error_message = f"Sorry, an error occurred while processing your request: ({type(chat_error).__name__}). Please check the console logs for details or try again."
                st.session_state.general_messages.append({"role": "assistant", "content": error_message})
                # Display error immediately in the chat area for user feedback (optional, as rerun will show it too)
                # st.error(f"An error occurred interacting with Alpha: {chat_error}")
            finally:
                 # Rerun to display the new user message and the assistant's response/error
                 st.rerun()


# --- Document Analysis UI ---
def render_document_analysis_ui():
    st.title("📄 Document Analysis (DocumentAlpha)")

    # --- Status Bar ---
    is_doc_alpha_ready = doc_alpha_available and st.session_state.doc_alpha_instance is not None
    status_color = "green" if is_doc_alpha_ready else ("orange" if doc_alpha_init_error and not is_doc_alpha_ready else "red")
    status_text = "Ready" if is_doc_alpha_ready else ("Import Error" if doc_alpha_init_error and not st.session_state.doc_alpha_instance else "Initialization Failed")
    # File status depends on whether a file has been successfully processed and is active
    file_context_obj = st.session_state.get('current_file_context')
    file_display_name = st.session_state.get('current_file_display_name')
    file_status = "File Context: None Loaded"
    if file_display_name and file_context_obj:
        # You could potentially check the File object state here again if needed
        # file_state = getattr(file_context_obj[0], 'state', None)
        # file_state_name = getattr(file_state, 'name', 'UNKNOWN')
        # file_status = f"File Context: `{file_display_name}` (State: {file_state_name})"
        file_status = f"File Context: `{file_display_name}`" # Simplified status
    elif file_display_name and not file_context_obj:
         file_status = f"File Context: Error with '{file_display_name}'"

    st.caption(f"Status: :{status_color}[{status_text}] | {file_status}")

    # --- Display Initialization Errors ---
    if not is_doc_alpha_ready:
        init_err = st.session_state.doc_alpha_init_error_msg or "Import failed or DocumentAlpha object could not be created."
        st.error(f"Document Analysis Assistant is unavailable. Reason: {init_err}")
        # Provide more specific guidance
        if "FileNotFoundError" in init_err:
             st.warning("Check that `DocumentAlpha.py` exists in the same directory as `app.py`.")
        elif "ImportError" in init_err or "ModuleNotFoundError" in init_err:
             st.warning("Check that `DocumentAlpha.py` has the `DocumentAlpha` class, `TEMP_UPLOAD_DIR`, `TimeoutException`, and required dependencies (like `google-generativeai`) are installed. Ensure Google AI API keys are configured (e.g., via environment variables or `genai.configure`).")
        elif "Runtime Error" in init_err:
             st.warning("An error occurred during DocumentAlpha's initialization. Check console logs for details.")
        return # Stop rendering chat UI

    st.divider()

    # --- Chat Message Display ---
    # chat_container_doc = st.container(height=500) # Optional height limit
    # with chat_container_doc:
    for message in st.session_state.document_messages:
        avatar = "👤" if message["role"] == "user" else "📄" # Document icon for assistant
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # --- Chat Input ---
    # Input should be disabled if DocAlpha isn't ready OR if no valid file context is loaded
    is_input_disabled = not is_doc_alpha_ready or not st.session_state.current_file_context
    input_placeholder = "Ask about the document..." if st.session_state.current_file_context else "Upload a document via the sidebar to enable chat"
    prompt_doc = st.chat_input(
        input_placeholder,
        key="document_chat_input",
        disabled=is_input_disabled
    )

    if prompt_doc:
        # This block should only be reached if input wasn't disabled, implying context exists
        if not st.session_state.current_file_context:
             # This is a safeguard, should ideally not happen if disabled logic is correct
             st.warning("Cannot process request: No document context is currently loaded.", icon="⚠️")
        else:
            # Add user message to state
            st.session_state.document_messages.append({"role": "user", "content": prompt_doc})
            # Display user message immediately (will be shown on rerun anyway)
            # with st.chat_message("user", avatar="👤"): st.markdown(prompt_doc) # Let rerun handle display

            # --- Get Assistant Response ---
            # Use a placeholder for the assistant's response while processing
            # with st.chat_message("assistant", avatar="📄"):
            #     message_placeholder = st.empty() # For streaming-like effect or final message
            with st.spinner("DocumentAlpha is analyzing... 🧐"):
                try:
                    # 1. Get the current file context from session state
                    file_context = st.session_state.get('current_file_context')

                    # 2. Validate the context format (should be a list of File objects)
                    valid_context = None
                    if isinstance(file_context, list) and len(file_context) > 0:
                        # Check if the File type was successfully imported earlier
                        if File:
                            # Ensure all items in the list are instances of the File type
                            if all(isinstance(f, File) for f in file_context):
                                valid_context = file_context
                                # Optionally, re-check if the file is ACTIVE (might be redundant if checked at upload)
                                # first_file = valid_context[0]
                                # if genai.get_file(first_file.name).state.name != "ACTIVE":
                                #     raise RuntimeError(f"File '{first_file.display_name}' is no longer active.")
                            else:
                                print("Warning: file_context list contains items not of type google.generativeai.types.File.")
                                # Decide how to handle - raise error or attempt to use as is?
                                # For now, let's raise an error for clarity.
                                raise TypeError("Invalid item type found in file context list.")
                        else:
                            # If File type couldn't be imported, maybe proceed with caution?
                            # This path is less likely if upload logic requires File type.
                            print("Warning: google.generativeai.types.File not imported, context validation skipped. Passing raw list.")
                            valid_context = file_context # Attempt to use as is, might fail later
                    elif file_context:
                        print(f"Warning: file_context is not a non-empty list: {type(file_context)}")
                        # Handle error? Attempt conversion? Set to None for now.
                        valid_context = None

                    if not valid_context:
                         # This error should ideally be caught by the disabled input, but good safeguard.
                         raise ValueError("Document context is missing, invalid, or empty.")

                    # 3. Get the DocumentAlpha instance
                    doc_ai = st.session_state.doc_alpha_instance
                    if not doc_ai:
                        raise RuntimeError("DocumentAlpha instance not found in session state.")

                    # 4. Call the chat method with the prompt and validated context
                    clean_prompt = str(prompt_doc).strip()
                    if not clean_prompt:
                         response = "Please ask a specific question about the document."
                    else:
                         response = doc_ai.chat_with_gemini(clean_prompt, file_context=valid_context)

                    # 5. Process response
                    response_text = str(response) if response is not None else "[DocumentAlpha provided no text response]"
                    if response is None:
                        st.warning("DocumentAlpha returned an empty response.")

                    # Append response to message history
                    st.session_state.document_messages.append({"role": "assistant", "content": response_text})
                    # Update placeholder (or just let rerun display it)
                    # message_placeholder.markdown(response_text)

                except TimeoutException as e:
                     error_msg = f"⚠️ Analysis timed out. The document might be too complex or the request took too long. Try asking something simpler or try again later. ({e})"
                     st.session_state.document_messages.append({"role": "assistant", "content": error_msg})
                     # message_placeholder.error(error_msg) # Let rerun display
                except (TypeError, ValueError, RuntimeError) as context_err: # Catch context validation errors
                     error_msg = f"An internal error occurred with the document context: {context_err}. Please try clearing context and re-uploading the file."
                     print(f"ERROR during DocumentAlpha chat (Context): {context_err}", file=sys.stderr)
                     st.session_state.document_messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                     # message_placeholder.error(error_msg) # Let rerun display
                except Exception as e:
                    print(f"ERROR during DocumentAlpha chat interaction: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    error_msg = f"An unexpected error occurred during analysis: {e}. Please check the console logs."
                    st.session_state.document_messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                    # message_placeholder.error(error_msg) # Let rerun display
                finally:
                    # Rerun to display user message and assistant response/error
                    st.rerun()


# === 11. Main App Logic ===
# Select which UI function to call based on the current mode stored in session state
if st.session_state.app_mode == '🤖 General Assistant':
    render_general_assistant_ui()
elif st.session_state.app_mode == '📄 Document Analysis':
    render_document_analysis_ui()
else:
    # Fallback in case app_mode state gets corrupted somehow
    st.error(f"Error: Invalid application mode '{st.session_state.app_mode}' found in session state. Resetting to default.")
    print(f"ERROR: Invalid app_mode '{st.session_state.app_mode}', resetting.")
    st.session_state.app_mode = '🤖 General Assistant'
    time.sleep(1) # Brief pause before rerunning
    st.rerun()