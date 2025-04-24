# app.py (with streamlit-autorefresh for simulated proactive reminders)
import streamlit as st
import os
import sys
import traceback # For detailed error logging if needed
from datetime import datetime # Needed for reminder check
import time # For potential brief sleep

# --- Add import for streamlit-autorefresh ---
from streamlit_autorefresh import st_autorefresh

# === 1. Streamlit Page Configuration ===
st.set_page_config(
    page_title="Alpha Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto" # Or "collapsed"
)

# === 2. Attempt to Import Alpha Class & Helpers ===
current_dir = os.path.dirname(os.path.abspath(__file__))
alpha_file_path = os.path.join(current_dir, 'alpha.py')

try:
    if not os.path.isfile(alpha_file_path):
        raise FileNotFoundError(f"'alpha.py' not found in the directory: {current_dir}")
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Import the main class AND the necessary helper functions from alpha.py
    from alpha import Alpha, _load_memory_data, _save_memory_data, MEMORY_FILE

    # Check for schedule dependency (still good practice as alpha.py imports it)
    try: import schedule
    except ImportError: st.warning("Optional Warning: 'schedule' library (imported by alpha.py) not installed. This is OK for the Streamlit app.")

except (ImportError, FileNotFoundError, ModuleNotFoundError) as e:
    st.error(f"**Fatal Error:** Failed to import `Alpha` or helpers from `alpha.py`.")
    st.error(f"**Reason:** `{e}`")
    st.warning(
        "Ensure `alpha.py` is in the same directory, has no errors, and dependencies "
        "(`google-generativeai`, `python-dotenv`, `python-dateutil`, `schedule`) are installed."
        # Note: schedule is technically only needed if running alpha.py directly now
    )
    with st.expander("Debugging Information", expanded=True):
        st.write(f"*   App Directory: `{current_dir}`")
        st.write(f"*   Expected Alpha Path: `{alpha_file_path}`")
        st.write(f"*   Does 'alpha.py' exist? `{os.path.isfile(alpha_file_path)}`")
        st.write("*   Python's Search Path (`sys.path`):")
        st.json(sys.path)
    st.stop()

except Exception as e:
    st.error(f"An unexpected error occurred during the import process:")
    st.exception(e)
    st.stop()


# === 3. Reminder Checking Function (Uses st.toast) ===

def check_and_display_reminders():
    """
    Loads reminders, displays due ones via st.toast, updates status in memory file.
    Called on every Streamlit rerun (including auto-refreshes).
    """
    # Add a lock or timestamp check to prevent rapid-fire checks if saves are slow?
    # For simplicity, we'll proceed, but be aware of potential race conditions on slow filesystems.
    # print(f"DEBUG: Checking reminders at {datetime.now()}") # For debug if needed
    try:
        now = datetime.now()
        # Use the imported helper function to load data
        data = _load_memory_data() # Reads the latest state from the file
        reminders = data.get("Reminders", [])
        if not isinstance(reminders, list):
             print("ERROR: Reminder data format error (not a list). Cannot check reminders.")
             # Avoid showing error repeatedly via toast if data is corrupt
             # st.error("Reminder data format error.", icon="🚨")
             return # Stop checking if data is corrupt

        reminders_updated = False
        # Keep track of IDs processed in this run to avoid duplicate toasts/updates
        processed_ids_this_run = set()

        for index, reminder in enumerate(reminders):
             try:
                reminder_id = reminder.get('id', f"index_{index}") # Get ID or use index

                # Skip if already processed in this specific run
                if reminder_id in processed_ids_this_run:
                     continue

                # Check structure and status
                if isinstance(reminder, dict) and reminder.get("status") == "pending":
                    reminder_time_str = reminder.get("reminder_time")

                    if not reminder_time_str:
                        print(f"Warning: Reminder {reminder_id} has no time.")
                        if reminder.get('status') != 'error_missing_time':
                           reminder['status'] = 'error_missing_time'; reminders_updated = True
                        processed_ids_this_run.add(reminder_id) # Mark as handled
                        continue

                    reminder_dt = datetime.strptime(reminder_time_str, '%Y-%m-%d %H:%M:%S')

                    # Check if due
                    if reminder_dt <= now:
                        message = reminder.get('message', 'Reminder!')
                        # Display using st.toast
                        st.toast(f"🔔 Reminder: {message}", icon="⏰")
                        print(f"INFO: Displayed UI reminder: {reminder_id}")

                        # Mark for status update
                        if reminder.get('status') != 'triggered':
                             reminder['status'] = 'triggered'; reminders_updated = True
                        processed_ids_this_run.add(reminder_id) # Mark as handled

             except (ValueError, TypeError) as parse_error:
                  print(f"ERROR: Could not parse time for reminder {reminder_id}: {parse_error}")
                  if isinstance(reminder, dict) and reminder.get('status') != 'error_processing':
                      reminder['status'] = 'error_processing'; reminders_updated = True
                  processed_ids_this_run.add(reminder_id) # Mark error state as handled
             except Exception as e:
                  print(f"ERROR: Unexpected error checking reminder {reminder_id}: {e}")
                  traceback.print_exc()
                  if isinstance(reminder, dict) and reminder.get('status') != 'error_unexpected':
                      reminder['status'] = 'error_unexpected'; reminders_updated = True
                  processed_ids_this_run.add(reminder_id) # Mark error state as handled

        # Save the modified data structure back *only if* updates occurred
        if reminders_updated:
            print(f"DEBUG (autorefresh): Found {len(processed_ids_this_run)} reminders triggered/errored this run. Saving updates.")
            if not _save_memory_data(data): # Use imported helper
                 print("ERROR (autorefresh): _save_memory_data failed after updating reminder statuses.")
                 st.error("Failed to save updated reminder statuses.", icon="💾") # Show error in UI
            else:
                 print("DEBUG (autorefresh): Successfully saved updated reminder statuses.")
                 # Optional: Short sleep if rapid file access is an issue
                 # time.sleep(0.1)

    except Exception as e:
         print(f"ERROR (autorefresh): General failure in check_and_display_reminders: {e}")
         traceback.print_exc()
         # Avoid showing error toast constantly, just log it.


# === 4. Streamlit App Title and Styling ===
st.title("🤖 Alpha Assistant")
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 1rem;
        border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# === 5. Initialize Alpha and Chat History in Session State ===
if 'alpha_instance' not in st.session_state:
    st.session_state.alpha_instance = None
    st.session_state.messages = []
    st.session_state.model_name = "Initializing..."
    st.session_state.init_error = None
    try:
        with st.spinner("Waking up Alpha... (Initializing Model & Memory)"):
            alpha_init_instance = Alpha()
        if alpha_init_instance and hasattr(alpha_init_instance, 'model'):
            st.session_state.alpha_instance = alpha_init_instance
            st.toast("Alpha initialized successfully!", icon="✅")
            try:
                model_obj = getattr(st.session_state.alpha_instance, 'model', None)
                full_model_path = getattr(model_obj, '_model_name', "Unknown Model")
                st.session_state.model_name = full_model_path.split('/')[-1]
            except Exception: st.session_state.model_name = "Model name N/A"
            # Add initial greeting only if messages are empty
            if not st.session_state.messages:
                 st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hello! I'm Alpha. How can I assist you today? You can ask me to remember things, retrieve memories, or set reminders."
                 })
        else: raise RuntimeError("Alpha instance created but appears incomplete.")
    except Exception as init_error:
        st.session_state.init_error = init_error
        st.session_state.alpha_instance = None
        st.session_state.model_name = "Initialization Failed"
        if not st.session_state.messages: # Add error message if list is empty
            st.session_state.messages = [{"role": "assistant", "content": "Error: Could not initialize Alpha."}]


# === 6. Display Initialization Error (if any) ===
if st.session_state.init_error:
    st.error(f"**Fatal Error Initializing Alpha:** `{st.session_state.init_error}`")
    st.warning("Initialization failed. Check API key, model name, network, `alpha.py`.")
    with st.expander("Show Full Error Traceback"): st.text(traceback.format_exc())


# === 7. Display Model Name and Status ===
is_ready = st.session_state.alpha_instance is not None and st.session_state.init_error is None
status_color = "green" if is_ready else "red"
status_text = "Ready" if is_ready else "Initialization Failed"
st.caption(f"Model: `{st.session_state.model_name}` | Status: :{status_color}[{status_text}]")
st.divider()


# === 8. Set up Auto-Refresh and Check Reminders ===
refresh_interval_seconds = 20 # Check reminders every 20 seconds (adjust as needed)
if is_ready:
    # Run the auto-refresh component. interval is in milliseconds.
    st_autorefresh(interval=refresh_interval_seconds * 1000, key="reminder_refresher")

    # Call the reminder check function on every run (including auto-refreshes)
    check_and_display_reminders()
# No need for an 'else' here, input is already disabled if not ready


# === 9. Display Chat History ===
# Display messages AFTER potentially checking reminders
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# === 10. Handle User Input ===
chat_input_disabled = not is_ready
prompt = st.chat_input(
    "Ask Alpha anything...",
    disabled=chat_input_disabled,
    key="chat_input" # Key helps maintain input state if needed
)

if prompt:
    if not is_ready:
        st.error("Cannot process input: Alpha is not initialized correctly.")
    else:
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"): st.markdown(prompt)

        # Handle exit command
        if prompt.lower().strip() in ["exit", "quit", "goodbye", "bye", "stop", "0"]:
            response = "Okay, goodbye! Feel free to ask if you need anything else later."
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant", avatar="🤖"): st.markdown(response)
            # Optionally clear input or disable further input? Usually just let it sit.
        else:
            # Process with Alpha
            try:
                ai = st.session_state.alpha_instance
                with st.spinner("Alpha is thinking... 🤔"):
                    ai_response = ai.chat_with_gemini(prompt.strip())

                # Append and display AI response
                if ai_response is not None:
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    with st.chat_message("assistant", avatar="🤖"): st.markdown(ai_response)
                else:
                    # Handle case where LLM explicitly returns no text (e.g., after silent success)
                    st.warning("Alpha returned an empty response this time.")
                    # Optionally add a placeholder or not, depending on preference
                    st.session_state.messages.append({"role": "assistant", "content": "[Alpha provided no text response]"})
                    with st.chat_message("assistant", avatar="🤖"): st.markdown("[Alpha provided no text response]")

            except Exception as chat_error:
                st.error(f"An error occurred while getting Alpha's response:")
                with st.expander("Show Error Details"): st.exception(chat_error)
                error_message = f"Sorry, I encountered an error processing that request. ({type(chat_error).__name__})"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                # Rerun to display the error message added to the chat history
                st.rerun()

# === 11. Sidebar Information ===
with st.sidebar:
    st.header("About Alpha")
    st.info("Alpha manages info, memories, and reminders using Google's Gemini model.")
    st.markdown("---")
    st.header("Reminders")
    st.success(f"Reminders set via chat will appear as toast notifications when due (page auto-refreshes ~every {refresh_interval_seconds}s).")
    st.markdown("---")
    st.header("Controls")
    if st.button("Clear Chat History"):
        # Keep initial greeting if it exists and is the first message
        if st.session_state.messages and st.session_state.messages[0]['role'] == 'assistant':
            st.session_state.messages = [st.session_state.messages[0]]
        else:
             # Otherwise start completely fresh or add a default greeting
             st.session_state.messages = [{
                "role": "assistant",
                "content": "Chat history cleared. How can I help?"
             }]
        st.toast("Chat history cleared!", icon="🧹")
        # Rerun to reflect the cleared history immediately
        st.rerun()