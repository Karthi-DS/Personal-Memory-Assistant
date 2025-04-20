# app.py (Example Streamlit Application - CORRECTED)

import streamlit as st
import os
import traceback
import google.generativeai as genai
from google.generativeai.types import File # Import File type explicitly
from typing import Optional, List # <--- IMPORT Optional and List HERE

# Assuming alpha1.py is in the same directory
# Ensure alpha1.py is fully correct as provided previously
from alpha1 import Alpha, TEMP_UPLOAD_DIR, TimeoutException


# --- Page Configuration ---
st.set_page_config(page_title="Alpha Assistant", layout="wide")
st.title("💬 Alpha - Document Analysis Assistant")

# --- Initialization ---
# Initialize Alpha instance in session state to avoid re-creating it on every interaction
if "alpha_instance" not in st.session_state:
    try:
        # Show spinner during potentially long initialization
        with st.spinner("Initializing Alpha Assistant..."):
            st.session_state.alpha_instance = Alpha()
        print("INFO: Alpha instance created and stored in session state.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the Alpha assistant: {e}")
        traceback.print_exc()
        st.stop() # Stop the app if Alpha can't initialize

# Ensure alpha_instance is available after initialization attempt
if not hasattr(st.session_state, 'alpha_instance') or not st.session_state.alpha_instance:
     st.error("Error: Alpha instance is not available in session state. Please refresh.")
     st.stop()

alpha: Alpha = st.session_state.alpha_instance # Get the instance

# Initialize chat history and file context in session state
if "messages" not in st.session_state:
    st.session_state.messages = [] # Store chat messages: {"role": "user/assistant", "content": "..."}
    print("INFO: Initialized messages in session state.")

if "current_file_context" not in st.session_state:
    # Store the File object(s) from genai.upload_file
    st.session_state.current_file_context = None
    print("INFO: Initialized current_file_context in session state.")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("📄 File Upload")
    st.markdown("Upload a document for Alpha to analyze.")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=None, # Allow common doc types, adjust as needed
        key="file_uploader", # Use key to help manage state
        accept_multiple_files=False # Keep it simple with one file for now
    )

    # Clear context if the uploader is cleared or a new file is being processed
    # This logic ensures old context doesn't linger if user removes file via UI
    if uploaded_file is None and "file_being_processed" not in st.session_state:
         if st.session_state.current_file_context is not None:
              print("DEBUG: Clearing file context as uploader is empty.")
              st.session_state.current_file_context = None

    if uploaded_file is not None:
        # Indicate processing started
        st.session_state["file_being_processed"] = True
        # Display spinner while processing the file
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            # Save temporary file (needed for genai.upload_file path argument)
            temp_file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                print(f"DEBUG: Temporarily saved uploaded file to: {temp_file_path}")

                # Upload the file to Google AI Studio (this makes it accessible to the model)
                print(f"DEBUG: Uploading '{temp_file_path}' to Google AI...")
                gemini_file_object = genai.upload_file(path=temp_file_path, display_name=uploaded_file.name)
                print(f"DEBUG: Upload successful. File Name: {gemini_file_object.name}, URI: {gemini_file_object.uri}")

                # Store the File object in session state as a list
                st.session_state.current_file_context = [gemini_file_object]

                st.success(f"✅ File '{uploaded_file.name}' uploaded and ready!")
                st.markdown(f"**Current File Context:** {uploaded_file.name}")
                # Add a message to chat indicating file is ready (optional clear existing first)
                # st.session_state.messages = [] # Uncomment to clear chat on new file upload
                st.session_state.messages.append({"role": "assistant", "content": f"File '{uploaded_file.name}' is ready. What would you like to know about it?"})

            except Exception as e:
                st.error(f"❌ Error processing file '{uploaded_file.name}': {e}")
                traceback.print_exc()
                st.session_state.current_file_context = None # Clear context on error
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        print(f"DEBUG: Removed temporary file: {temp_file_path}")
                    except Exception as e_rem:
                        print(f"Warning: Could not remove temporary file {temp_file_path}: {e_rem}")
                # Indicate processing finished
                if "file_being_processed" in st.session_state:
                     del st.session_state["file_being_processed"]


    # Display current context if no new file is uploaded but context exists
    elif st.session_state.current_file_context:
        try:
            file_display_name = st.session_state.current_file_context[0].display_name or st.session_state.current_file_context[0].name
            st.markdown(f"**Current File Context:** {file_display_name}")
        except Exception as e:
             st.markdown("**Current File Context:** Active (error fetching details)")
             print(f"Error accessing file context details: {e}")
    else:
        st.info("Upload a file to analyze its content.")

# --- Main Chat Interface ---
st.markdown("---") # Separator

# Display chat messages from history
# Use a container for chat messages for better scrolling/layout
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input - THIS IS ALWAYS AVAILABLE using st.chat_input
if prompt := st.chat_input("Ask Alpha about the document or general questions..."):
    # Add user message immediately to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container: # Make sure it appears in the container
        with st.chat_message("user"):
            st.markdown(prompt)

    # --- Call Alpha's chat method ---
    # Display thinking indicator within the assistant message area
    with chat_container: # Display assistant response in the same container
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Create a placeholder for the streaming/final response
            with st.spinner("Alpha is thinking..."):
                try:
                    # Retrieve the current file context (it might be None or a list with one File)
                    # The type hint is Optional[List[File]]
                    file_context_to_send: Optional[List[File]] = st.session_state.get('current_file_context', None)

                    context_info = f"(File Context: {file_context_to_send[0].display_name})" if file_context_to_send else "(No File Context)"
                    print(f"DEBUG: Calling chat_with_gemini. Prompt: '{prompt[:50]}...' {context_info}")

                    # CORE CALL: Pass the prompt and the current file context
                    response_text = alpha.chat_with_gemini(prompt, file_context=file_context_to_send)

                    # Display the final response
                    message_placeholder.markdown(response_text)

                    # Add assistant response to chat history AFTER getting it
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except TimeoutException as e:
                     error_msg = f"⚠️ Alpha took too long to respond. Please try again or simplify your request. ({e})"
                     message_placeholder.error(error_msg)
                     st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {e}"
                    message_placeholder.error(error_msg)
                    traceback.print_exc()
                    # Add error details to history as well
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})

# # Optional: Add a button to clear chat history and context
# if st.sidebar.button("Clear Chat and Context"):
#     st.session_state.messages = []
#     st.session_state.current_file_context = None
#     # Potentially delete associated Gemini files if needed (more complex)
#     print("INFO: Chat history and file context cleared.")
#     st.rerun() # Rerun the app to reflect the cleared state