import streamlit as st
import os
import sys

# === 1. Streamlit Page Configuration (MUST BE FIRST ST COMMAND) ===
# Moved to the absolute top after imports
st.set_page_config(page_title="Alpha Assistant", page_icon="🤖", layout="wide")

# === 2. Attempt to Import Alpha Class ===
# Combined diagnostic info into the except block for cleaner output on failure
try:
    # Check if the file exists *before* trying to import
    alpha_file_path = os.path.join(os.getcwd(), 'alpha3.py')
    if not os.path.isfile(alpha_file_path):
        # If file doesn't exist, raise a more specific error *before* Python tries importing
        raise FileNotFoundError(f"'alpha3.py' not found in the current directory: {os.getcwd()}")

    from alpha3 import Alpha # Try the import
    # If successful, maybe log to console (optional) or do nothing here
    # print("DEBUG: Successfully imported Alpha.")

except (ImportError, FileNotFoundError, ModuleNotFoundError) as e:
    # Display detailed error message and diagnostics ONLY if import fails
    st.error(f"Fatal Error: Failed to import the 'Alpha' class.")
    st.error(f"Reason: {e}")
    st.error(
        "Please ensure 'alpha.py' exists in the correct directory and has no syntax errors."
    )
    # Show diagnostics within the error message
    st.subheader("Debugging Information:")
    st.write(f"*   Current Working Directory: `{os.getcwd()}`")
    st.write(f"*   Does 'alpha.py' exist here? `{os.path.isfile(alpha_file_path)}`") # Use the path checked earlier
    st.write("*   Python's Search Path (`sys.path`):")
    st.json(sys.path) # Use json for better readability
    st.stop() # Stop the app if import fails

except Exception as e:
    # Catch any other unexpected errors during import
    st.error(f"An unexpected error occurred during the import process: {e}")
    st.stop()


# === 3. Streamlit App Title ===
st.title("🤖 Alpha Assistant")


# === 4. Initialize Alpha and Chat History in Session State ===
# This ensures the AI instance and message history persist across reruns

if 'alpha_instance' not in st.session_state:
    try:
        st.session_state.alpha_instance = Alpha()
        st.toast("Alpha initialized successfully!")
        # Store the model name once if needed
        # Ensure the 'model' attribute and 'model_name' exist in your Alpha class
        if hasattr(st.session_state.alpha_instance, 'model') and hasattr(st.session_state.alpha_instance.model, 'model_name'):
             st.session_state.model_name = st.session_state.alpha_instance.model.model_name
        else:
             st.session_state.model_name = "Model name not available" # Fallback

        st.session_state.messages = [] # Initialize chat history list
        # print("DEBUG: Alpha instance created and stored in session state.") # Keep for debugging if needed

    except Exception as init_error:
        st.error(f"Fatal Error initializing Alpha: {init_error}")
        st.warning("Please ensure your GOOGLE_API_KEY is set correctly (e.g., in environment variables or .env file).")
        st.exception(init_error) # Show the full traceback for initialization errors
        st.stop() # Stop execution if initialization fails


# Display the model name (fetched during initialization)
if 'model_name' in st.session_state:
    st.caption(f"Using model: {st.session_state.model_name}")
else:
     st.caption("Initializing model...") # Fallback message


# === 5. Display Chat History ===
# Iterate through the stored messages and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# === 6. Display Initial Greeting (Optional) ===
# Moved this before the input handling, so it appears above the input box on first run
if not st.session_state.messages:
    st.info("Enter your first message below to start chatting with Alpha!")


# === 7. Handle User Input ===
# Use st.chat_input for a better chat UI
if prompt := st.chat_input("What can I help you with?"):
    # 1. Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check for exit command
    if prompt.lower().strip() in ["0", "exit", "quit"]:
        response = "Goodbye!"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # st.stop() # Usually not needed, let the app wait for new input

    else:
        # 2. Get AI response
        try:
            # Access the persistent Alpha instance
            ai = st.session_state.alpha_instance
            with st.spinner("Alpha is thinking..."):
                # Call the chat method (assuming it exists and handles function calls)
                # Make sure chat_with_gemini returns the string response
                ai_response = ai.chat_with_gemini(prompt.strip())

            # 3. Add AI response to chat history and display it
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)

            # --- REMOVED THIS LINE ---
            # ai.speak(ai_response) # This prints to console, not the Streamlit UI

        except Exception as chat_error:
            st.error(f"An error occurred while getting Alpha's response:")
            st.exception(chat_error) # Show full traceback in the UI for chat errors
            # Optionally add error message to chat history
            error_message = f"Sorry, I encountered an error processing your request."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            # Display error within the chat context *as well*
            # with st.chat_message("assistant"): # Already handled by st.error above? Decide which you prefer.
            #    st.error(error_message)