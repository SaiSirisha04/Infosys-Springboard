import google.generativeai as Gen_Ai

API_KEY = "AIzaSyABKHOKaU40kMsQRMpeD0G9r-jneGj4u0s"

# Configure the Gemini API
Gen_Ai.configure(api_key=API_KEY)

# Set up the generation configuration
generation_config = {
    "temperature": 0.7,  # Moderate creativity
    "top_p": 0.9,        # Nucleus sampling
    "top_k": 40,         # Valid value for top_k (within range)
    "max_output_tokens": 512,  # Shorter responses
    "response_mime_type": "text/plain",
}

# Initialize the Gemini generative model
model = Gen_Ai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Start a chat session
chat = model.start_chat()

def process_sequence_by_sequence(transcript):
    context_history = ""  # Keeps the ongoing conversation history

    for sequence in transcript:
        print(f"\nProcessing sequence: {sequence}")
        try:
            # Add the new sequence to the conversation history
            context_history += f"Customer: {sequence}\n"
            
            # Send the updated context to the model
            response = chat.send_message(context_history)
            
            # Print the model's response
            print(f"Model Response: {response.text}")
        except Exception as e:
            print(f"Error processing sequence: {e}")
            continue

# Example usage
if __name__ == "__main__":
    # Simulated call transcript
    call_transcript = [
        "give an analysis on 2024 Andhra Pradesh elections"
    ]
    
    process_sequence_by_sequence(call_transcript)


