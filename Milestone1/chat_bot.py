import google.generativeai as Gen_Ai

# Configure Gemini API
API_KEY = "AIzaSyABKHOKaU40kMsQRMpeD0G9r-jneGj4u0s"
Gen_Ai.configure(api_key=API_KEY)

# Set up LLM
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

model = Gen_Ai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)
chat = model.start_chat()

def get_llm_response(context):
    try:
        response = chat.send_message(context)
        if hasattr(response, 'text'):
            llm_response = response.text.strip()
            return llm_response
        else:
            llm_response = "An error occurred in generating a response."
    except Exception as e:
        print(f"Error with LLM interaction: {e}")
        return None