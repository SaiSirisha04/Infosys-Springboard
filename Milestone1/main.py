from speech_text import recognize_speech, text_to_speech
from chat_bot import get_llm_response
from sentiment_analyzer import analyze_sentiment  # Hypothetical module for sentiment analysis

def process_voice_interaction():
    # Instructional prompt for AI Sales Assistant
    base_instruction = (
        "You are a Real-Time AI Sales Intelligence and Sentiment-Driven Deal Negotiation Assistant. "
        "Your role is to assist users with product recommendations, answer questions, and negotiate deals in real-time. "
        "Use sentiment analysis to adapt your tone and strategy based on the user's mood. "
        "Handle objections effectively, negotiate bulk purchases by offering tiered discounts, and provide clear, persuasive responses. "
        "Always maintain a friendly, professional tone while focusing on customer satisfaction and driving sales.\n\n"
        "Example interactions:\n"
        "1. Customer: 'This seems too expensive.'\n"
        "   AI Sales Assistant: 'I understand your concern. Let me highlight how this product offers excellent value for its price...'\n"
        "2. Customer: 'Do you offer discounts for bulk orders?'\n"
        "   AI Sales Assistant: 'Absolutely! For bulk purchases, we offer tiered discounts...'\n"
        "3. Customer: 'How do I know this will work for me?'\n"
        "   AI Sales Assistant: 'That’s a great question! This product has been tested by hundreds of satisfied customers...'\n"
    )
    context_history = f"{base_instruction}\n"
    MAX_CONTEXT_LENGTH = 500  # Limit for efficient processing

    print("Real-Time AI Sales Interaction. Say 'exit' to quit.")

    while True:
        # Capture user input via speech
        user_input = recognize_speech()
        if user_input is None:
            continue

        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Ending the conversation. Goodbye!")
            break

        # Analyze the user's sentiment
        sentiment = analyze_sentiment(user_input)  # Returns 'positive', 'neutral', or 'negative'

        # Append sentiment analysis to the context history
        context_history += f"Customer (Sentiment: {sentiment}): {user_input}\n"

        # Ensure context doesn't exceed the maximum length
        if len(context_history) > MAX_CONTEXT_LENGTH:
            context_history = base_instruction + context_history[-MAX_CONTEXT_LENGTH:]

        # Adjust base prompt for negotiation based on sentiment
        if sentiment == "positive":
            additional_instruction = "The customer is in a good mood. Focus on upselling or offering premium products."
        elif sentiment == "neutral":
            additional_instruction = "The customer seems undecided. Provide clear, detailed explanations and highlight product value."
        elif sentiment == "negative":
            additional_instruction = "The customer seems hesitant or unhappy. Respond empathetically, address concerns, and offer reassurance."

        # Combine base instruction with additional guidance
        full_prompt = f"{base_instruction}\n\n{additional_instruction}\n{context_history}"

        # Get LLM response
        llm_response = get_llm_response(full_prompt)
        if llm_response:
            print(f"AI Sales Assistant: {llm_response}")

            # Append the assistant's response to context history
            context_history += f"AI Sales Assistant: {llm_response}\n"

            # Convert the response to speech
            text_to_speech(llm_response)

if __name__ == "__main__":
    process_voice_interaction()