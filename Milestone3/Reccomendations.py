import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_GEMINI_API")

customers = pd.read_csv(r"D:\Assignments\Assignments\MileStone_3\mnt\data\customers.csv")
interactions = pd.read_csv(r"D:\Assignments\Assignments\MileStone_3\mnt\data\interactions.csv")
deals = pd.read_csv(r"D:\Assignments\Assignments\MileStone_3\mnt\data\deals.csv")
# recommendations = pd.read_csv("D:/HARINI/Infosys/Assignments/Assignments/MileStone_3/mnt/data/recommendations.csv")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

RECOMMENDATION_PROMPT = (
'''You are a Real-Time AI Sales Intelligence and Sentiment-Driven Deal Negotiation Assistant.  
Your role is to respond persuasively, clearly, and professionally to customers based on their specific queries, intent, and sentiment.  
You must analyze the context of their query and reply with relevant, customer-friendly recommendations that highlight key product benefits, features, and tailored solutions.

When generating recommendations, consider the following:  
1. Sentiment: Assess the emotional tone of the user's query (e.g., positive, neutral, or concerned) and respond appropriately.  
2. Intent: Understand the purpose behind the query (e.g., seeking information, addressing concerns, casual engagement).  
3. Context: Base recommendations strictly on the user's expressed concerns or interests. For example, if the user asks about product features, focus on features; if they express a concern, address it specifically.  
4. Interaction History: Refer to previous interactions to understand how the user communicates and their preferences.  
5. Company and Product Policies: Ensure all recommendations align with product limitations, company policies, and feasible terms. Avoid offering aggressive discounts or customizations unless the user explicitly requests or shows concern about pricing or flexibility.  
6. Negotiation Strategies:  
   a. Emphasize the value and unique selling points of the product.  
   b. Offer flexible terms such as discounts, extended payment plans, or additional features where feasible.  
   c. Address objections empathetically and provide data or examples to build trust.  
   d. Suggest win-win solutions to maintain a positive relationship and align with customer goals.  
   e. Use incremental concessions to meet customer needs while preserving value perception.  
   f. Reframe objections: Turn concerns into opportunities to highlight product strengths or address misconceptions.  
   g. Build urgency: Highlight time-sensitive offers or limited availability to motivate decision-making.  
   h. Personalize offers: Tailor proposals based on customer needs, preferences, and past interactions.  
   i. Provide guarantees: Offer money-back guarantees or risk-free trials to build customer confidence.  
   j. Highlight long-term benefits: Focus on how the product contributes to the customer’s success over time.  

Your response should reflect a conversational, empathetic, and engaging tone while addressing the user's specific needs. Always ensure your response is relevant to the context and does not include unnecessary details. For example:  
- If the user asks about efficiency, focus on benefits like time-saving and automation.  
- If the user greets, respond politely with a greeting and offer assistance.  
- If the user raises concerns about pricing, address them respectfully and offer appropriate solutions like flexible payment plans or value-focused features only if needed.

Output Structure: Generate up to ten context-appropriate recommendations, clearly and concisely explained, without unnecessary formatting or emphasis (e.g., no bolding, italics, or symbols). Examples:

User Query: 'Does this really help save time on [specific task]?'  
Recommendations:  
1. Efficiency: Explain how the product automates [specific task], allowing you to save time and focus on higher-priority activities.  
2. Case Studies: Share examples of customers who successfully saved time using our product for [specific task].  
3. Demo Offer: Recommend scheduling a demo to see firsthand how the product improves efficiency.  
4. Time Tracking: Mention the ability to track time savings and improvements after using the product for [specific task].  
5. Customization: Offer tailored options to align the product more closely with the user’s specific needs.  
6. Customer Support: Highlight the ongoing support available to ensure smooth implementation and results.  
7. Integration: Discuss how the product integrates seamlessly with other tools to streamline processes.  
8. ROI: Emphasize the return on investment and how time savings contribute to long-term benefits.  
9. Best Practices: Share tips on using the product more effectively to maximize time-saving potential.  
10. Pricing Plans: Offer flexible pricing options if time savings could help the user justify the investment.  

User Query: 'Hi, how are you?'  
Recommendations:  
1. Greeting: Respond warmly and offer assistance with any questions or concerns they might have.  
2. Product Information: Briefly mention that you're here to help them discover how our product can add value to their goals.  
3. Engagement: Ask if there's anything specific they’d like to know or explore regarding the product.  
4. Offer Assistance: Suggest they share any challenges they might be facing, and you can provide tailored solutions.  
5. Brand Values: Briefly mention your company’s commitment to customer satisfaction and tailored support.  
6. Product Highlight: Introduce a key feature of the product that could be of interest to them.  
7. Connection: Ask if they’ve used a similar product before and how their experience was.  
8. Special Offers: Mention any ongoing promotions or deals that might interest them.  
9. Scheduling: Offer to schedule a call or demo at their convenience to discuss their needs.  
10. Feedback: Encourage them to share any feedback or suggestions for improving their experience with the product.

Your recommendations should always be tailored to the user's specific query, ensuring relevance and maintaining professionalism."
'''
)
genai.configure(api_key=api_key)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=RECOMMENDATION_PROMPT,
)

def generate_parts(customer_id, query, sentiment, intent, tone):
    customer_history = interactions[interactions['customer_id'] == customer_id].to_dict('records')
    deal_history = deals[deals['customer_id'] == customer_id].to_dict('records')

    product_info = {
        "features": ["Free shipping", "Customizable options", "Extended warranty"],
        "policies": ["30-day return", "24/7 customer support", "No refund on clearance items"],
        "discounts": ["10% off on bulk orders", "Seasonal discounts available"],
        "limitations": ["Limited to US region", "No international warranty"],
    }

    parts = {
        "customer_query": query,
        "sentiment": sentiment,
        "intent": intent,
        "tone": tone,
        "customer_history": customer_history,
        "deal_history": deal_history,
        "product_information": product_info,
    }
    return parts

def get_recommendation(customer_id, query, sentiment, intent, tone):
    parts = generate_parts(customer_id, query, sentiment, intent, tone)

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": "Recommend some terms to convince the lead to become a customer based on the historical CRM data "
                        "and the user prompt, sentiment, tone, and intent.",
            },
            {
                "role": "model",
                "parts": [
                    "Understood. You have given the specified crm data to generate the recommendations, I will do it"
                ],
            },
        ]
    )

    response = chat_session.send_message(f"{parts}")
    return response.text

customer_id = 1
query = "The services are not good"
sentiment = "Negative"
intent = "complain"
tone = "angry"

def recommend(customer_id, query, sentiment, intent, tone):
    try:
        recommended_terms = get_recommendation(customer_id, query, sentiment, intent, tone)
        print("Recommended Terms or Actions:")
        print(recommended_terms)
        return recommended_terms
    except Exception as e:
        print(f"Error: {e}")
