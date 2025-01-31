import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta

# Gemini API configuration - Get API key from environment
API_KEY = os.getenv("GOOGLE_GEMINI_API")
if not API_KEY:
    raise ValueError("Please set the GOOGLE_GEMINI_API environment variable in the .env file")
genai.configure(api_key=API_KEY)

# Model Configuration
MODEL_NAME = "gemini-1.5-pro"

# Load deal data
def load_deal_data():
    deals_data = """
    deal_id,customer_id,deal_stage,proposed_terms,negotiation_notes,deal_value,closing_date
1,1,Interest,10% discount,Awaiting confirmation,5000,2025-01-15
2,2,Proposal,Extended warranty,Customer wants flexibility,12000,2025-01-20
3,3,Negotiation,Free shipping,Discussing shipping costs,8000,2025-01-25
4,4,Closed-Won,Custom feature,Terms accepted,15000,2025-01-30
5,5,Interest,Return policy,Considering competitors,4000,2025-02-01
6,6,Proposal,Premium access,Upselling premium features,20000,2025-02-05
7,7,Closed-Lost,Flexible billing,Customer not interested,10000,2025-02-10
8,8,Negotiation,Discounted pricing,Price concern unresolved,6000,2025-02-15
9,9,Interest,Loyalty perks,Exploring options,3000,2025-02-20
10,10,Closed-Won,Early delivery,Satisfied with terms,25000,2025-02-25
    """
    deals = {}
    lines = deals_data.strip().split('\n')
    header = lines[0].split(',')
    for line in lines[1:]:
        values = line.split(',')
        deal_dict = dict(zip(header, values))
        deals[int(deal_dict["customer_id"])] = deal_dict
    return deals

# Function to get deal data for customer_id
def get_deal_data(customer_id, deals):
    return deals.get(customer_id, None)

# Generate call summary
def generate_summary(transcription, summary_of_call, deal_data, customer_id):
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"""
        Analyze the following conversation and provide:

        1. Summary of call insights, highlighting key points, sentiment, topics, deal stage, and negotiation notes.
        2. JSON structured output with:
            - Overall sentiment (positive, negative, neutral).
            - Key tones expressed.
            - Key topics discussed.
            - Actionable follow-up recommendations.
            - Deal stage inference.
            - Negotiation notes.

        Conversation: {transcription}
        Deal Information: {deal_data}
        Call Summary: {summary_of_call}

        Output format:
        Summary: [concise text summary]
        JSON Output:
        {{
            "sentiment": "example",
            "tone": ["example", "example"],
            "key_topics": ["example", "example"],
            "recommendations": ["example", "example"],
            "deal_stage": "example",
            "negotiation_notes": "example"
        }}
    """
    
    try:
        response = model.generate_content(prompt)
        response.resolve()
        
        summary_match = re.search(r"Summary:\s*(.*?)\s*JSON Output:", response.text, re.DOTALL)
        json_match = re.search(r"(?:json)?\s*({.+?})\s*", response.text, re.DOTALL)

        summary_text = summary_match.group(1).strip() if summary_match else "No summary extracted"

        if json_match:
            json_string = json_match.group(1)
            try:
                analysis = json.loads(json_string)
                return format_summary(analysis, summary_text, customer_id)
            except json.JSONDecodeError:
                return None
        else:
            return None
    except Exception as e:
        return None

# Formatting function
import csv

# Formatting function
def format_summary(analysis, summary_text, customer_id):
    sentiment = analysis.get("sentiment", "Unknown")
    tone = analysis.get("tone", [])
    topics = analysis.get("key_topics", [])
    recommendations = analysis.get("recommendations", [])
    deal_stage = analysis.get("deal_stage", "Unknown")
    negotiation_notes = analysis.get("negotiation_notes", "No notes provided")
    closing_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
    # Prepare new deal data
    new_row = {
        "deal_id": get_next_deal_id(),  # Auto-generate new deal ID
        "customer_id": customer_id,  # Example placeholder, adjust as needed
        "deal_stage": deal_stage,
        "proposed_terms": ", ".join(recommendations),
        "negotiation_notes": negotiation_notes,
        "deal_value": 5000,  # Placeholder value, update logic if necessary
        "closing_date": closing_date  # Placeholder date, update logic if necessary
    }

    # Append new row to deals.csv
    try:
        deals = "D:/Codes/Deep_Learning/Infosys_internship/Real-Time-AI-Sales-Intelligence-and-Sentiment-Driven-Deal-Negotiation-Assistant/Assignments/MileStone_3/mnt/data/deals.csv"
        with open(deals, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=new_row.keys())

            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()
                
            writer.writerow(new_row)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

    summary = f"""
    Post-Call Summary:
    ------------------
    Summary: {summary_text}

    Sentiment: {sentiment}
    Tone: {', '.join(tone)}
    Key Topics: {', '.join(topics)}
    Deal Stage: {deal_stage}
    Negotiation Notes: {negotiation_notes}
    Recommendations:
    - {chr(10).join(recommendations)}
    """
    return summary

# Function to get the next deal ID from the CSV file
def get_next_deal_id():
    try:
        deals = "D:/Codes/Deep_Learning/Infosys_internship/Real-Time-AI-Sales-Intelligence-and-Sentiment-Driven-Deal-Negotiation-Assistant/Assignments/MileStone_3/mnt/data/deals.csv"
        with open(deals, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            deal_ids = [int(row["deal_id"]) for row in reader]
            return max(deal_ids) + 1 if deal_ids else 1
    except (FileNotFoundError, ValueError):
        return 1


# Main function
def generate_post_call_analysis(transcription, audio_analysis, customer_id):
    deals = load_deal_data()
    deal_data = get_deal_data(customer_id, deals)
    
    if not deal_data:
        return

    summary = generate_summary(transcription, audio_analysis, deal_data, customer_id)
    if summary:
        print(summary)

# Entry point
if __name__ == "__main__":
    transcription = "Hello, I am interested in your product. Can you tell me more about it?"
    audio_analysis = "Sentiment: Neutral, Tone: Neutral, Intent: Asking a question"
    customer_id = 1

    generate_post_call_analysis(transcription, audio_analysis, customer_id)
