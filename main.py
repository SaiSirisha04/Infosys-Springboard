import csv
import gradio as gr
import os
import google.generativeai as genai
from dotenv import load_dotenv
from MileStone_1.speech_to_text import record_audio, transcribe_audio
from MileStone_1.generate_response import generate_response
from MileStone_1.text_to_speech import text_to_speech
from MileStone_2.Analyze_user_audio import analyze_audio
from MileStone_2.Analyze_user_statement import Analyze_text
from MileStone_3.Reccomendations import recommend
from MileStone_3.PostCallAnalysis import generate_post_call_analysis
import time
import datetime

load_dotenv()
api_key = os.getenv("GOOGLE_GEMINI_API")
genai.configure(api_key=api_key)

def read_csv_content(csv_file):
    conversation = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) == 2: 
                conversation.append(f"{row[0]}: {row[1]}")
    return " ".join(conversation)

def get_next_interaction_id(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)
        interaction_ids = [int(row[0]) for row in reader]
        return max(interaction_ids, default=0) + 1

def extract_negotiation_tips(recommendation_prompt):
    tips = []
    lines = recommendation_prompt.split("\n")
    extracting = False
    for line in lines:
        if "Negotiation Strategies:" in line:
            extracting = True
            continue
        if extracting:
            if line.strip() == "":
                break
            tips.append(line.strip(" -\t"))
    return tips

def analyze_audio_with_error_handling(audio_path):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"The audio file at {audio_path} does not exist.")
    try:
        audio_file = genai.upload_file(audio_path, mime_type="audio/wav")
        return {"sentiment": "Neutral", "tone": "Calm", "intent": "General Inquiry"}
    except Exception as e:
        raise RuntimeError(f"An error occurred during audio processing: {e}")

def process_user_input(audio_file, csv_file, interaction_csv_file, customer_id):
    user_input = transcribe_audio(audio_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as file, \
         open(interaction_csv_file, mode='a', newline='', encoding='utf-8') as interaction_file:

        writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_ALL)
        interaction_writer = csv.writer(interaction_file, quotechar='"', quoting=csv.QUOTE_ALL)

        writer.writerow(["User", user_input])
        file.flush()

        if "exit" in user_input.lower():
            ai_response = "Goodbye! Have a great day!"
            writer.writerow(["AI", ai_response])
            file.flush()
            full_conversation = read_csv_content(csv_file)
            analysis = Analyze_text(full_conversation)
            generate_post_call_analysis(full_conversation, analysis, customer_id)
            return "", "", ai_response, ""

        summary = analyze_audio_with_error_handling(audio_file)
        interaction_id = get_next_interaction_id(interaction_csv_file)
        interaction_writer.writerow([
            interaction_id, customer_id, datetime.datetime.now(), "call",
            user_input, summary["sentiment"], summary["tone"], summary["intent"]
        ])
        interaction_file.flush()

        recommended_terms = recommend(
            customer_id, user_input,
            sentiment=summary["sentiment"], intent=summary["intent"], tone=summary["tone"]
        )

        recommendation_prompt = recommend.__globals__["RECOMMENDATION_PROMPT"]
        negotiation_tips = "\n".join(extract_negotiation_tips(recommendation_prompt))

        ai_response = generate_response(
            user_input, recommended_terms=recommended_terms,
            sentiment=summary["sentiment"], intent=summary["intent"], tone=summary["tone"]
        )

        writer.writerow(["AI", ai_response])
        file.flush()

        return recommended_terms, negotiation_tips, ai_response, summary["sentiment"]

def main_ui():
    csv_file = "conversation_log.csv"
    interaction_csv_file = r"D:\Assignments\Assignments\MileStone_3\mnt\data\interactions.csv"
    customer_id = 1

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["Speaker", "Message"])

    def gradio_response(audio_file):
        return process_user_input(audio_file, csv_file, interaction_csv_file, customer_id)

    interface = gr.Interface(
        fn=gradio_response,
        inputs=gr.Audio(type="filepath"),
        outputs=[
            gr.Textbox(label="Recommendations"),
            gr.Textbox(label="Negotiation Tips"),
            gr.Textbox(label="AI Response"),
            gr.Textbox(label="Sentiment Analysis")
        ],  
        title="Real-Time AI Sales Assistant",
        description="An AI-driven tool for sales assistance and negotiation tips."
    )

    interface.launch()

if __name__ == "__main__":
    main_ui()
