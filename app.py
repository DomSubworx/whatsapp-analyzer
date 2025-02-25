import streamlit as st
import re
import pandas as pd
import openai
import os
from transformers import pipeline

# OpenAI API-Key aus Umgebungsvariable laden
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sentiment-Analyse-Modell ohne torch oder tensorflow
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    """ Sentiment-Analyse für Nachrichten """
    result = sentiment_pipeline(text[:512])  # Maximale Länge begrenzen
    return result[0]['label']

def parse_whatsapp_chat(chat_text):
    """ WhatsApp-Chat Datei parsen und strukturieren """
    pattern = r'\[(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}:\d{2})\] ([^:]+): (.+)'
    messages = []
    
    for line in chat_text.split("\n"):
        match = re.match(pattern, line)
        if match:
            date, time, sender, message = match.groups()
            messages.append([date, time, sender, message])
    
    df = pd.DataFrame(messages, columns=['Datum', 'Uhrzeit', 'Absender', 'Nachricht'])
    df['Timestamp'] = pd.to_datetime(df['Datum'] + ' ' + df['Uhrzeit'], format='%d.%m.%y %H:%M:%S')
    df.drop(columns=['Datum', 'Uhrzeit'], inplace=True)

    # Sentiment-Analyse hinzufügen
    df['Sentiment'] = df['Nachricht'].apply(analyze_sentiment)

    return df

import openai

client = openai.OpenAI()

def analyze_relationship(df):
    """ GPT-4o Analyse der Beziehungsdynamik """
    chat_history = "\n".join(df.apply(lambda row: f"{row['Timestamp']} - {row['Absender']}: {row['Nachricht']}", axis=1))

    prompt = f"""
    Hier ist ein WhatsApp-Chatverlauf zwischen zwei Personen. Analysiere die Beziehung basierend auf den Nachrichten.

    1. **Emotionale Grundstimmung:** Ist die Kommunikation eher positiv, neutral oder negativ?
    2. **Kommunikationsmuster:** Wer schreibt häufiger? Wer antwortet schneller?
    3. **Sprachstil & Wortwahl:** Ist die Sprache humorvoll, kurz, lang, formell oder informell?
    4. **Entwicklung:** Gibt es eine erkennbare Veränderung im Gesprächsverlauf?
    5. **Besondere Auffälligkeiten:** Gibt es interessante Muster oder Dynamiken?

    Hier ist der Chatverlauf:
    {chat_history}

    Gib eine detaillierte, aber leicht verständliche Analyse!
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Du bist ein erfahrener Kommunikationsanalyst."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# Streamlit UI
st.title("📱 WhatsApp Chat Analyzer")

# Datei-Upload
uploaded_file = st.file_uploader("Lade deine WhatsApp-Chat-Datei hoch (.txt)", type="txt")

if uploaded_file is not None:
    chat_text = uploaded_file.getvalue().decode("utf-8")
    
    # Chat-Daten verarbeiten
    df = parse_whatsapp_chat(chat_text)
    
    # Ergebnisse anzeigen
    st.subheader("📊 Analysierter Chatverlauf")
    st.dataframe(df)
    
    # GPT-4o Beziehungsanalyse
    st.subheader("🧠 KI-Analyse der Beziehung")
    with st.spinner("Analysiere den Chat..."):
        analysis = analyze_relationship(df)
        st.write(analysis)
