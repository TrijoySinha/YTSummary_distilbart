import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()  # Loads all the environment variables

# Function to get the transcript data from YouTube videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e

# Function to generate the summary using DistilBART
def generate_distilbart_summary(transcript_text, max_words):
    try:
        # Load pre-trained DistilBART model and tokenizer
        model_name = "sshleifer/distilbart-cnn-12-6"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Prepare the input for the model
        inputs = tokenizer.encode("summarize: " + transcript_text, return_tensors="pt", max_length=1024, truncation=True)

        # Approximate tokens based on words (1.33 tokens per word as a heuristic)
        max_tokens = int(max_words * 1.33)

        # Generate the summary
        summary_ids = model.generate(
            inputs,
            max_length=max_tokens,
            min_length=int(max_tokens * 0.5),  # Ensure a minimum length proportional to max_tokens
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Truncate the summary to the exact word count specified by max_words
        summary_words = summary.split()[:max_words]
        final_summary = " ".join(summary_words)
        return final_summary

    except Exception as e:
        return f"Error in generating summary: {e}"

# Streamlit App
st.title("ShortIt")
youtube_link = st.text_input("Enter YouTube Video Link:")

# User can select the summary length in words
summary_length = st.slider("Select Summary Length (in words):", min_value=50, max_value=500, value=250, step=10)

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)

    if transcript_text:
        # Generate the summary using DistilBART
        summary = generate_distilbart_summary(transcript_text, max_words=summary_length)

        st.markdown("## Detailed Notes:")
        st.write(summary)
