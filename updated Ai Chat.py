import os
import requests
from bs4 import BeautifulSoup
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
import anthropic
import gradio as gr 

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')


if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

claude = anthropic.Anthropic()

google.generativeai.configure()

system_message = "You are a helpful assistant"

def stream_gemini(prompt):
    gemini = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_message
    )
    
    stream = gemini.generate_content(prompt, stream=True)
    
    result = ""
    for chunk in stream:
        try:
            part = chunk.text
            if part:
                result += part
                yield result   
        except Exception as e:
            print("Chunk error:", e)

def stream_claude(prompt):
    result = claude.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response

view = gr.Interface(
    fn=stream_model,
    inputs=[gr.Textbox(label="Your message:"), gr.Dropdown(["GPT", "Claude","Gemini"], label="Select model", value="GPT")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never"
)
view.launch()
