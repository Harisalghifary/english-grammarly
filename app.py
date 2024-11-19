import os, getpass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import gradio as gr
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the language model
model = ChatOpenAI(model="gpt-4o-mini")

# Function to generate a conversational response with model-based autocorrection
def chatbot_autocorrect_response(input_text: str):
    # Define the prompt asking the model to correct the sentence
    prompt = (
        f"The user said: '{input_text}'. Please correct this sentence if necessary, "
        "and make it sound friendly and casual. Acknowledge the correction and make it sound "
        "like an American native conversation. If appropriate, make it sound like an IELTS 9.0 level response. "
        "If sentences are in Indonesian, translate them to sound like native American conversation. "
        "Please only respond with the corrected sentence. If nothing needs to be changed, repeat the sentence."
    )
    human_message = HumanMessage(content=prompt)
    response = model.invoke([human_message])
    return response.content

# Function to provide vocabulary or sentence explanations
def chatbot_explanation_response(input_text: str):
    # Define the prompt to give explanations and examples
    prompt = (
        f"The user is asking for a detailed explanation of the following phrase or sentence: '{input_text}'. "
        "Please provide an explanation of the vocabulary or sentence, including definitions, usage examples, and "
        "similar expressions or structures. Make it clear and easy to understand, offering alternative ways "
        "to express the same idea if possible."
    )
    human_message = HumanMessage(content=prompt)
    response = model.invoke([human_message])
    return response.content

# Gradio interface setup
def gradio_chatbot(input_text, explanation_text):
    # Get responses for autocorrection and explanation
    autocorrect_response = chatbot_autocorrect_response(input_text)
    explanation_response = chatbot_explanation_response(explanation_text)
    return autocorrect_response, explanation_response

# Launch Gradio interface with two text inputs
interface = gr.Interface(
    fn=gradio_chatbot,
    inputs=["text", "text"], 
    outputs=[gr.Textbox(label="Corrected Sentence"), gr.Markdown(label="Explanation")], 
    title="Chatbot with Auto-Correction and Vocabulary Explanations",
    description="Enter a sentence for autocorrection and another for a vocabulary or sentence explanation."
) 
interface.launch()