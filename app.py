import os
import re
import gradio as gr
import requests
import inspect
import pandas as pd
import tempfile
import subprocess
from PIL import Image
import pytesseract
from smolagents import CodeAgent, DuckDuckGoSearchTool, WikipediaSearchTool, LiteLLMModel
from io import StringIO
import whisper
from pytube import YouTube
import langdetect

# URL for the GAIA scoring API
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# Instantiate the model using the LiteLLM wrapper
model = LiteLLMModel(model_id="gpt-4o-mini", api_key=os.environ.get('OPEN_AI_KEY'))

# ------------------------------------------------------
# Utility function to fetch and preprocess task-related files
# ------------------------------------------------------
def fetch_file_content(task_id: str, limit=8000):
    """
    Downloads and interprets the file associated with a task.
    Handles Excel, CSV, audio (MP3), image (PNG), code (.py), and text files.
    Returns a string summary or extracted content depending on the type.
    """
    try:
        url = f"{DEFAULT_API_URL}/files/{task_id}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")

        # Handle Excel file (.xlsx)
        if "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type:
            with open("temp.xlsx", "wb") as f:
                f.write(resp.content)
            df = pd.read_excel("temp.xlsx")
            os.remove("temp.xlsx")
            preview = df.head().to_markdown()
            print(f"Loaded Excel content for task {task_id}: {preview}")
            return f"Excel Preview:\n{preview}"

        # Handle CSV files
        elif "text/csv" in content_type or task_id.endswith(".csv"):
            content = resp.content.decode("utf-8")
            df = pd.read_csv(StringIO(content))
            preview = df.head().to_markdown()
            print(f"Loaded CSV content for task {task_id}: {preview}")
            return f"CSV Preview:\n{preview}"

        # Handle audio transcription using Whisper
        elif "audio/mpeg" in content_type or task_id.endswith(".mp3"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                f.write(resp.content)
                temp_mp3 = f.name
            print(f"Transcribing audio for task {task_id} using Whisper...")
            try:
                whisper_model = whisper.load_model("base")
                result = whisper_model.transcribe(temp_mp3)
                os.remove(temp_mp3)
                print(f"Transcription: {result['text'][:200]}...")
                return f"Transcription:\n{result['text']}"
            except Exception as e:
                print(f"Whisper transcription failed: {e}")
                return "[Audio could not be transcribed.]"

        # Handle image-to-text using Tesseract OCR
        elif "image/png" in content_type or task_id.endswith(".png"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                f.write(resp.content)
                temp_png = f.name
            print(f"Performing OCR on image for task {task_id}...")
            try:
                img = Image.open(temp_png)
                text = pytesseract.image_to_string(img)
                os.remove(temp_png)
                print(f"Extracted text from image: {text[:200]}...")
                return f"Image OCR Text:\n{text}"
            except Exception as e:
                print(f"OCR failed: {e}")
                return "[Image text could not be extracted.]"

        # Handle plain text, Python or JSON files
        elif task_id.endswith(".py") or "text/x-python" in content_type or task_id.endswith(".txt") or "text/plain" in content_type or task_id.endswith(".json"):
            content = resp.text[:limit]
            print(f"Loaded plain text/Python/JSON file content: {content[:200]}...")
            return f"File Content:\n{content}"

        # Fallback for unknown content types
        else:
            content = resp.text[:limit]
            print(f"Fetched file content for task {task_id} (truncated): {content[:200]}...")
            return content

    except Exception as e:
        print(f"Failed to fetch file for task {task_id}: {e}")
        return ""

# ------------------------------------------------------
# Agent class definition to answer questions using tools
# ------------------------------------------------------
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        # Initialize agent with search tools and model
        self.agent = CodeAgent(tools=[DuckDuckGoSearchTool(), WikipediaSearchTool()], model=model)

        # Define system-level instruction that guides the agent
        SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and
        finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated
        list of numbers and/or strings.
        If you are asked for a number, don't use comma to write your number neither use units such as $ or
        percent sign unless specified otherwise.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the
        digits in plain text unless specified otherwise.
        If you are asked for a comma separated list, apply the above rules depending of whether the element
        to be put in the list is a number or a string.
        """
        self.agent.prompt_templates["system_prompt"] = self.agent.prompt_templates["system_prompt"] + SYSTEM_PROMPT

    def __call__(self, question: str, task_id: str = "") -> str:
        """
        Main callable method for the agent.
        Optionally appends file content and YouTube transcriptions to the question before sending it to the model.
        """
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        # Append file context if a task file exists
        if task_id:
            file_context = fetch_file_content(task_id)
            if file_context:
                question += f"\n\nAttached File Content:\n{file_context}"

        # Find YouTube links in the question and transcribe their content
        yt_urls = re.findall(r"https?://(?:www\\.)?(?:youtube\\.com/watch\\?v=|youtu\\.be/)[\\w-]+", question)
        if yt_urls:
            try:
                for url in yt_urls:
                    yt = YouTube(url)
                    stream = yt.streams.filter(only_audio=True).first()
                    temp_path = stream.download(filename_prefix="yt_audio_")
                    whisper_model = whisper.load_model("base")
                    result = whisper_model.transcribe(temp_path)
                    lang = langdetect.detect(result['text'])
                    os.remove(temp_path)
                    question += f"\n\nYouTube Transcription (lang={lang}) from {url}:\n{result['text']}"
            except Exception as e:
                print(f"YouTube transcription failed: {e}")

        # Run the enhanced question through the agent and return the model's output
        final_answer = self.agent.run(question)
        print(f"Agent returning final answer: {final_answer}")
        return final_answer

# ------------------------------------------------------
# Function to run agent on all questions and submit results
# ------------------------------------------------------
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Main evaluation pipeline:
    1. Authenticates user.
    2. Fetches GAIA benchmark questions.
    3. Runs agent to answer each question.
    4. Submits all answers and displays evaluation score.
    """
    space_id = os.getenv("SPACE_ID")
    
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()

        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        
        print(f"Fetched {len(questions_data)} questions.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    results_log = []
    answers_payload = []
    
    print(f"Running agent on {len(questions_data)} questions...")
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        try:
            submitted_answer = agent(question_text, task_id=task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!"
            f"User: {result_data.get('username')}"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)"
            f"Message: {result_data.get('message', 'No message received.')}"
        )

        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
   
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        
        print(status_message)
        results_df = pd.DataFrame(results_log)
        
        return status_message, results_df
    
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

# ------------------------------------------------------
# Gradio UI definition for running the evaluation agent
# ------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        Please clone this space, then modify the code to define your agent's logic within the `BasicAgent` class. 
        Log in to your Hugging Face account using the button below. This uses your HF username for submission. 
        Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

# ------------------------------------------------------
# Main entry point for launching the app
# ------------------------------------------------------
if __name__ == "__main__":
    print(" " + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
