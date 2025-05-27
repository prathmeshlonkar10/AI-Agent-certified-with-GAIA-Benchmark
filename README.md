# 🧠 GAIA Benchmark Agent – HuggingFace AI Agent Certification

This repository contains my complete submission for the **[Hugging Face AI Agent Certification](https://huggingface.co/learn/agents-course)**, where I built a fully autonomous AI agent capable of solving real-world reasoning tasks from the **GAIA (General AI Assistant) Benchmark**.


## 🎓 About the Certification (Just for Context!)

The **Hugging Face AI Agents Course** is a free, certified program designed to teach the theory, design, and practical application of AI agents. The course covers:

- **Agent Fundamentals**: Understanding tools, thoughts, actions, observations, LLMs, messages, special tokens, and chat templates.
- **Frameworks for AI Agents**: Hands-on experience with popular libraries and frameworks such as `smolagents`, `LlamaIndex`, and `LangGraph`.
- **Use Cases**: Building real-world applications and contributing to the community.
- **Final Project**: Developing an AI agent for the GAIA benchmark test and competing on a leaderboard.

The course is structured into units containing written materials, coding notebooks, and interactive quizzes. Completing the full course involves building and evaluating an AI agent using a subset of the GAIA benchmark. 

So that's the overall context.


## 🚀 What I Built

I developed an intelligent agent using:

- 🤖 [`smolagents`](https://github.com/huggingface/smolagents)
- 🔍 Tool-augmented search with **DuckDuckGo** & **Wikipedia**
- 🧠 A custom prompt system aligned to GAIA's strict answer format
- 🔄 Task-aware context injection (e.g., file parsing, OCR, YouTube transcription)
- 📜 Submission pipeline for automatic evaluation and scoring

> **Achievement**: The agent was evaluated on **20 Level 1 GAIA benchmark tasks** and successfully submitted to Hugging Face's scoring API.


## 🧪 GAIA Benchmark: What Is It?

The **GAIA benchmark** is a rigorous test suite designed to assess the general reasoning, retrieval, and tool-use capabilities of AI agents. Tasks require:

- Real-time web search
- Information synthesis
- Working with auxiliary file data (CSV, Excel, MP3, PNG, etc.)
- Interpreting YouTube links, performing OCR, and more

It's used to evaluate general-purpose AI assistants and is modeled as a stepping stone toward AGI-level capabilities.


## 💡 How the Agent Works

### 🛠️ Tools Integrated

| Tool               | Purpose                                 |
|--------------------|------------------------------------------|
| `DuckDuckGoSearch` | Web search queries for factual data      |
| `WikipediaSearch`  | Specific topic lookups                   |
| `Whisper`          | Audio + YouTube transcription            |
| `Tesseract OCR`    | Extract text from `.png` images          |
| `pandas`           | Preview and parse `.csv` or `.xlsx` data |

### 🧩 System Prompt Formatting

The agent adheres strictly to GAIA's required output format:

No extra words, units, or explanations — just the direct result, optimized for exact-match evaluation. 


## 🖼️ Screenshots (📷)

I've included screenshots below showing:

- Each GAIA task question
- The agent-generated response
- My final submission and result

![Screenshot 2025-05-26 185338](https://github.com/user-attachments/assets/c849a13c-1875-4e43-b532-942960b45f3a)
![Screenshot 2025-05-26 185402](https://github.com/user-attachments/assets/26f804b5-88f7-4037-99e6-6ac07cea97a9)
![Screenshot 2025-05-26 185442](https://github.com/user-attachments/assets/462f5607-5dd2-4883-a45f-ba873169c402)
![Screenshot 2025-05-26 185501](https://github.com/user-attachments/assets/1b2c4232-33a8-4dcf-a251-f5437d0141e9)
![Screenshot 2025-05-26 185514](https://github.com/user-attachments/assets/502dd3d6-6042-4696-8d92-6698da7ac20f)
![Screenshot 2025-05-26 185529](https://github.com/user-attachments/assets/cf406005-994e-44b9-ac1b-81f3afa0d206)
![Screenshot 2025-05-26 185544](https://github.com/user-attachments/assets/d4880548-e24b-49e4-b508-3b077de364af)
![Screenshot 2025-05-26 185556](https://github.com/user-attachments/assets/40679944-631f-47ef-8250-d90a9c85c814)



## 🧾 Submission Details

- ✅ Authenticated via Hugging Face OAuth
- ✅ Pulled questions dynamically from HF API
- ✅ Automatically attached auxiliary files
- ✅ Posted all answers to the `/submit` endpoint
- ✅ Received official score & result breakdown


## 📊 My GAIA Result

> 📈 **Final Score:** _✓ [7/20 correct]_  
> 🏅 **Certification Status:** _Passed with ≥ 30% as required_  
> 🧾 View: [[Certificate Link](https://drive.google.com/file/d/1e9cqwTSaD541Gjn2Yya1IPRzM89AzozP/view?usp=sharing)]


## 🧠 Reflections

This project pushed me to design a system that combines:

- **LLM reasoning**
- **Real-time retrieval**
- **Multi-modal input understanding**
- **Structured output formatting**

It simulated real-world agent deployment scenarios and was an excellent hands-on exercise for tool-augmented agents. I would say, more than the satisfaction of obtaining the certification, I'm happy to have learned all these theories and concepts. And to have applied & implemented them!

