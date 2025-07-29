from flask import Flask, render_template, request, send_from_directory
from langchain_groq import ChatGroq
import os, re

app = Flask(__name__)

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model="llama3-8b-8192"
)

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json', mimetype='application/manifest+json')

# Prompt to extract the main topic
TOPIC_EXTRACTION_PROMPT = """
You are a helpful assistant. Extract the core technical or conceptual topic from the following question.
Reply in 5–7 words, without extra explanation.

Question: "{user_question}"
Topic:
"""

# Enhanced ELI5 prompt
ELI5_PROMPT_TEMPLATE = """
You are an expert at explaining complex ideas in a very simple way.

The user is curious about the topic: **{main_topic}**

They asked: "{user_question}"

Explain this topic in a way a 5-year-old could understand, using **up to 5 cue cards**.

Each cue card should:
- Be short (3–4 lines max)
- Focus on one idea
- Be logically ordered
- Start with a fun title like “What’s Happening?”, “Let’s Pretend”, “A Tiny Example”, etc.

If you use analogies or metaphors (like robots), **make sure follow-up questions relate back to the actual topic ({main_topic})**, not the analogy.

After the cue cards, suggest 3–5 follow-up questions that:
- Explore {main_topic} in more detail
- Help the user learn related or deeper concepts
- Stay on-topic even if the explanation used analogies

Format your response like this:

**Cue Card 1: <title>**
<short explanation>

...

Follow-up Questions:
1. <question 1>
2. <question 2>
3. ...
"""


@app.route("/", methods=["GET", "POST"])
def index():
    cue_cards = []
    followups = []
    user_question = ""

    if request.method == "POST":
        user_question = request.form.get("question", "").strip()
        if user_question:
            try:
                # Step 1: Extract topic
                topic_prompt = TOPIC_EXTRACTION_PROMPT.format(user_question=user_question)
                main_topic = llm.invoke(topic_prompt).content.strip()

                # Step 2: Generate cue cards and followups
                final_prompt = ELI5_PROMPT_TEMPLATE.format(
                    main_topic=main_topic,
                    user_question=user_question
                )
                response = llm.invoke(final_prompt).content.strip()
                cue_cards = parse_cue_cards(response)
                followups = parse_followups(response)

            except Exception as e:
                cue_cards = [{"title": "Oops!", "content": f"Something went wrong: {e}"}]

    return render_template("index.html", cue_cards=cue_cards, followups=followups, user_question=user_question)


def parse_cue_cards(response_text):
    pattern = r"\*\*Cue Card \d+: (.*?)\*\*\n(.*?)(?=\n\*\*Cue Card \d+:|\nFollow-up Questions:|\Z)"
    matches = re.findall(pattern, response_text, re.DOTALL)
    return [{"title": title.strip(), "content": content.strip()} for title, content in matches]


def parse_followups(response_text):
    followup_section = re.search(r"(?i)\*?\*?Follow[- ]?up Questions:?[\*\*]?\s*\n+(.*)", response_text, re.DOTALL)
    if followup_section:
        block = followup_section.group(1)
        lines = re.findall(r"\d+\.\s+(.*)", block)
        return [line.strip() for line in lines if line.strip()]
    return []


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='4000', debug=True)
