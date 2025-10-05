import os
from flask import Flask, jsonify, request, render_template_string
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("LLM_ENDPOINT"),
    )

@app.route("/", methods=["GET", "POST"])
def index():
    poem = None
    if request.method == "POST":
        try:
            input_message = request.form["input"]
            
            response = client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "allama3.2"),
                messages=[
                    {"role": "system", "content": "You're an AI chatbot which specializes in writing poems."},
                    {"role": "user", "content": input_message}
                ]
            )             
            poem = response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            poem = "An error occurred while generating the poem."
            
    return render_template_string("""
        <html>
            <head>
                <title>MJ GPT</title>
            </head>
            <body>
                <h1>MJ GPT</h1>
                <form method="post">
                
                    <label for="input">Enter a topic for your poem:</label><br>
                    <input type="text" id="input" name="input" required><br><br>
                    <input type="submit" value="Generate Poem">
                </form>
                {% if poem %}
                    <h2>Your Poem:</h2>
                    <p>{{ poem }}</p>
                {% endif %}
            </body>
        </html>
    """, poem=poem)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True) 