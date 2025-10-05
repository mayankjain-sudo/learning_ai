import os
import requests
from flask import Flask, jsonify, request, render_template_string
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    contain = None
    if request.method == "POST":
        try:
            input_message = request.form["input"]
            llm_endpoint = os.environ.get("LLM_ENDPOINT")
            if not llm_endpoint:
                contain = "LLM_ENDPOINT environment variable not set."
            else:
                # Use requests to call Ollama API
                url = llm_endpoint.rstrip('/v1').rstrip('/') + "/api/generate"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": os.environ.get("MODEL_NAME"),
                    "prompt": input_message,
                    "stream": False
                }
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                contain = data.get("response", "No contain generated.")
        except Exception as e:
            print(f"Error: {e}")
            contain = "An error occurred while generating the contain."
            
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MJ GPT</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 0;
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .container {
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    padding: 40px;
                    max-width: 600px;
                    width: 90%;
                }
                h1 {
                    text-align: center;
                    color: #333;
                    margin-bottom: 30px;
                    font-weight: 600;
                }
                .input-group {
                    margin-bottom: 20px;
                }
                label {
                    display: block;
                    margin-bottom: 8px;
                    color: #555;
                    font-weight: 500;
                }
                input[type="text"] {
                    width: 100%;
                    padding: 12px 16px;
                    border: 2px solid #e1e5e9;
                    border-radius: 8px;
                    font-size: 16px;
                    transition: border-color 0.3s;
                    box-sizing: border-box;
                }
                input[type="text"]:focus {
                    outline: none;
                    border-color: #667eea;
                }
                button {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: transform 0.2s;
                    width: 100%;
                }
                button:hover {
                    transform: translateY(-2px);
                }
                .contain-output {
                    margin-top: 30px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }
                .contain-output h2 {
                    color: #333;
                    margin-top: 0;
                    font-size: 20px;
                }
                .contain-output p {
                    color: #555;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    margin: 0;
                }
                .error {
                    color: #dc3545;
                    font-weight: 500;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Ask Me</h1>
                <form method="post">
                    <div class="input-group">
                        <label for="input">Enter a topic you like to search:</label>
                        <input type="text" id="input" name="input" required placeholder="e.g., love, nature, adventure">
                    </div>
                    <button type="submit">Ask</button>
                </form>
                {% if contain %}
                    <div class="contain-output">
                        <h2>Answer:</h2>
                        <p{% if 'error' in contain.lower() %} class="error"{% endif %}>{{ contain }}</p>
                    </div>
                {% endif %}
            </div>
        </body>
        </html>
    """, contain=contain)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True) 