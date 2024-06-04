from flask import Flask, request, jsonify
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Retrieve environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if environment variables are loaded correctly
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set correctly")

# Flask app
app = Flask(__name__)

def get_essay_chain():
    """
    Create and return the essay writing chain for the language model.
    """
    prompt_template = """
    Write a detailed essay about the following topic. The essay should be approximately {word_count} words.

    Topic:
    {topic}

    Essay:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "word_count"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

async def generate_essay(topic, word_count):
    """
    Generate an essay for the given topic and word count using the essay chain.
    """
    chain = get_essay_chain()
    response = chain.run({"topic": topic, "word_count": word_count})
    return response

def process_user_input(data):
    """
    Extract the topic and word count from the request data.
    """
    topic = data.get('topic')
    word_count = data.get('word_count')
    return topic, word_count

def create_response_message(response):
    """
    Create a JSON response message from the generated essay.
    """
    return jsonify({"essay": response})

@app.route('/api/write_essay', methods=['POST'])
def write_essay_api():
    """
    API endpoint to handle essay writing requests.
    """
    if request.method == 'POST':
        data = request.json
        topic, word_count = process_user_input(data)

        if topic and word_count:
            try:
                word_count = int(word_count)
            except ValueError:
                return jsonify({"error": "Invalid word count provided"}), 400

            response = asyncio.run(generate_essay(topic, word_count))
            return create_response_message(response)
        else:
            return jsonify({"error": "Topic and word count must be provided"}), 400
    else:
        return jsonify({"error": "Only POST requests are supported"}), 405

if __name__ == "__main__":
    app.run()
