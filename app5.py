from flask import Flask, request, jsonify, render_template
import os
import google.generativeai as genai # type: ignore
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Change to DEBUG for more detailed logs

# Configure Google Generative AI
api_key = "AIzaSyA-1Vr42dijC2GAvnSs_DaJ0mZhMYC__FQ"  # Replace with your actual API key
genai.configure(api_key=api_key)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    try:
        logging.debug(f"Uploading file '{path}' with MIME type '{mime_type}'")
        file = genai.upload_file(path, mime_type=mime_type)
        logging.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        logging.error(f"Error uploading file to Gemini: {e}")
        return None

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

@app.route('/')
def index():
    return render_template('index5.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('/tmp', file.filename)
    os.makedirs('/tmp', exist_ok=True)  # Ensure the /tmp directory exists
    file.save(file_path)
    logging.debug(f"Saved file to path: {file_path}")

    uploaded_file = upload_to_gemini(file_path, mime_type=file.mimetype)
    if uploaded_file is None:
        return jsonify({'error': 'Failed to upload file to Gemini'}), 500

    try:
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        uploaded_file,
                        "Is the given image of food spoilt or not?\nReply in YES or NO\n",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        "NO.",
                    ],
                },
            ]
        )

        response = chat_session.send_message("Is the given image of food spoilt or not?\nReply in YES or NO")
        reasoning = chat_session.send_message("Why do you think so?")

        logging.info(f"Response: {response.text.strip()}")
        logging.info(f"Reasoning: {reasoning.text.strip()}")

        return jsonify({
            'answer': response.text.strip(),
            'reasoning': reasoning.text.strip()
        })
    except Exception as e:
        logging.error(f"Error during chat session: {e}")
        return jsonify({'error': 'Failed to process the image with Generative AI'}), 500

if __name__ == '__main__':
    app.run(debug=True)
