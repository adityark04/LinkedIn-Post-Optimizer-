from flask import Flask, request, jsonify
from flask_cors import CORS
# Import RAG-enhanced service for better quality
try:
    from ml_model_service_rag import generate_hook, generate_concise, generate_rephrased
    using_rag = True
    print("✅ Using RAG-enhanced ML service")
except ImportError:
    from ml_model_service import generate_hook, generate_concise, generate_rephrased
    using_rag = False
    print("⚠️ Falling back to basic ML service")

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) to allow your frontend
# to communicate with this backend.
# In a production environment, you would restrict this to your frontend's domain.
CORS(app)

@app.route('/api/optimize', methods=['POST'])
def optimize_post():
    """
    API endpoint to receive a draft post and return optimized suggestions.
    """
    # 1. Get the JSON data from the request body
    data = request.get_json()

    # 2. Validate the input
    if not data or 'draft' not in data or not data['draft'].strip():
        return jsonify({"error": "Request must contain a non-empty 'draft' field."}), 400

    draft_text = data['draft']

    try:
        # 3. Call the RAG-enhanced model service to get suggestions
        suggestions = [
            generate_hook(draft_text),
            generate_concise(draft_text),
            generate_rephrased(draft_text)
        ]

        # 4. Return the suggestions as a JSON response
        return jsonify(suggestions)

    except Exception as e:
        # 5. Handle any potential errors during model inference
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred while generating suggestions."}), 500

if __name__ == '__main__':
    # Run the app on port 5001 to avoid conflicts with other services.
    # use_reloader=False prevents import issues with OpenAI
    app.run(debug=True, port=5001, use_reloader=False)