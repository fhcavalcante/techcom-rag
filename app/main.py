from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Carregar manual
manual_file = 'manual.txt'
with open(manual_file, 'r') as file:
    manual_text = file.read()

# Especificar explicitamente o modelo de QA em português
model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

# Função para dividir o texto em partes menores
def split_text(text, max_length=500):
    sentences = text.split('. ')
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence.split()) > max_length:
            yield ' '.join(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence.split())
    if current_chunk:
        yield ' '.join(current_chunk)

# Função para gerar resposta com o modelo de QA
def generate_response(question, context):
    inputs = {
        'question': question,
        'context': context
    }
    response = qa_pipeline(inputs)
    return response['answer'], response['score']

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Dividir o manual em partes menores
    contexts = list(split_text(manual_text))

    best_answer = None
    best_score = 0
    for context in contexts:
        answer, score = generate_response(question, context)
        if score > best_score:
            best_answer = answer
            best_score = score

    return jsonify({"answer": best_answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
