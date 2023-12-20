from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

@app.route('/manipulate', methods=['POST'])
def manipulate_text():
    data = request.json
    original_text = data['text']
    manipulated_text = original_text.upper()
    return jsonify({'result': manipulated_text}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)