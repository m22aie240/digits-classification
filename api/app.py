from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

# Loading multiple models
models = {
    'svm': load("models/svm_gamma:0.01_C:1.joblib"),
    'lr': load("models/lr_solver:newton-cg.joblib"),  
    'tree': load("models/tree_max_depth:50.joblib")  
}

@app.route("/")
def hello_world():
    return "<b>Hello, World!</b>"

#  

@app.route("/predict/<model_type>", methods=['POST'])
def predict(model_type):
    if model_type not in models:
        return jsonify({"error": "Model type not supported"}), 400

    image = request.json.get('image')
    if image is None:
        return jsonify({"error": "Missing image data"}), 400

    model = models[model_type]
    predicted = model.predict([image])
    return jsonify({"y_predicted": int(predicted[0])})

# ... [rest of your Flask app]

if __name__ == '__main__':
    app.run(debug=True)
