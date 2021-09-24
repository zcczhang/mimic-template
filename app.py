from flask import Flask, request, render_template, jsonify
import nltk
from summary import Summarize

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/summary', methods=["GET"])
def summary():
    return render_template('text_Summarization.html')


@app.route('/installation', methods=["GET"])
def installation():
    return render_template('installation.html')


@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')


@app.route("/summarize", methods=["GET", "POST"])
def summarize():
    text = request.form['text']
    sent = nltk.sent_tokenize(text)
    if len(sent) < 2:
        summary1 = "please pass more than 3 sentences to summarize the text"
    else:
        summary1 = Summarize(text=text, lang="english").summarize()
    result = {
        "result": summary1
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(debug=True)

