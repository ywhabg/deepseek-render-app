from flask import Flask, request, render_template, jsonify

app = Flask(__name__, template_folder="template")

@app.route("/")
def home():
    return render_template("index.html")

