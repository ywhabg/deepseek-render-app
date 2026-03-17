from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.get("/")
def root():
    return render_template("index.html")
