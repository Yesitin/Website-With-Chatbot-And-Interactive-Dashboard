from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from utils import get_response, predict_class
import subprocess
import threading
import os

# functions for script
def start_panel_server(script_path):
    # Command to start the Panel server
    command = ["panel", "serve", script_path, "--port", "5006"]

    subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)



# initializing flask app
app = Flask(__name__, template_folder="templates")

app.secret_key = "your_secret_key"  

VALID_USERNAME = "username"
VALID_PASSWORD = "password"


@app.route("/")
def index():
    return render_template("index.htm")


@app.route("/handle_message", methods=["POST"])
def handle_message():
    message = request.json["message"]
    intents_list = predict_class(message)
    response = get_response(intents_list)

    return jsonify({"response": response})


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("dashboard"))
        else:
            return "Invalid credentials, please try again."
    
    return render_template("login.htm")


@app.route("/dashboard")
def dashboard():
    if not session.get("authenticated"):
        return redirect(url_for("index"))
    
    return render_template("dashboard.htm")


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))  # setting the path
    script_path = os.path.join(script_dir, "dashboard.ipynb")  # File in the same folder    

    # start the Panel server in a separate thread
    panel_thread = threading.Thread(target=start_panel_server, args=(script_path,))
    panel_thread.start()

    # to run flask app
    app.run(host="127.0.0.1", port=5000, debug=True)

    