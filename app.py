from flask import Flask


app = Flask(__name__)


@app.route("/hello", methods=["GET"])
def hello_world():
    return "<h1>Hello world</h1>"


# TODO: /app route

# TODO: /predict route

if __name__ == "__main__":
    app.run(port=5678, host="0.0.0.0", debug=True)
