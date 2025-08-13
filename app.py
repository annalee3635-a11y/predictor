from flask import Flask, request, jsonify
import predictor

app = Flask(__name__)
app.config["DEBUG"] = False


@app.route('/', methods=['GET'])
def home():
    return """<h1>Stock Predictor</h1>
              <p>This site predicts stock prices using an LSTM deep-learning model.
              </p>"""


@app.route("/predictor", methods=["GET"])
def predict():
    
    # check if ticker provided (eg /predictor?ticker=AAPL). if so, assign it to variable
    # if not, display an error

    if 'ticker' in request.args:
        ticker = request.args['ticker']
    else:
        return "Error: No ticker provided. Please provide stock ticker (eg. AAPL)."
    
    print("Ticker provided, loading prediction...")

    predictor.predict(ticker)

    return "Your data will open in a new window"

if __name__ == "__main__":
    #app.run(host='0.0.0.0')
    app.run()