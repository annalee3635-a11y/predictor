import base64
from io import BytesIO
from flask import Flask, request, jsonify
from matplotlib.figure import Figure
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

    results = predictor.predict(ticker)

    fig = Figure()
    past, future = fig.subplots(1, 2, sharey=True)

    past.plot(results[0], results[1])
    past.plot(results[0], results[2])

    future.plot(results[3], results[4])
    
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

if __name__ == "__main__":
    #app.run(host='0.0.0.0')
    app.run()