import base64
from io import BytesIO
from flask import Blueprint, redirect, render_template, request, session, url_for
from markupsafe import escape
from matplotlib.figure import Figure
import lstm
from predicter.db import get_db
from flask import g

bp = Blueprint("predictor", __name__, url_prefix="/predictor")

@bp.route("/<tckr>", methods=["GET", "POST"])
def display(tckr): 
    #escaping for safety
    ticker = escape(tckr)
    #get the prediction
    user_id = session.get("user_id")
    if user_id is None:
        results = lstm.predict(ticker)
    #make the plots as subplots of a figure
    fig = Figure()
    past, future = fig.subplots(1, 2, sharey=True)
    past.plot(results[0], results[1])
    past.plot(results[0], results[2])
    future.plot(results[3], results[4])
    # Save it to a temporary buffer
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data_for_passing = {
        'data' : base64.b64encode(buf.getbuffer()).decode("ascii")
    }
    return render_template("predictionGraphs.html", **data_for_passing)

@bp.route("/", methods=["GET", "POST"])
def pred():
    if request.method == "POST":
       # getting input in HTML form
       ticker = escape(request.form.get("tckr"))

       return redirect(url_for('predictor.display', tckr = ticker))
    
    return render_template("getTicker.html")
