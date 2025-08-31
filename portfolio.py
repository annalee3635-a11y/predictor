import base64
from io import BytesIO
from flask import Blueprint, redirect, render_template, request, session, url_for
from markupsafe import escape
from matplotlib.figure import Figure
from predicter import lstm
from predicter.db import get_db
from flask import g

bp = Blueprint("portfolio", __name__, url_prefix="/portfolio")

@bp.route("/", methods=["GET", "POST"])
def show(): 
    db = get_db()
    user_id = session.get("user_id")
    portfolio_table = db.execute("SELECT * FROM stocks WHERE author_id = ?", user_id)
    print(portfolio_table)
    data_for_passing = []
    for table in portfolio_table:
        cursor = table[4]
        image_data_row = cursor.fetchone()
        image_data = image_data_row[0]
        data = {
        '   data' : image_data.decode("ascii")
        }
        data_for_passing.append(data)
    return render_template("portfolio.html", **data_for_passing)