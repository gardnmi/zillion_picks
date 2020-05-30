from flask import Blueprint, render_template, url_for, flash


landing_page_bp = Blueprint('landing_page_bp', __name__)


@landing_page_bp.route("/", methods=['GET', 'POST'])
def landing_page():
    flash(
        f"Our picks  will be FREE through Week 03. <br> <a id='flash' href= {url_for('dashboard_bp.picks')}>  Check them Out!</a>", 'success')

    return render_template("landing_page.html")
