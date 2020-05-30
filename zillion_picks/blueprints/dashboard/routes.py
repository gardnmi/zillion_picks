from flask import Blueprint, render_template, request, current_app, flash, redirect, url_for
from flask_login import login_user, current_user
from flask_mail import Message
import stripe
import pandas as pd
from pathlib import Path
from zillion_picks import mail, db
from zillion_picks.models import Purchases, User
from zillion_picks.blueprints.dashboard.utils import create_sidebar, get_results, table_cleanup
from zillion_picks.blueprints.dashboard.forms import AccessForm, ContactForm, AdminLoginForm

dashboard_bp = Blueprint('dashboard_bp', __name__)

stripe.api_key = current_app.config['STRIPE_API_KEY']
endpoint_secret = current_app.config['STRIPE_ENDPOINT_SECRET']

picks_file_path = Path(__file__).absolute().parent.parent.parent/'picks'


@dashboard_bp.route("/picks/", methods=['GET'])
@dashboard_bp.route("/picks/<is_premium>/", methods=['GET', 'POST'])
def picks(is_premium=None, flash_message=None):

    sidebar_links = create_sidebar(picks_file_path)

    if is_premium == 'premium':
        default_season, default_week = sorted(
            (picks_file_path/'premium').rglob('*.csv'),  key=lambda f: f.stem)[-1].stem.split('_')
    else:
        default_season, default_week = sorted(
            (picks_file_path/'free').rglob('*.csv'),  key=lambda f: f.stem)[-1].stem.split('_')

    season = request.args.get('season', default=default_season)
    week = request.args.get('week', default=default_week)
    flash_message = request.args.get('flash_message', default=None)

    if is_premium == 'premium':
        form = AccessForm()
        if form.validate_on_submit():
            # Validates Purchase
            user = Purchases.query.filter_by(email=form.email.data).first()
            if user:
                purchase = Purchases.query.filter_by(email=user.email,
                                                     product=f"{season} {(lambda x: 'Week '+ x if x != 'postseason' else 'Post Season') (week)} Premium Picks").first()
                customer = user.customer_id
                customer_email = None
            else:
                purchase = None
                customer = None
                customer_email = form.email.data

            if purchase:
                df = pd.read_csv(
                    picks_file_path/f'premium/{season}_{week}.csv')
                spread_results, straight_results, std_spread_results, std_straight_results = get_results(
                    df, picks_file_path, season, week)

                return render_template(
                    "picks.html",
                    df=table_cleanup(df, week),
                    sidebar_links=sidebar_links,
                    season=season,
                    week=week,
                    spread_results=spread_results,
                    straight_results=straight_results,
                    std_spread_results=std_spread_results,
                    std_straight_results=std_straight_results,
                    active=season)
            else:
                stripe_session = stripe.checkout.Session.create(
                    customer=customer,
                    customer_email=customer_email,
                    payment_method_types=["card"],
                    success_url='https://www.zillionpicks.com/success?session_id={CHECKOUT_SESSION_ID}',
                    cancel_url='https://www.zillionpicks.com/picks/premium/',
                    line_items=[{
                        'name': f"{season} {(lambda x: 'Week '+ x if x != 'postseason' else 'Post Season') (week)} Premium Picks",
                        'description': 'Access to this weeks premiums picks',
                        'amount': 5000,
                        'currency': 'usd',
                        'quantity': 1,
                    }],)

                stripe_public_key = current_app.config['STRIPE_PUB_KEY']

                return render_template("checkout.html", sidebar_links=sidebar_links, stripe_session=stripe_session.id, active=season, stripe_public_key=stripe_public_key)

        if flash_message:
            flash(flash_message, 'success')

        return render_template("login.html", sidebar_links=sidebar_links, form=form, active=season)

    else:

        df = pd.read_csv(picks_file_path/f'free/{season}_{week}.csv')
        spread_results, straight_results, std_spread_results, std_straight_results = get_results(
            df, picks_file_path, season, week)

        return render_template("picks.html",
                               df=table_cleanup(df, week),
                               sidebar_links=sidebar_links,
                               season=season,
                               week=week,
                               spread_results=spread_results,
                               straight_results=straight_results,
                               std_spread_results=std_spread_results,
                               std_straight_results=std_straight_results,
                               active=season
                               )


@dashboard_bp.route("/payment_confirmation/", methods=['POST'])
def payment_confirmation():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    event = None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret)

    except ValueError:
        # Invalid payload
        return 'Invalid payload', 400

    except stripe.error.SignatureVerificationError:
        # Invalid signature
        return 'Invalid signature', 400

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':

        if event['data']['object']['customer_email'] is None:
            customer_id = (event['data']['object']['customer'])
            email = stripe.Customer.retrieve(customer_id)['email']
        else:
            email = event['data']['object']['customer_email']

        # REPLACE THESE WITH VARIABLES LIKE EMAIL
        purchase = Purchases(
            customer_id=event['data']['object']['customer'],
            session_id=event['data']['object']['id'],
            event_id=event['id'],
            email=email,
            product=event['data']['object']['display_items'][0]['custom']['name'],
            amount=event['data']['object']['display_items'][0]['amount']
        )

        db.session.add(purchase)
        db.session.commit()

    return 'Success', 200


@dashboard_bp.route("/success/")
def success():
    sidebar_links = create_sidebar(picks_file_path)

    return render_template("success.html", sidebar_links=sidebar_links)


@dashboard_bp.route("/contact/", methods=['GET', 'POST'])
def contact():

    sidebar_links = create_sidebar(picks_file_path)

    form = ContactForm()

    if form.validate_on_submit():

        email = form.email.data
        body = form.body.data
        msg = Message(subject=f'ZILLION PICKS: {email}',
                      sender='gardnmi@gmail.com',
                      recipients=['gardnmi@gmail.com'],
                      body=body)
        mail.send(msg)
        flash('Your message has been sent! We will get back to you as soon as possible.', 'success')
        form.body.data = None
    return render_template("contact.html", form=form, sidebar_links=sidebar_links, active='contact_us')


@dashboard_bp.route("/charts/")
def charts():
    sidebar_links = create_sidebar(picks_file_path)

    return render_template("charts.html", sidebar_links=sidebar_links, active='charts')


@ dashboard_bp.route("/about/")
def about():
    sidebar_links = create_sidebar(picks_file_path)

    return render_template("about.html", sidebar_links=sidebar_links, active='about')


@dashboard_bp.route("/admin_login", methods=['GET', 'POST'])
def admin_login():

    sidebar_links = create_sidebar(picks_file_path)

    if current_user.is_authenticated:

        return redirect(url_for('purchases.index_view'))

    form = AdminLoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(name=form.username.data).first()

        if user and current_app.config['ADMIN_PASSWORD'] == form.password.data:
            login_user(user)
            return redirect(url_for('purchases.index_view'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')

    return render_template('admin_login.html', form=form, sidebar_links=sidebar_links)
