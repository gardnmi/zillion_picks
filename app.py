import re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import stripe
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_admin import Admin, AdminIndexView
from flask_admin.contrib.sqla import ModelView
from flask_login import UserMixin, LoginManager, current_user, login_user
from sqlalchemy_utils import create_database, database_exists
from wtforms import StringField, SubmitField, TextAreaField, ValidationError, PasswordField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from utils import create_sidebar, get_results, table_cleanup

# -------------- Config -------------- #
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'gardnmi@gmail.com'
app.config['MAIL_PASSWORD'] = 'mB9LrNn^0RxQPikqsj3O8b3z#MO%lv'
app.config['SECRET_KEY'] = 'x4@q@c&1_@q4-sy9swme5wk%2mt^4nhb-p4taiw3^^vmou4l+i'
stripe.api_key = 'sk_test_STdRsQH9I95FeiVPnI1QuWSg00HgPQXOTw' #'sk_live_p3mKPEvsDkJhj39eEyljPiHN00Hw30n89x'
endpoint_secret = 'whsec_Oz4UvznVbAn8fUWWXXiEclVDAFWR7qMh'  #'whsec_RM6cA7GXhD5P5QeXejuzuMjypvFwNwRB'
admin_password = '9643602Mjg$'

# -------------- Extensions -------------- #
db = SQLAlchemy(app)
mail = Mail(app)
login = LoginManager(app)

@login.user_loader
def load_user(user_id):
    return User.query.get(user_id)

# -------------- Models -------------- #
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)

class Purchases(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(120), unique=False, nullable=False) 
    session_id = db.Column(db.String(120), unique=False, nullable=False) 
    event_id = db.Column(db.String(120), unique=False, nullable=False)
    email = db.Column(db.String(120), unique=False, nullable=False)
    product = db.Column(db.String(120), unique=False, nullable=False)
    amount = db.Column(db.Integer, unique=False, nullable=False)
    coupon = db.Column(db.String(120), unique=False, nullable=True)
    created_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# -------------- Admin -------------- #
class MyAdminIndexView(AdminIndexView):
    def is_accessible(self):
        return current_user.is_authenticated

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('admin_login'))

class MyModelView(ModelView):
    def is_accessible(self):
        return current_user.is_authenticated

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('admin_login'))

admin = Admin(app, index_view=MyAdminIndexView())
admin.add_view(MyModelView(Purchases, db.session))
admin.add_view(MyModelView(User, db.session))

# -------------- Forms -------------- #
class AccessForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    submit = SubmitField('Submit')

class ContactForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    body = TextAreaField("Message", validators=[DataRequired(), Length(min=5)])
    submit = SubmitField('Send')

class AdminLoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


# -------------- Views -------------- #
@app.route("/", methods=['GET', 'POST'])
def home():
    flash(f"Our picks  will be FREE through Week 03. <br> <a id='flash' href= {url_for('picks')}>  Check them Out!</a>", 'success') 

    contact_form = ContactForm()      
    if contact_form.validate_on_submit():
        
        email = contact_form.email.data
        body = contact_form.body.data
        msg = Message(subject=f'[Insert Name] From:<{email}>', 
                      sender='gardnmi@gmail.com', 
                      recipients = ['gardnmi@gmail.com'],
                      body = body)
        mail.send(msg)
        flash('Your message has been sent!', 'success')      
        contact_form.body.data = None 
    return render_template("home.html", contact_form=contact_form)


@app.route("/admin_login", methods=['GET', 'POST'])
def admin_login():
    file_path = Path('picks')
    sidebar_links = create_sidebar(file_path)
    
    if current_user.is_authenticated:
    
        return redirect(url_for('purchases.index_view'))

    form = AdminLoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(name=form.username.data).first()

        if user and admin_password == form.password.data:
            login_user(user)
            return redirect(url_for('purchases.index_view'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')

    return render_template('admin_login.html', form=form, sidebar_links=sidebar_links)


@app.route("/contact/", methods=['GET', 'POST'])
def contact():
    
    file_path = Path('picks')
    sidebar_links = create_sidebar(file_path)

    form = ContactForm()  

    if form.validate_on_submit():
        
        email = form.email.data
        body = form.body.data
        msg = Message(subject=f'CUSTOMER SUPPORT From:<{email}>', 
                      sender='gardnmi@gmail.com', 
                      recipients = ['gardnmi@gmail.com'],
                      body = body)
        mail.send(msg)
        flash('Your message has been sent! We will get back to you as soon as possible.', 'success')      
        form.body.data = None 
    return render_template("contact.html", form=form, sidebar_links=sidebar_links)
    

@app.route("/success/")
def success():
    file_path = Path('picks')
    sidebar_links = create_sidebar(file_path)
    
    return render_template("success.html", sidebar_links=sidebar_links)


@app.route("/charts/")
def charts():
    file_path = Path('picks')
    sidebar_links = create_sidebar(file_path)
    
    return render_template("charts.html", sidebar_links=sidebar_links, active='charts')


@app.route("/about/")
def about():
    file_path = Path('picks')
    sidebar_links = create_sidebar(file_path)
    
    return render_template("about.html", sidebar_links=sidebar_links, active='about')


@app.route("/picks/", methods=['GET'])
@app.route("/picks/<is_premium>/", methods=['GET', 'POST'])
def picks(is_premium=None):
    
    file_path = Path('picks')
    sidebar_links = create_sidebar(file_path)

    if is_premium == 'premium':
        default_season, default_week = sorted((file_path/'premium').rglob('*.csv'),  key=lambda f: f.stem)[-1].stem.split('_')
    else:
        default_season, default_week = sorted((file_path/'free').rglob('*.csv'),  key=lambda f: f.stem)[-1].stem.split('_')

    season = request.args.get('season', default=default_season)
    week = request.args.get('week', default=default_week)

    if is_premium == 'premium':

        form = AccessForm()

        if form.validate_on_submit():
            # VALIDATE THE PURCHASE
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
                df = pd.read_csv(file_path/f'premium/{season}_{week}.csv')
                spread_results, straight_results, std_spread_results, std_straight_results = get_results(df, file_path, season, week)
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
                        'name':f"{season} {(lambda x: 'Week '+ x if x != 'postseason' else 'Post Season') (week)} Premium Picks",
                        'description': 'Access to this weeks premiums picks',
                        'amount': 5000,
                        'currency': 'usd',
                        'quantity': 1,
                    }],
                    )
                               
                return render_template("checkout.html", sidebar_links=sidebar_links, stripe_session=stripe_session.id, active=season)
        return render_template("login.html", sidebar_links=sidebar_links, form=form, active=season)    

    else:
        
        df = pd.read_csv(file_path/f'free/{season}_{week}.csv')
        spread_results, straight_results, std_spread_results, std_straight_results = get_results(df, file_path, season, week)

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


@app.route("/payment_confirmation/", methods=['POST'])
def payment_confirmation():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    event = None

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)

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
                        customer_id = event['data']['object']['customer'], 
                        session_id = event['data']['object']['id'],
                        event_id = event['id'],
                        email = email,
                        product = event['data']['object']['display_items'][0]['custom']['name'],
                        amount = event['data']['object']['display_items'][0]['amount']          
                    )
        
        db.session.add(purchase)
        db.session.commit()

    return 'Success', 200


if __name__ == '__main__':
    app.run()
