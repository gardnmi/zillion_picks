from flask_login import UserMixin, LoginManager, current_user, login_user
from datetime import datetime
from zillion_picks import db, login


@login.user_loader
def load_user(user_id):
    return User.query.get(user_id)


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
    created_date = db.Column(
        db.DateTime, nullable=False, default=datetime.utcnow)
