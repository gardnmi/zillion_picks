from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, PasswordField
from wtforms.validators import DataRequired, Email, Length


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
