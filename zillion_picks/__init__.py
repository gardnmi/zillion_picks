from flask import Flask, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user
from flask_mail import Mail
from flask_admin import Admin, AdminIndexView
from zillion_picks.config import DevelopmentConfig, ProductionConfig
from sqlalchemy_utils import database_exists


# Resources:
# https://github.com/hackersandslackers/flask-blueprint-tutorial/blob/master/application/__init__.py
# https://github.com/CoreyMSchafer/code_snippets/blob/master/Python/Flask_Blog/12-Error-Pages/flaskblog/__init__.py

# Extensions
db = SQLAlchemy()
mail = Mail()
login = LoginManager()
admin = Admin()


def create_app():
    """Create Flask application."""
    app = Flask(__name__)
    app.config.from_object(DevelopmentConfig)

    # Admin Index View
    class MyAdminIndexView(AdminIndexView):
        def is_accessible(self):
            return current_user.is_authenticated

        def inaccessible_callback(self, name, **kwargs):
            return redirect(url_for('dashboard_bp.admin_login'))

    # Initialize Extensions
    db.init_app(app)
    mail.init_app(app)
    login.init_app(app)
    admin.init_app(app, index_view=MyAdminIndexView())

    with app.app_context():
        # Initialize DB
        # https://flask-sqlalchemy.palletsprojects.com/en/2.x/contexts/

        if database_exists(app.config['SQLALCHEMY_DATABASE_URI']):
            pass
        else:
            from zillion_picks.models import User

            db.create_all()
            user = User(name='gardnmi')
            db.session.add(user)
            db.session.commit()

        # Import parts of our application
        from zillion_picks.blueprints.admin.routes import admin_bp
        from zillion_picks.blueprints.dashboard.routes import dashboard_bp
        from zillion_picks.blueprints.landing_page.routes import landing_page_bp

        # Register Blueprints
        app.register_blueprint(admin_bp)
        app.register_blueprint(dashboard_bp)
        app.register_blueprint(landing_page_bp)

        return app
