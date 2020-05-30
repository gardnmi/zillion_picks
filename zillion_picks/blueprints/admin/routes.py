from flask import Blueprint, redirect, url_for
from flask_admin import AdminIndexView
from flask_login import current_user
from pathlib import Path
from flask_admin.contrib.sqla import ModelView
from zillion_picks import admin, db
from zillion_picks.models import User, Purchases

admin_bp = Blueprint('admin_bp', __name__)

# Resource:
# https://stackoverflow.com/questions/28508723/how-do-i-properly-set-up-flask-admin-views-with-using-an-application-factory


class MyAdminIndexView(AdminIndexView):
    def is_accessible(self):
        return current_user.is_authenticated

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('dashboard_bp.admin_login'))


class MyModelView(ModelView):
    def is_accessible(self):
        return current_user.is_authenticated

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('dashboard_bp.admin_login'))


admin.add_view(MyModelView(Purchases, db.session))
admin.add_view(MyModelView(User, db.session))
