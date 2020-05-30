from os import getenv
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

sql_path = Path(__file__).absolute().parent.parent/'site.db'


class DevelopmentConfig:
    SECRET_KEY = getenv('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{str(sql_path)}'
    MAIL_SERVER = getenv('MAIL_SERVER')
    MAIL_PORT = getenv('MAIL_PORT')
    MAIL_USE_TLS = getenv('MAIL_USE_TLS')
    MAIL_USERNAME = getenv('MAIL_USERNAME')
    MAIL_PASSWORD = getenv('MAIL_PASSWORD')
    STRIPE_API_KEY = getenv('STRIPE_DEV_API_KEY')
    STRIPE_PUB_KEY = getenv('STRIPE_DEV_PUB_KEY')
    STRIPE_ENDPOINT_SECRET = getenv('STRIPE_DEV_ENDPOINT_SECRET')
    ADMIN_PASSWORD = getenv('ADMIN_PASSWORD')


class ProductionConfig:
    SECRET_KEY = getenv('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{str(sql_path)}'
    MAIL_SERVER = getenv('MAIL_SERVER')
    MAIL_PORT = getenv('MAIL_PORT')
    MAIL_USE_TLS = getenv('MAIL_USE_TLS')
    MAIL_USERNAME = getenv('MAIL_USERNAME')
    MAIL_PASSWORD = getenv('MAIL_PASSWORD')
    STRIPE_API_KEY = getenv('STRIPE_PROD_API_KEY')
    STRIPE_PUB_KEY = getenv('STRIPE_PROD_PUB_KEY')
    STRIPE_ENDPOINT_SECRET = getenv('STRIPE_PROD_ENDPOINT_SECRET')
    ADMIN_PASSWORD = getenv('ADMIN_PASSWORD')
