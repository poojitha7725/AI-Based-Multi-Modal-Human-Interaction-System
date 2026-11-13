'''Create a default test user for quick login during development'''
import os
import sys

# Ensure the project root is on PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from app import app, db
from database.database import User
from werkzeug.security import generate_password_hash

with app.app_context():
    # Create tables if they don't exist
    db.create_all()
    # Check if test user already exists
    if not User.query.filter_by(username='test').first():
        test_user = User(
            username='test',
            email='test@example.com',
            full_name='Test User'
        )
        test_user.set_password('test123')
        db.session.add(test_user)
        db.session.commit()
        print('Test user created: username="test", password="test123"')
    else:
        print('Test user already exists')
