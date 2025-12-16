# create_db.py
import os
import sys

# Ensure the script directory is on sys.path so imports like `from extensions import db` work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print("Working directory:", os.getcwd())
print("Script directory:", SCRIPT_DIR)
print("sys.path[0]:", sys.path[0])
print("Files in script dir:", os.listdir(SCRIPT_DIR))

try:
    from extensions import db
    import models  # make sure models are imported so SQLAlchemy sees them
except Exception as e:
    print("Import error:", e)
    print("Make sure `extensions.py` and `models.py` are in the same folder as create_db.py")
    raise

from flask import Flask

app = Flask(__name__)
app.config["SECRET_KEY"] = "devkey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///brain_chat.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()
    print("Database created (or already exists): brain_chat.db")
