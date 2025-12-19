from extensions import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # patient / doctor

    # ✅ NEW FIELDS (ONLY FOR DOCTORS)
    hospital = db.Column(db.String(150), nullable=True)
    experience_years = db.Column(db.Integer, nullable=True)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
# models.py (add this class to the existing file)
from datetime import datetime
from extensions import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# (keep your existing User and Message classes)

class LastRead(db.Model):
    """
    Records the last time `user_id` opened a chat with `peer_id`.
    We store per (user, peer) pair. When user opens /chat/<peer_id>,
    we'll set/update this to now().
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    peer_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    last_read = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # simple uniqueness constraint (user_id, peer_id)
    __table_args__ = (db.UniqueConstraint('user_id', 'peer_id', name='_user_peer_uc'),)
