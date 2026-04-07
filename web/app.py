import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parents[1]
if (BASE_DIR / "src").exists():
    PROJECT_DIR = BASE_DIR
elif (BASE_DIR / "crop-yield-prediction" / "src").exists():
    PROJECT_DIR = BASE_DIR / "crop-yield-prediction"
else:
    raise FileNotFoundError("Could not locate 'src' directory.")

DB_PATH = BASE_DIR / "web" / "app.db"
MODEL_PATH = PROJECT_DIR / "models" / "crop_yield_rf_pipeline.joblib"

sys.path.insert(0, str(PROJECT_DIR / "src"))
from predict import predict_crop_yield  # noqa: E402
from utils import load_model  # noqa: E402


REQUIRED_FIELDS = [
    "N",
    "P",
    "K",
    "pH",
    "organic_matter",
    "rainfall",
    "temp_min",
    "temp_max",
    "fertilizer_usage",
    "crop_type",
]

CROP_TYPES = ["rice", "maize", "wheat", "barley", "oats"]


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")
app.config["ADMIN_REGISTRATION_CODE"] = os.getenv("ADMIN_REGISTRATION_CODE", "admin-register-2026")


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'farmer')),
            approved INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_json TEXT NOT NULL,
            predicted_yield REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    # Database migration for older databases that do not have approval status.
    columns = db.execute("PRAGMA table_info(users)").fetchall()
    column_names = [col[1] for col in columns]
    if "approved" not in column_names:
        db.execute("ALTER TABLE users ADD COLUMN approved INTEGER NOT NULL DEFAULT 1")

    db.commit()


def ensure_default_admin() -> None:
    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
    if existing is None:
        db.execute(
            "INSERT INTO users (username, password_hash, role, approved, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                "admin",
                generate_password_hash("admin123"),
                "admin",
                1,
                datetime.now(UTC).isoformat(),
            ),
        )
        db.commit()


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


def role_required(role_name):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if session.get("role") != role_name:
                flash("You do not have permission to access that page.", "error")
                if session.get("role") == "admin":
                    return redirect(url_for("admin_dashboard"))
                if session.get("role") == "farmer":
                    return redirect(url_for("farmer_dashboard"))
                session.clear()
                return redirect(url_for("login"))
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


def farmer_approved_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if session.get("role") != "farmer":
            return redirect(url_for("dashboard"))

        db = get_db()
        user = db.execute("SELECT approved FROM users WHERE id = ?", (session.get("user_id"),)).fetchone()
        if not user or int(user["approved"]) != 1:
            flash("Your farmer account is pending admin approval.", "error")
            session.clear()
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


def parse_prediction_input(form_data) -> dict:
    values = {}
    numeric_fields = [
        "N",
        "P",
        "K",
        "pH",
        "organic_matter",
        "rainfall",
        "temp_min",
        "temp_max",
        "fertilizer_usage",
    ]
    for field in numeric_fields:
        values[field] = float(form_data.get(field, "").strip())

    crop_type = form_data.get("crop_type", "").strip().lower()
    if crop_type not in CROP_TYPES:
        raise ValueError("Invalid crop type selected.")

    values["crop_type"] = crop_type
    return values


def get_model():
    if "model" not in g:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Run training first: python main.py --mode train-only"
            )
        g.model = load_model(str(MODEL_PATH))
    return g.model


@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute(
            "SELECT id, username, password_hash, role, approved FROM users WHERE username = ?", (username,)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            if user["role"] == "farmer" and int(user["approved"]) != 1:
                flash("Your account is pending admin approval. Please wait.", "error")
                return render_template("login.html")

            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        role = request.form.get("role", "farmer").strip().lower()
        admin_code = request.form.get("admin_code", "").strip()

        if not username or not password:
            flash("Username and password are required.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif role not in {"admin", "farmer"}:
            flash("Invalid role selected.", "error")
        elif role == "admin" and admin_code != app.config["ADMIN_REGISTRATION_CODE"]:
            flash("Invalid admin registration code.", "error")
        else:
            db = get_db()
            try:
                db.execute(
                    "INSERT INTO users (username, password_hash, role, approved, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        username,
                        generate_password_hash(password),
                        role,
                        1 if role == "admin" else 0,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                db.commit()
                if role == "farmer":
                    flash("Account created. Please wait for admin approval before logging in.", "success")
                else:
                    flash("Admin account created successfully. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Username already exists.", "error")

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    if session.get("role") == "farmer":
        return redirect(url_for("farmer_dashboard"))

    flash("Session role is invalid. Please log in again.", "error")
    session.clear()
    return redirect(url_for("login"))


@app.route("/farmer")
@login_required
@role_required("farmer")
@farmer_approved_required
def farmer_dashboard():
    return render_template("farmer_dashboard.html", crop_types=CROP_TYPES)


@app.route("/farmer/predict", methods=["POST"])
@login_required
@role_required("farmer")
@farmer_approved_required
def farmer_predict():
    try:
        input_data = parse_prediction_input(request.form)
        model = get_model()
        predicted_yield = float(predict_crop_yield(model, input_data, feature_names=[]))

        db = get_db()
        db.execute(
            "INSERT INTO predictions (user_id, input_json, predicted_yield, created_at) VALUES (?, ?, ?, ?)",
            (
                session["user_id"],
                json.dumps(input_data),
                predicted_yield,
                datetime.now(UTC).isoformat(),
            ),
        )
        db.commit()

        return render_template(
            "farmer_dashboard.html",
            crop_types=CROP_TYPES,
            predicted_yield=predicted_yield,
            input_data=input_data,
        )
    except FileNotFoundError as model_error:
        flash(str(model_error), "error")
    except ValueError as value_error:
        flash(f"Input error: {value_error}", "error")
    except Exception as error:
        flash(f"Prediction failed: {error}", "error")

    return render_template("farmer_dashboard.html", crop_types=CROP_TYPES)


@app.route("/farmer/history")
@login_required
@role_required("farmer")
@farmer_approved_required
def farmer_history():
    db = get_db()
    rows = db.execute(
        "SELECT id, input_json, predicted_yield, created_at FROM predictions WHERE user_id = ? ORDER BY id DESC",
        (session["user_id"],),
    ).fetchall()

    history = []
    for row in rows:
        history.append(
            {
                "id": row["id"],
                "input": json.loads(row["input_json"]),
                "predicted_yield": row["predicted_yield"],
                "created_at": row["created_at"],
            }
        )

    return render_template("farmer_history.html", history=history)


@app.route("/admin")
@login_required
@role_required("admin")
def admin_dashboard():
    db = get_db()

    user_count = db.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    farmer_count = db.execute("SELECT COUNT(*) AS c FROM users WHERE role = 'farmer'").fetchone()["c"]
    prediction_count = db.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]

    recent_predictions = db.execute(
        """
        SELECT p.id, u.username, p.predicted_yield, p.created_at
        FROM predictions p
        JOIN users u ON u.id = p.user_id
        ORDER BY p.id DESC
        LIMIT 10
        """
    ).fetchall()

    return render_template(
        "admin_dashboard.html",
        user_count=user_count,
        farmer_count=farmer_count,
        prediction_count=prediction_count,
        recent_predictions=recent_predictions,
    )


@app.route("/admin/users", methods=["GET", "POST"])
@login_required
@role_required("admin")
def admin_users():
    db = get_db()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "farmer")

        if not username or not password:
            flash("Username and password are required.", "error")
        elif role not in {"admin", "farmer"}:
            flash("Invalid role selected.", "error")
        else:
            try:
                db.execute(
                    "INSERT INTO users (username, password_hash, role, approved, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        username,
                        generate_password_hash(password),
                        role,
                        1,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                db.commit()
                flash("User created successfully.", "success")
            except sqlite3.IntegrityError:
                flash("Username already exists.", "error")

    users = db.execute(
        "SELECT id, username, role, approved, created_at FROM users ORDER BY id ASC"
    ).fetchall()
    return render_template("admin_users.html", users=users)


@app.route("/admin/users/approve/<int:user_id>", methods=["POST"])
@login_required
@role_required("admin")
def admin_approve_user(user_id):
    db = get_db()
    user = db.execute("SELECT id, role, approved FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("admin_users"))

    if user["role"] != "farmer":
        flash("Only farmer accounts require approval.", "error")
        return redirect(url_for("admin_users"))

    if int(user["approved"]) == 1:
        flash("Farmer account is already approved.", "success")
        return redirect(url_for("admin_users"))

    db.execute("UPDATE users SET approved = 1 WHERE id = ?", (user_id,))
    db.commit()
    flash("Farmer account approved.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/delete/<int:user_id>", methods=["POST"])
@login_required
@role_required("admin")
def admin_delete_user(user_id):
    if user_id == session.get("user_id"):
        flash("You cannot delete your own account while logged in.", "error")
        return redirect(url_for("admin_users"))

    db = get_db()
    db.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    flash("User deleted.", "success")
    return redirect(url_for("admin_users"))


@app.context_processor
def inject_global_values():
    return {
        "current_user": {
            "id": session.get("user_id"),
            "username": session.get("username"),
            "role": session.get("role"),
        }
    }


with app.app_context():
    init_db()
    ensure_default_admin()


if __name__ == "__main__":
    app.run(debug=True)
