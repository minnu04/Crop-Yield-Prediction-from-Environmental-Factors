import json
import os
import sqlite3
import sys
import importlib
import csv
import io
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for, Response
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parents[1]
try:
    dotenv_module = importlib.import_module("dotenv")
    load_dotenv = getattr(dotenv_module, "load_dotenv", None)
    if callable(load_dotenv):
        load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

if (BASE_DIR / "src").exists():
    PROJECT_DIR = BASE_DIR
elif (BASE_DIR / "crop-yield-prediction" / "src").exists():
    PROJECT_DIR = BASE_DIR / "crop-yield-prediction"
else:
    raise FileNotFoundError("Could not locate 'src' directory.")

DB_PATH = BASE_DIR / "web" / "app.db"
MODEL_PATH = PROJECT_DIR / "models" / "crop_yield_rf_pipeline.joblib"

sys.path.insert(0, str(PROJECT_DIR / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction_service import (  # noqa: E402
    CROP_TYPES,
    PredictionInputError,
    build_prediction_response,
    build_simulation_response,
    normalize_input,
)
from model_utils import load_model  # noqa: E402
from weather_service import fetch_weather_forecast  # noqa: E402


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
            result_json TEXT,
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

    prediction_columns = db.execute("PRAGMA table_info(predictions)").fetchall()
    prediction_column_names = [col[1] for col in prediction_columns]
    if "result_json" not in prediction_column_names:
        db.execute("ALTER TABLE predictions ADD COLUMN result_json TEXT")

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
            if user and int(user["approved"]) == -1:
                flash("Your farmer account was rejected by admin. Please contact support.", "error")
            else:
                flash("Your farmer account is pending admin approval.", "error")
            session.clear()
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


def parse_prediction_input(form_data) -> dict:
    return normalize_input(dict(form_data))


def get_model():
    if "model" not in g:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Run training first: python main.py --mode train-only"
            )
        g.model = load_model(str(MODEL_PATH))
    return g.model


def save_prediction_record(user_id: int, input_data: dict, result: dict) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO predictions (user_id, input_json, result_json, predicted_yield, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            user_id,
            json.dumps(input_data),
            json.dumps(result),
            result["predictedYield"],
            datetime.now(UTC).isoformat(),
        ),
    )
    db.commit()


def build_prediction_result(payload: dict) -> dict:
    model = get_model()
    return build_prediction_response(model, payload)


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
                if int(user["approved"]) == -1:
                    flash("Your account was rejected by admin. Please contact support.", "error")
                else:
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
    return render_template("farmer_dashboard.html", crop_types=CROP_TYPES, prediction_result=None)


def _handle_prediction_submission(payload: dict) -> dict:
    normalized = normalize_input(payload)
    result = build_prediction_result(normalized)
    save_prediction_record(session["user_id"], normalized, result)
    return result


def _fetch_prediction_rows_for_user(user_id: int):
    db = get_db()
    return db.execute(
        "SELECT id, input_json, predicted_yield, created_at FROM predictions WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    ).fetchall()


def _normalize_history_row(row) -> dict:
    try:
        payload = json.loads(row["input_json"])
    except Exception:
        payload = {}

    crop_type = str(payload.get("crop_type", "-")).title() if payload.get("crop_type") else "-"
    nutrient = f"N:{payload.get('N', '-')}, P:{payload.get('P', '-')}, K:{payload.get('K', '-')}"
    weather = (
        f"Rain: {payload.get('rainfall', '-')} mm | "
        f"Tmin: {payload.get('temp_min', '-')} C | "
        f"Tmax: {payload.get('temp_max', '-')} C"
    )

    return {
        "id": row["id"],
        "crop_type": crop_type,
        "nutrients": nutrient,
        "ph": payload.get("pH", "-"),
        "organic_matter": payload.get("organic_matter", "-"),
        "weather": weather,
        "fertilizer_usage": payload.get("fertilizer_usage", "-"),
        "predicted_yield": row["predicted_yield"],
        "created_at": row["created_at"],
    }


def _compute_admin_analytics() -> dict:
    db = get_db()

    user_count = db.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    farmer_count = db.execute("SELECT COUNT(*) AS c FROM users WHERE role = 'farmer'").fetchone()["c"]
    pending_approval = db.execute(
        "SELECT COUNT(*) AS c FROM users WHERE role = 'farmer' AND approved = 0"
    ).fetchone()["c"]

    prediction_rows = db.execute(
        "SELECT predicted_yield, created_at, input_json, result_json FROM predictions ORDER BY id DESC"
    ).fetchall()
    prediction_count = len(prediction_rows)

    recent_prediction_rows = db.execute(
        """
        SELECT p.id, u.username, p.predicted_yield, p.created_at
        FROM predictions p
        JOIN users u ON u.id = p.user_id
        ORDER BY p.id DESC
        LIMIT 10
        """
    ).fetchall()
    recent_predictions = [
        {
            "id": row["id"],
            "username": row["username"],
            "predicted_yield": row["predicted_yield"],
            "created_at": row["created_at"],
        }
        for row in recent_prediction_rows
    ]

    avg_yield = 0.0
    if prediction_count:
        avg_yield = round(sum(float(row["predicted_yield"] or 0.0) for row in prediction_rows) / prediction_count, 2)

    # Last 7-day prediction trend.
    today = datetime.now(UTC).date()
    trend_map = {(today - timedelta(days=i)).isoformat(): 0 for i in range(6, -1, -1)}

    crop_counts = {crop: 0 for crop in CROP_TYPES}
    risk_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}

    for row in prediction_rows:
        created_raw = row["created_at"] or ""
        try:
            created_dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            day_key = created_dt.date().isoformat()
            if day_key in trend_map:
                trend_map[day_key] += 1
        except Exception:
            pass

        try:
            input_payload = json.loads(row["input_json"] or "{}")
            crop = str(input_payload.get("crop_type", "")).lower()
            if crop in crop_counts:
                crop_counts[crop] += 1
        except Exception:
            pass

        try:
            result_payload = json.loads(row["result_json"] or "{}")
            risk_level = str((result_payload.get("risk") or {}).get("level", "unknown")).lower()
            if risk_level not in risk_counts:
                risk_level = "unknown"
            risk_counts[risk_level] += 1
        except Exception:
            risk_counts["unknown"] += 1

    trend = [{"date": day, "count": count} for day, count in trend_map.items()]
    crop_distribution = [{"crop": crop, "count": count} for crop, count in crop_counts.items()]
    risk_distribution = [{"level": level, "count": count} for level, count in risk_counts.items()]

    return {
        "summary": {
            "users": user_count,
            "farmers": farmer_count,
            "pendingApproval": pending_approval,
            "predictions": prediction_count,
            "avgYield": avg_yield,
        },
        "trend": trend,
        "cropDistribution": crop_distribution,
        "riskDistribution": risk_distribution,
        "recentPredictions": recent_predictions,
    }


@app.route("/predict-yield", methods=["POST"])
@app.route("/api/predict-yield", methods=["POST"])
@login_required
@role_required("farmer")
@farmer_approved_required
def predict_yield_api():
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    try:
        result = _handle_prediction_submission(payload)
        return jsonify(result)
    except PredictionInputError as prediction_error:
        return jsonify({"error": str(prediction_error)}), 400
    except FileNotFoundError as model_error:
        return jsonify({"error": str(model_error)}), 500
    except Exception as error:
        return jsonify({"error": f"Prediction failed: {error}"}), 500


@app.route("/simulate", methods=["POST"])
@login_required
@role_required("farmer")
@farmer_approved_required
def simulate_api():
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()

    try:
        if isinstance(payload, dict) and "input" in payload:
            base_input = payload.get("input") or {}
            adjustments = payload.get("adjustments") or {}
        else:
            base_input = payload
            adjustments = {}

        result = build_simulation_response(get_model(), base_input, adjustments)
        return jsonify(result)
    except PredictionInputError as prediction_error:
        return jsonify({"error": str(prediction_error)}), 400
    except FileNotFoundError as model_error:
        return jsonify({"error": str(model_error)}), 500
    except Exception as error:
        return jsonify({"error": f"Simulation failed: {error}"}), 500


@app.route("/weather-forecast", methods=["GET"])
@login_required
@role_required("farmer")
@farmer_approved_required
def weather_forecast_api():
    location = request.args.get("location", "").strip()
    latitude_raw = request.args.get("latitude")
    longitude_raw = request.args.get("longitude")

    latitude = None
    longitude = None
    if latitude_raw not in (None, "") and longitude_raw not in (None, ""):
        try:
            latitude = float(latitude_raw)
            longitude = float(longitude_raw)
        except ValueError:
            return jsonify({"error": "Latitude and longitude must be numeric values."}), 400

    result = fetch_weather_forecast(location=location, latitude=latitude, longitude=longitude)
    if result.get("status") == "fallback":
        return jsonify(result), 200

    return jsonify(result)


@app.route("/recommend-crops", methods=["POST"])
@login_required
@role_required("farmer")
@farmer_approved_required
def recommend_crops_api():
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    try:
        normalized = normalize_input(payload)
        result = build_prediction_result(normalized)
        return jsonify({
            "recommendation": result["recommendation"],
            "predictedYield": result["predictedYield"],
            "selectedCrop": normalized["crop_type"],
        })
    except PredictionInputError as prediction_error:
        return jsonify({"error": str(prediction_error)}), 400
    except FileNotFoundError as model_error:
        return jsonify({"error": str(model_error)}), 500
    except Exception as error:
        return jsonify({"error": f"Recommendation failed: {error}"}), 500


@app.route("/risk-assessment", methods=["POST"])
@login_required
@role_required("farmer")
@farmer_approved_required
def risk_assessment_api():
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    try:
        normalized = normalize_input(payload)
        result = build_prediction_result(normalized)
        return jsonify({
            "risk": result["risk"],
            "predictedYield": result["predictedYield"],
            "confidence": result["confidence"],
            "advisory": result["advisory"],
        })
    except PredictionInputError as prediction_error:
        return jsonify({"error": str(prediction_error)}), 400
    except FileNotFoundError as model_error:
        return jsonify({"error": str(model_error)}), 500
    except Exception as error:
        return jsonify({"error": f"Risk assessment failed: {error}"}), 500


@app.route("/farmer/predict", methods=["POST"])
@login_required
@role_required("farmer")
@farmer_approved_required
def farmer_predict():
    try:
        prediction_result = _handle_prediction_submission(request.form.to_dict())

        return render_template(
            "farmer_dashboard.html",
            crop_types=CROP_TYPES,
            prediction_result=prediction_result,
        )
    except PredictionInputError as value_error:
        flash(f"Input error: {value_error}", "error")
    except FileNotFoundError as model_error:
        flash(str(model_error), "error")
    except Exception as error:
        flash(f"Prediction failed: {error}", "error")

    return render_template("farmer_dashboard.html", crop_types=CROP_TYPES, prediction_result=None)


@app.route("/farmer/history")
@login_required
@role_required("farmer")
@farmer_approved_required
def farmer_history():
    rows = _fetch_prediction_rows_for_user(session["user_id"])
    history = [_normalize_history_row(row) for row in rows]

    return render_template("farmer_history.html", history=history)


@app.route("/api/farmer/history", methods=["GET"])
@login_required
@role_required("farmer")
@farmer_approved_required
def farmer_history_api():
    rows = _fetch_prediction_rows_for_user(session["user_id"])
    items = [_normalize_history_row(row) for row in rows]
    return jsonify({
        "status": "ok",
        "count": len(items),
        "items": items,
    })


@app.route("/farmer/history/export.csv", methods=["GET"])
@login_required
@role_required("farmer")
@farmer_approved_required
def farmer_history_export_csv():
    rows = _fetch_prediction_rows_for_user(session["user_id"])
    items = [_normalize_history_row(row) for row in rows]

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id",
            "crop_type",
            "nutrients",
            "ph",
            "organic_matter",
            "weather",
            "fertilizer_usage",
            "predicted_yield",
            "created_at",
        ],
    )
    writer.writeheader()
    for item in items:
        writer.writerow(item)

    csv_data = output.getvalue()
    output.close()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
    )


@app.route("/admin")
@login_required
@role_required("admin")
def admin_dashboard():
    analytics = _compute_admin_analytics()
    db = get_db()
    pending_farmers = db.execute(
        "SELECT id, username, created_at FROM users WHERE role = 'farmer' AND approved = 0 ORDER BY id ASC"
    ).fetchall()

    return render_template(
        "admin_dashboard.html",
        user_count=analytics["summary"]["users"],
        farmer_count=analytics["summary"]["farmers"],
        prediction_count=analytics["summary"]["predictions"],
        pending_approval=analytics["summary"]["pendingApproval"],
        avg_yield=analytics["summary"]["avgYield"],
        trend=analytics["trend"],
        crop_distribution=analytics["cropDistribution"],
        risk_distribution=analytics["riskDistribution"],
        recent_predictions=analytics["recentPredictions"],
        pending_farmers=pending_farmers,
    )


@app.route("/admin/analytics", methods=["GET"])
@login_required
@role_required("admin")
def admin_analytics_page():
    analytics = _compute_admin_analytics()
    return render_template(
        "admin_analytics.html",
        summary=analytics["summary"],
        trend=analytics["trend"],
        crop_distribution=analytics["cropDistribution"],
        risk_distribution=analytics["riskDistribution"],
        recent_predictions=analytics["recentPredictions"],
    )


def _get_system_health_payload() -> tuple[dict, int]:
    try:
        db = sqlite3.connect(DB_PATH)
        db.execute("SELECT 1")
        db.close()
        model_ready = MODEL_PATH.exists()
        return {
            "status": "ok",
            "service": "crop-yield-portal",
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": {
                "database": "ok",
                "model": "ok" if model_ready else "missing",
            },
        }, 200
    except Exception as error:
        return {
            "status": "degraded",
            "service": "crop-yield-portal",
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": {
                "database": "error",
                "model": "unknown",
            },
            "error": str(error),
        }, 500


@app.route("/admin/health", methods=["GET"])
@login_required
@role_required("admin")
def admin_health_page():
    health, _ = _get_system_health_payload()
    return render_template("admin_health.html", health=health)


@app.route("/api/admin/analytics", methods=["GET"])
@login_required
@role_required("admin")
def admin_analytics_api():
    try:
        analytics = _compute_admin_analytics()
        return jsonify(
            {
                "status": "ok",
                "generatedAt": datetime.now(UTC).isoformat(),
                **analytics,
            }
        )
    except Exception as error:
        return jsonify(
            {
                "status": "error",
                "generatedAt": datetime.now(UTC).isoformat(),
                "error": f"Could not build analytics: {error}",
                "summary": {
                    "users": 0,
                    "farmers": 0,
                    "pendingApproval": 0,
                    "predictions": 0,
                    "avgYield": 0.0,
                },
                "trend": [],
                "cropDistribution": [],
                "riskDistribution": [],
                "recentPredictions": [],
            }
        ), 500


@app.route("/api/system/health", methods=["GET"])
def system_health_api():
    payload, status_code = _get_system_health_payload()
    return jsonify(payload), status_code


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
    next_page = request.args.get("next", "admin_users")
    return redirect(url_for(next_page))


@app.route("/admin/users/reject/<int:user_id>", methods=["POST"])
@login_required
@role_required("admin")
def admin_reject_user(user_id):
    db = get_db()
    user = db.execute("SELECT id, role, approved FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("admin_users"))

    if user["role"] != "farmer":
        flash("Only farmer accounts can be rejected.", "error")
        return redirect(url_for("admin_users"))

    if int(user["approved"]) == -1:
        flash("Farmer account is already rejected.", "success")
        return redirect(url_for("admin_users"))

    db.execute("UPDATE users SET approved = -1 WHERE id = ?", (user_id,))
    db.commit()
    flash("Farmer account rejected.", "success")
    next_page = request.args.get("next", "admin_users")
    return redirect(url_for(next_page))


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
