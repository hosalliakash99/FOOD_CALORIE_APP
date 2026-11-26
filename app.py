import os
import io
import csv
import uuid
import sqlite3
from datetime import date, timedelta
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, flash, Response
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import pipeline

from nutrition_db import get_nutrition, list_all_foods

app = Flask(__name__)
app.secret_key = "nutrivision_super_secret_key_change_me"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "database.db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

print("Loading Food Recognition ML model (nateraw/food)...")
food_classifier = pipeline("image-classification", model="nateraw/food")
print("ML Model loaded successfully!")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                height_cm REAL DEFAULT 170,
                weight_kg REAL DEFAULT 70,
                age INTEGER DEFAULT 22,
                gender TEXT DEFAULT 'male',
                activity_level REAL DEFAULT 1.55,
                fitness_goal TEXT DEFAULT 'maintain',
                daily_goal INTEGER DEFAULT 2200,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                meal_type TEXT DEFAULT 'Lunch',
                food_name TEXT NOT NULL,
                confidence REAL DEFAULT 100.0,
                portion_grams REAL NOT NULL,
                calories REAL NOT NULL,
                protein REAL NOT NULL,
                carbs REAL NOT NULL,
                fat REAL NOT NULL,
                image_path TEXT,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS water_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                glasses INTEGER DEFAULT 0,
                log_date TEXT NOT NULL,
                UNIQUE(user_id, log_date),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.commit()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_health_metrics(height_cm, weight_kg, age, gender, activity_level, fitness_goal):
    height_m = max(height_cm / 100.0, 0.5)
    bmi = round(weight_kg / (height_m ** 2), 2)

    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_color = "sky"
    elif bmi < 25.0:
        bmi_category = "Normal Weight"
        bmi_color = "emerald"
    elif bmi < 30.0:
        bmi_category = "Overweight"
        bmi_color = "amber"
    else:
        bmi_category = "Obese"
        bmi_color = "rose"

    healthy_min_kg = round(18.5 * (height_m ** 2), 1)
    healthy_max_kg = round(24.9 * (height_m ** 2), 1)

    # Mifflin-St Jeor BMR Formula
    if gender == "female":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5

    tdee = bmr * float(activity_level)

    if fitness_goal == "lose":
        recommended_calories = max(1200, round(tdee - 500))
    elif fitness_goal == "gain":
        recommended_calories = round(tdee + 400)
    else:
        recommended_calories = round(tdee)

    return {
        "bmi": bmi,
        "bmi_category": bmi_category,
        "bmi_color": bmi_color,
        "healthy_weight_range": f"{healthy_min_kg} kg - {healthy_max_kg} kg",
        "bmr": round(bmr),
        "tdee": round(tdee),
        "recommended_calories": recommended_calories
    }


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ---------------- AUTH ROUTES ---------------- #

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        height_cm = float(request.form.get("height_cm", 170))
        weight_kg = float(request.form.get("weight_kg", 70))
        age = int(request.form.get("age", 22))
        gender = request.form.get("gender", "male")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        metrics = calculate_health_metrics(height_cm, weight_kg, age, gender, 1.55, "maintain")

        try:
            with get_db() as conn:
                conn.execute("""
                    INSERT INTO users (username, email, password_hash, height_cm, weight_kg, age, gender, daily_goal)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    username, email, generate_password_hash(password),
                    height_cm, weight_kg, age, gender, metrics["recommended_calories"]
                ))
                conn.commit()
            flash("Account created! Your daily calorie goal was auto-calculated from your BMI/BMR.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ? OR email = ?",
                (username, username.lower())
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- FRONTEND ROUTE ---------------- #

@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session.get("username"))


# ---------------- ML & API ROUTES ---------------- #

@app.route("/api/predict", methods=["POST"])
@login_required
def predict_food():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload PNG, JPG, JPEG, or WEBP."}), 400

    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    try:
        img = Image.open(save_path).convert("RGB")
        predictions = food_classifier(img, top_k=3)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    portion_grams = request.form.get("portion_grams")
    portion_val = float(portion_grams) if portion_grams else None

    # Include nutrition for all Top-3 predictions so user can switch between them seamlessly
    formatted_preds = []
    for p in predictions:
        nutri = get_nutrition(p["label"], portion_val)
        formatted_preds.append({
            "label": p["label"],
            "display_name": nutri["display_name"],
            "confidence": round(p["score"] * 100, 2),
            "nutrition": nutri
        })

    return jsonify({
        "predictions": formatted_preds,
        "image_url": f"/static/uploads/{unique_name}"
    })


@app.route("/api/foods", methods=["GET"])
@login_required
def get_foods():
    return jsonify({"foods": list_all_foods()})


@app.route("/api/meals", methods=["GET", "POST"])
@login_required
def manage_meals():
    user_id = session["user_id"]

    if request.method == "POST":
        data = request.get_json()
        with get_db() as conn:
            conn.execute("""
                INSERT INTO meals (user_id, meal_type, food_name, confidence, portion_grams, calories, protein, carbs, fat, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                data.get("meal_type", "Lunch"),
                data["food_name"],
                data.get("confidence", 100.0),
                data["portion_grams"],
                data["calories"],
                data["protein"],
                data["carbs"],
                data["fat"],
                data.get("image_path", "")
            ))
            conn.commit()
        return jsonify({"message": "Meal logged successfully!"}), 201

    with get_db() as conn:
        user_row = conn.execute("SELECT daily_goal FROM users WHERE id = ?", (user_id,)).fetchone()
        daily_goal = user_row["daily_goal"] if user_row else 2000

        meals = conn.execute("""
            SELECT id, meal_type, food_name, confidence, portion_grams, calories, protein, carbs, fat, image_path,
                   strftime('%Y-%m-%d %H:%M', logged_at, 'localtime') as logged_at
            FROM meals
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 30
        """, (user_id,)).fetchall()

        today_totals = conn.execute("""
            SELECT
                COALESCE(SUM(calories), 0) as calories,
                COALESCE(SUM(protein), 0) as protein,
                COALESCE(SUM(carbs), 0) as carbs,
                COALESCE(SUM(fat), 0) as fat
            FROM meals
            WHERE user_id = ? AND DATE(logged_at, 'localtime') = DATE('now', 'localtime')
        """, (user_id,)).fetchone()

        meal_type_rows = conn.execute("""
            SELECT meal_type, COALESCE(SUM(calories), 0) as total_cal
            FROM meals
            WHERE user_id = ? AND DATE(logged_at, 'localtime') = DATE('now', 'localtime')
            GROUP BY meal_type
        """, (user_id,)).fetchall()

        trend_rows = conn.execute("""
            SELECT DATE(logged_at, 'localtime') as day, COALESCE(SUM(calories), 0) as total_cal
            FROM meals
            WHERE user_id = ? AND DATE(logged_at, 'localtime') >= DATE('now', '-6 days', 'localtime')
            GROUP BY DATE(logged_at, 'localtime')
        """, (user_id,)).fetchall()

    trend_map = {r["day"]: round(r["total_cal"], 1) for r in trend_rows}
    today_dt = date.today()
    weekly_trend = []
    for i in range(6, -1, -1):
        d = today_dt - timedelta(days=i)
        d_str = d.isoformat()
        weekly_trend.append({
            "date": d_str,
            "label": d.strftime("%b %d"),
            "calories": trend_map.get(d_str, 0)
        })

    meal_type_map = {"Breakfast": 0, "Lunch": 0, "Dinner": 0, "Snack": 0}
    for r in meal_type_rows:
        if r["meal_type"] in meal_type_map:
            meal_type_map[r["meal_type"]] = round(r["total_cal"], 1)

    return jsonify({
        "daily_goal": daily_goal,
        "meals": [dict(m) for m in meals],
        "today_totals": {
            "calories": round(today_totals["calories"], 1),
            "protein": round(today_totals["protein"], 1),
            "carbs": round(today_totals["carbs"], 1),
            "fat": round(today_totals["fat"], 1),
        },
        "weekly_trend": weekly_trend,
        "meal_type_breakdown": meal_type_map
    })


@app.route("/api/meals/<int:meal_id>", methods=["DELETE"])
@login_required
def delete_meal(meal_id):
    with get_db() as conn:
        conn.execute("DELETE FROM meals WHERE id = ? AND user_id = ?", (meal_id, session["user_id"]))
        conn.commit()
    return jsonify({"message": "Meal deleted"})


@app.route("/api/profile", methods=["GET", "POST"])
@login_required
def profile_and_bmi():
    user_id = session["user_id"]

    if request.method == "POST":
        data = request.get_json()
        height_cm = float(data["height_cm"])
        weight_kg = float(data["weight_kg"])
        age = int(data["age"])
        gender = data["gender"]
        activity_level = float(data["activity_level"])
        fitness_goal = data["fitness_goal"]

        metrics = calculate_health_metrics(height_cm, weight_kg, age, gender, activity_level, fitness_goal)
        custom_goal = data.get("daily_goal")
        new_goal = int(custom_goal) if custom_goal else metrics["recommended_calories"]

        with get_db() as conn:
            conn.execute("""
                UPDATE users
                SET height_cm = ?, weight_kg = ?, age = ?, gender = ?,
                    activity_level = ?, fitness_goal = ?, daily_goal = ?
                WHERE id = ?
            """, (height_cm, weight_kg, age, gender, activity_level, fitness_goal, new_goal, user_id))
            conn.commit()

        return jsonify({"message": "Health profile & goal updated!", "metrics": metrics, "daily_goal": new_goal})

    with get_db() as conn:
        u = conn.execute("""
            SELECT username, email, height_cm, weight_kg, age, gender, activity_level, fitness_goal, daily_goal
            FROM users WHERE id = ?
        """, (user_id,)).fetchone()

    metrics = calculate_health_metrics(
        u["height_cm"], u["weight_kg"], u["age"], u["gender"], u["activity_level"], u["fitness_goal"]
    )
    return jsonify({"profile": dict(u), "metrics": metrics})


@app.route("/api/water", methods=["GET", "POST"])
@login_required
def manage_water():
    user_id = session["user_id"]
    today_str = date.today().isoformat()

    with get_db() as conn:
        if request.method == "POST":
            delta = int(request.get_json().get("delta", 1))
            row = conn.execute(
                "SELECT glasses FROM water_logs WHERE user_id = ? AND log_date = ?",
                (user_id, today_str)
            ).fetchone()
            current = row["glasses"] if row else 0
            new_val = max(0, min(20, current + delta))
            conn.execute("""
                INSERT INTO water_logs (user_id, glasses, log_date)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, log_date) DO UPDATE SET glasses = excluded.glasses
            """, (user_id, new_val, today_str))
            conn.commit()
            return jsonify({"glasses": new_val})

        row = conn.execute(
            "SELECT glasses FROM water_logs WHERE user_id = ? AND log_date = ?",
            (user_id, today_str)
        ).fetchone()
        return jsonify({"glasses": row["glasses"] if row else 0})


@app.route("/api/export")
@login_required
def export_csv():
    user_id = session["user_id"]
    with get_db() as conn:
        meals = conn.execute("""
            SELECT logged_at, meal_type, food_name, portion_grams, calories, protein, carbs, fat, confidence
            FROM meals WHERE user_id = ? ORDER BY id DESC
        """, (user_id,)).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date/Time", "Meal Type", "Food Name", "Portion (g)", "Calories (kcal)", "Protein (g)", "Carbs (g)", "Fat (g)", "Confidence (%)"])
    for m in meals:
        writer.writerow([m["logged_at"], m["meal_type"], m["food_name"], m["portion_grams"], m["calories"], m["protein"], m["carbs"], m["fat"], m["confidence"]])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=nutrivision_meal_history.csv"}
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)