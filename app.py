from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import subprocess

# Import emotion predictor
try:
    from predict_emotion import get_predictor
    EMOTION_PREDICTOR_AVAILABLE = True
    print("✅ Emotion predictor loaded successfully")
except Exception as e:
    EMOTION_PREDICTOR_AVAILABLE = False
    print(f"⚠️ Emotion predictor not available: {e}")

app = Flask(__name__)
app.secret_key = "teccr_secret_key"

# =========================
# FILE UPLOAD CONFIG
# =========================
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# DATABASE CONFIG (WAMP)
# =========================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "teccr_db",
    "charset": "utf8"
}

def get_db():
    return mysql.connector.connect(**DB_CONFIG)


# =========================
# BASIC PAGES
# =========================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/login")
def user_login():
    return render_template("user_login.html")


@app.route("/admin")
def admin_login():
    return render_template("admin_login.html")


# =========================
# AUTH: USER REGISTER
# =========================
@app.route("/auth/register", methods=["POST"])
def auth_register():
    username = request.form["username"]
    email = request.form["email"]
    password = generate_password_hash(request.form["password"])

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (%s,%s,%s,%s)",
            (username, email, password, datetime.now())
        )
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for("user_login"))
    except mysql.connector.Error as e:
        cursor.close()
        conn.close()
        return f"Registration error: {e}", 400


# =========================
# AUTH: USER LOGIN
# =========================
@app.route("/auth/user-login", methods=["POST"])
def auth_user_login():
    email = request.form["email"]
    password = request.form["password"]

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user and check_password_hash(user["password"], password):
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        return redirect(url_for("user_dashboard"))

    return redirect(url_for("user_login"))


# =========================
# AUTH: ADMIN LOGIN
# =========================
@app.route("/auth/admin-login", methods=["POST"])
def auth_admin_login():
    admin_id = request.form["admin_id"]
    password = request.form["password"]

    if admin_id == "admin" and password == "admin123":
        session["admin_id"] = 1  # dummy ID
        session["admin_name"] = "admin"
        return redirect(url_for("admin_dashboard"))

    return redirect(url_for("admin_login"))


# =========================
# USER DASHBOARD
# =========================
@app.route("/dashboard")
def user_dashboard():
    if "user_id" not in session:
        return redirect(url_for("user_login"))

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # Get all posts with admin info
    cursor.execute("""
        SELECT posts.id, posts.content, posts.image_path, posts.created_at, admins.admin_id
        FROM posts
        JOIN admins ON posts.admin_id = admins.id
        ORDER BY posts.created_at DESC
    """)
    posts = cursor.fetchall()

    # Get comments for each post
    for post in posts:
        cursor.execute("""
            SELECT comments.id, comments.comment_text, comments.created_at,
                   users.username
            FROM comments
            JOIN users ON comments.user_id = users.id
            WHERE comments.post_id = %s
            ORDER BY comments.created_at ASC
        """, (post['id'],))
        post['comments'] = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("user_dashboard.html", 
                         posts=posts, 
                         username=session.get('username'))


# =========================
# USER PROFILE
# =========================
@app.route("/profile")
def user_profile():
    if "user_id" not in session:
        return redirect(url_for("user_login"))

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # Get user info
    cursor.execute("SELECT * FROM users WHERE id = %s", (session["user_id"],))
    user = cursor.fetchone()

    # Get user's comments with post info and emotion results
    cursor.execute("""
        SELECT 
            comments.id,
            comments.comment_text,
            comments.created_at,
            posts.content as post_content,
            posts.id as post_id,
            emotion_results.primary_emotion,
            emotion_results.secondary_emotion,
            emotion_results.teccr_context
        FROM comments
        JOIN posts ON comments.post_id = posts.id
        LEFT JOIN emotion_results ON comments.id = emotion_results.comment_id
        WHERE comments.user_id = %s
        ORDER BY comments.created_at DESC
    """, (session["user_id"],))
    
    user_comments = cursor.fetchall()

    # Get stats
    cursor.execute("SELECT COUNT(*) as total FROM comments WHERE user_id = %s", 
                  (session["user_id"],))
    total_comments = cursor.fetchone()['total']

    cursor.close()
    conn.close()

    return render_template("user_profile.html", 
                         user=user,
                         comments=user_comments,
                         total_comments=total_comments)


# =========================
# ADMIN DASHBOARD
# =========================
@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin_id" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # Get statistics
    cursor.execute("SELECT COUNT(*) as total FROM posts WHERE admin_id = %s", 
                  (session["admin_id"],))
    total_posts = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as total FROM comments")
    total_comments = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as total FROM users")
    total_users = cursor.fetchone()['total']

    # Get model status
    cursor.execute("SELECT * FROM model_status ORDER BY last_trained DESC LIMIT 1")
    model_status = cursor.fetchone()

    # Get recent posts
    cursor.execute("""
        SELECT posts.id, posts.content, posts.image_path, posts.created_at,
               COUNT(comments.id) as comment_count
        FROM posts
        LEFT JOIN comments ON posts.id = comments.post_id
        WHERE posts.admin_id = %s
        GROUP BY posts.id
        ORDER BY posts.created_at DESC
        LIMIT 5
    """, (session["admin_id"],))
    recent_posts = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("admin_dashboard.html",
                         total_posts=total_posts,
                         total_comments=total_comments,
                         total_users=total_users,
                         model_status=model_status,
                         recent_posts=recent_posts,
                         admin_name=session.get('admin_name'))


# =========================
# ADMIN: MANAGE USERS
# =========================
@app.route("/admin/users")
def admin_manage_users():
    if "admin_id" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # Get all users with their stats
    cursor.execute("""
        SELECT 
            users.id,
            users.username,
            users.email,
            users.created_at,
            COUNT(DISTINCT comments.id) as comment_count
        FROM users
        LEFT JOIN comments ON users.id = comments.user_id
        GROUP BY users.id
        ORDER BY users.created_at DESC
    """)
    
    users = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("admin_users.html", users=users)


# =========================
# ADMIN: DELETE USER
# =========================
@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
def admin_delete_user(user_id):
    if "admin_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Delete user (comments will be deleted due to CASCADE)
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({"success": True, "message": "User deleted successfully"})
    except Exception as e:
        cursor.close()
        conn.close()
        return jsonify({"error": str(e)}), 500


# =========================
# ADMIN: DELETE POST
# =========================
@app.route("/admin/delete-post/<int:post_id>", methods=["POST"])
def admin_delete_post(post_id):
    if "admin_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Check if post belongs to this admin
        cursor.execute("SELECT admin_id, image_path FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        
        if not post or post[0] != session["admin_id"]:
            return jsonify({"success": False, "message": "Post not found or access denied"}), 403

        # Delete associated image file if exists
        if post[1]:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], post[1])
            if os.path.exists(image_path):
                os.remove(image_path)

        # Delete post (comments and emotion results will be deleted due to CASCADE)
        cursor.execute("DELETE FROM posts WHERE id = %s", (post_id,))
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({"success": True, "message": "Post deleted successfully"})
    except Exception as e:
        cursor.close()
        conn.close()
        return jsonify({"success": False, "message": str(e)}), 500


# =========================
# ADMIN: CREATE POST (WITH IMAGE)
# =========================
@app.route("/admin/create-post", methods=["POST"])
def create_post():
    if "admin_id" not in session:
        return redirect(url_for("admin_login"))

    content = request.form["content"]
    image_path = None

    # Handle image upload
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to filename to make it unique
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = f"uploads/{filename}"

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO posts (admin_id, content, image_path, created_at) VALUES (%s,%s,%s,%s)",
        (session["admin_id"], content, image_path, datetime.now())
    )

    conn.commit()
    cursor.close()
    conn.close()

    return redirect(url_for("admin_dashboard"))


# =========================
# USER: ADD COMMENT
# =========================
@app.route("/comment", methods=["POST"])
def add_comment():
    if "user_id" not in session:
        return redirect(url_for("user_login"))

    post_id = request.form["post_id"]
    comment_text = request.form["comment"]

    conn = get_db()
    cursor = conn.cursor()

    # Insert comment
    cursor.execute(
        "INSERT INTO comments (post_id, user_id, comment_text, created_at) VALUES (%s,%s,%s,%s)",
        (post_id, session["user_id"], comment_text, datetime.now())
    )

    comment_id = cursor.lastrowid
    conn.commit()

    # ===== EMOTION PREDICTION =====
    if EMOTION_PREDICTOR_AVAILABLE:
        try:
            predictor = get_predictor()
            result = predictor.predict(comment_text)

            if result['success']:
                cursor.execute("""
                    INSERT INTO emotion_results
                    (comment_id, primary_emotion, secondary_emotion, teccr_context, predicted_at)
                    VALUES (%s,%s,%s,%s,%s)
                """, (
                    comment_id,
                    result['primary_emotion'],
                    result.get('secondary_emotion', 'None'),
                    result['teccr_context'],
                    datetime.now()
                ))
                conn.commit()
                print(f"✅ Emotion predicted: {result['primary_emotion']}")
        except Exception as e:
            print(f"⚠️ Emotion prediction failed: {e}")

    cursor.close()
    conn.close()

    return redirect(url_for("user_dashboard"))


# =========================
# ADMIN: VIEW COMMENTS WITH EMOTIONS
# =========================
@app.route("/admin/comments")
def admin_view_comments():
    if "admin_id" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            comments.id,
            comments.comment_text,
            comments.created_at,
            users.username,
            posts.content as post_content,
            emotion_results.primary_emotion,
            emotion_results.secondary_emotion,
            emotion_results.teccr_context
        FROM comments
        JOIN users ON comments.user_id = users.id
        JOIN posts ON comments.post_id = posts.id
        LEFT JOIN emotion_results ON comments.id = emotion_results.comment_id
        WHERE posts.admin_id = %s
        ORDER BY comments.created_at DESC
    """, (session["admin_id"],))

    comments = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("admin_comments.html", comments=comments)


# =========================
# ADMIN: TRAIN MODEL (SIMULATED)
# =========================
@app.route("/admin/train-model", methods=["POST"])
def train_model():
    if "admin_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    # Check if model training script exists
    if not os.path.exists("train_model.py"):
        return jsonify({
            "status": "error",
            "message": "Training script not found."
        })

    # Start training in background
    try:
        subprocess.Popen(['python', 'train_model.py'])
        print("✅ Model training started in background")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to start training: {str(e)}"})

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Update model status in database
        cursor.execute("""
            INSERT INTO model_status (status, accuracy, last_trained)
            VALUES (%s,%s,%s)
        """, ("Training Started", "Pending", datetime.now()))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "message": "Model training started successfully! Check the console for progress."
        })
    except Exception as e:
        cursor.close()
        conn.close()
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================
# API: Get Emotion Stats
# =========================
@app.route("/admin/api/emotion-stats")
def get_emotion_stats():
    if "admin_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            primary_emotion,
            COUNT(*) as count
        FROM emotion_results
        GROUP BY primary_emotion
        ORDER BY count DESC
    """)

    stats = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify(stats)


# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 STARTING TECCR APPLICATION")
    print("=" * 60)
    print(f"Emotion Predictor: {'✅ Available' if EMOTION_PREDICTOR_AVAILABLE else '❌ Not Available'}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print("=" * 60 + "\n")
    app.run(debug=True)