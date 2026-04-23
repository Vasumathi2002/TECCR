"""
Database Creation Script for TECCR System
WAMP Server Compatible (utf8)
Updated with image_path support for posts
"""

import mysql.connector
from werkzeug.security import generate_password_hash
from datetime import datetime

# ============================
# DATABASE CONFIGURATION
# ============================
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = ''          # WAMP default
DB_NAME = 'teccr_db'


# ============================
# CREATE DATABASE
# ============================
def create_database():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            charset='utf8'
        )
        cursor = conn.cursor()

        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
            "CHARACTER SET utf8 COLLATE utf8_general_ci"
        )

        print(f"✅ Database '{DB_NAME}' created successfully")

        cursor.close()
        conn.close()
        return True

    except mysql.connector.Error as e:
        print(f"❌ Error creating database: {e}")
        return False


# ============================
# CREATE TABLES
# ============================
def create_tables():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8'
        )
        cursor = conn.cursor()

        # USERS
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) NOT NULL,
            email VARCHAR(150) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            created_at DATETIME NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # ADMINSpu
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INT AUTO_INCREMENT PRIMARY KEY,
            admin_id VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(150) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            created_at DATETIME NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # POSTS (with image_path support)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            admin_id INT NOT NULL,
            content TEXT NOT NULL,
            image_path VARCHAR(255) DEFAULT NULL,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (admin_id) REFERENCES admins(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # COMMENTS
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            post_id INT NOT NULL,
            user_id INT NOT NULL,
            comment_text TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # EMOTION RESULTS (TECCR OUTPUT)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotion_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            comment_id INT NOT NULL,
            primary_emotion VARCHAR(50),
            secondary_emotion VARCHAR(50),
            teccr_context VARCHAR(255),
            predicted_at DATETIME NOT NULL,
            FOREIGN KEY (comment_id) REFERENCES comments(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # MODEL STATUS
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_status (
            id INT AUTO_INCREMENT PRIMARY KEY,
            status VARCHAR(50),
            accuracy VARCHAR(10),
            last_trained DATETIME
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # Add image_path column if not exists
        try:
            cursor.execute("ALTER TABLE posts ADD COLUMN image_path VARCHAR(255) DEFAULT NULL")
        except mysql.connector.Error:
            pass  # Column might already exist

        conn.commit()
        cursor.close()
        conn.close()

        print("✅ All tables created successfully")
        return True

    except mysql.connector.Error as e:
        print(f"❌ Error creating tables: {e}")
        return False


# ============================
# INSERT DEFAULT ADMIN
# ============================
def insert_default_admin():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8'
        )
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM admins WHERE admin_id = 'admin'"
        )

        if cursor.fetchone():
            print("ℹ️ Default admin already exists")
        else:
            hashed_password = generate_password_hash("admin123")
            cursor.execute("""
                INSERT INTO admins (admin_id, email, password, created_at)
                VALUES (%s, %s, %s, %s)
            """, (
                "admin",
                "admin@teccr.com",
                hashed_password,
                datetime.now()
            ))
            conn.commit()
            print("✅ Default admin created")
            print("   Admin ID : admin")
            print("   Password : admin123")

        cursor.close()
        conn.close()
        return True

    except mysql.connector.Error as e:
        print(f"❌ Error inserting admin: {e}")
        return False


# ============================
# MAIN
# ============================
def main():
    print("=" * 60)
    print("TECCR DATABASE SETUP (Updated with Image Support)")
    print("=" * 60)

    print("\nStep 1: Creating database...")
    if not create_database():
        return

    print("\nStep 2: Creating tables...")
    if not create_tables():
        return

    print("\nStep 3: Creating default admin...")
    if not insert_default_admin():
        return

    print("\n" + "=" * 60)
    print("✅ TECCR DATABASE SETUP COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nNew Features:")
    print("  • Posts now support image uploads (image_path column)")
    print("  • Upload folder: static/uploads/")


if __name__ == "__main__":
    main()