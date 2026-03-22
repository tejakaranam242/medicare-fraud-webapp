import sqlite3

def create_tables():
    print("Connecting to database...")
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    try:
        print("Creating users table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print("Creating predictions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(100) NOT NULL,
                user_input TEXT NOT NULL,
                model VARCHAR(50) NOT NULL,
                prediction INT NOT NULL,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        print("Tables created successfully!")
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    create_tables()
