"""Database connection utilities."""

from database.models import create_tables, SessionLocal, engine


def init_database():
    """Initialize database tables."""
    create_tables()
    print("[DB] Database initialized successfully.")


def check_connection():
    """Verify database connectivity."""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"[DB] Connection failed: {e}")
        return False