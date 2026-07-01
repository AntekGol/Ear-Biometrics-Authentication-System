import psycopg2
import numpy as np
from cryptography.fernet import Fernet
import logging
from datetime import datetime, timedelta

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection configuration
DB_HOST = "localhost"
DB_NAME = "ear_db"
DB_USER = "postgres"
DB_PASSWORD = "post"
DB_PORT = "5432"

# Key generation: Fernet.generate_key()
FERNET_KEY = b'tz8Uj8J4SvZ7xzw9-OrPJQHPNP-UJKyfnX1GlkXvaaU='


def encrypt_embedding(embedding):
    #Encrypt embedding to binary format
    try:
        fernet = Fernet(FERNET_KEY)
        # Convert embedding to bytes
        embedding_bytes = embedding.tobytes()
        return fernet.encrypt(embedding_bytes)
    except Exception as e:
        logger.error(f"Błąd szyfrowania: {e}")
        raise


def decrypt_embedding(encrypted):
    #Decrypt embedding from binary format
    try:
        fernet = Fernet(FERNET_KEY)
        # Decrypt to bytes
        decrypted_bytes = fernet.decrypt(bytes(encrypted))
        # Convert back to numpy array
        return np.frombuffer(decrypted_bytes, dtype=np.float32)
    except Exception as e:
        logger.error(f"Błąd deszyfrowania: {e}")
        raise


def init_db():
    #Initialize database connection and create table
    try:
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                ear_embedding BYTEA NOT NULL,
                failed_attempts INTEGER DEFAULT 0,
                lockout_until TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Baza danych zainicjalizowana")
    except Exception as e:
        logger.error(f"Błąd inicjalizacji bazy: {e}")
        raise


def save_user(username, embedding):
    """Save user to the database"""
    try:
        encrypted = encrypt_embedding(embedding)
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, ear_embedding) VALUES (%s, %s)",
            (username, encrypted)
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Zapisano użytkownika: {username}")
    except Exception as e:
        logger.error(f"Błąd zapisu użytkownika: {e}")
        raise


def get_user_embedding(username):
    """Retrieve user embedding from the database"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT ear_embedding FROM users WHERE username = %s", (username,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            # Convert to bytes if necessary
            encrypted_data = result[0]
            if isinstance(encrypted_data, memoryview):
                encrypted_data = encrypted_data.tobytes()
            elif isinstance(encrypted_data, bytearray):
                encrypted_data = bytes(encrypted_data)
            return decrypt_embedding(encrypted_data)
        return None
    except Exception as e:
        logger.error(f"Błąd pobierania użytkownika: {e}")
        raise


def user_exists(username):
    #Check if user exists in the database
    try:
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
        exists = cur.fetchone() is not None
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        logger.error(f"Błąd sprawdzania użytkownika: {e}")
        return False


def increment_failed_attempts(username):
    #Increment the failed login attempts counter
    try:
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        cur = conn.cursor()

        # Increment failed attempts counter
        cur.execute("""
            UPDATE users 
            SET failed_attempts = failed_attempts + 1 
            WHERE username = %s
            RETURNING failed_attempts;
        """, (username,))

        result = cur.fetchone()
        if result and result[0] >= 3:
            # Set lockout for 5 minutes
            lockout_time = datetime.now() + timedelta(minutes=5)
            cur.execute("""
                UPDATE users 
                SET lockout_until = %s 
                WHERE username = %s
            """, (lockout_time, username))

        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Błąd inkrementacji prób: {e}")
        return False


def reset_failed_attempts(username):
    #Reset the failed attempts counter
    try:
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("""
            UPDATE users 
            SET failed_attempts = 0, lockout_until = NULL 
            WHERE username = %s
        """, (username,))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Błąd resetowania prób: {e}")
        return False


def is_account_locked(username):
    #Check if account is locked
    try:
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT lockout_until 
            FROM users 
            WHERE username = %s
        """, (username,))

        result = cur.fetchone()
        cur.close()
        conn.close()

        if result and result[0]:
            lockout_time = result[0]
            if lockout_time > datetime.now():
                return True
        return False
    except Exception as e:
        logger.error(f"Błąd sprawdzania blokady: {e}")
        return False


#Initialization on first run
try:
    init_db()
except:
    logger.warning("Błąd inicjalizacji bazy, kontynuowanie...")