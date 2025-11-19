"""
Database utilities for check processing workflow.
SQLite database for tracking processed checks and audit trail.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator
from loguru import logger


class Database:
    """SQLite database manager for check processing."""

    def __init__(self, db_path: str):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.

        Yields:
            SQLite connection with row factory enabled

        Raises:
            Exception: Database errors are logged and re-raised
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema if not exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Checks table - main tracking table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_id TEXT UNIQUE NOT NULL,
                    shareable_id TEXT,

                    -- Check details
                    check_number TEXT,
                    routing_number TEXT,
                    account_number TEXT,
                    amount REAL,
                    check_date TEXT,

                    -- Customer info
                    customer_name TEXT,
                    customer_id TEXT,

                    -- Email info
                    email_uid TEXT,
                    email_from TEXT,
                    email_subject TEXT,
                    email_received_at TEXT,

                    -- OCR details
                    ocr_confidence REAL,

                    -- QuickBooks info
                    qbo_salesreceipt_id TEXT,
                    qbo_salesreceipt_number TEXT,
                    qbo_created_at TEXT,

                    -- Processing status
                    status TEXT NOT NULL,  -- pending, processed, failed, manual_review, duplicate
                    error_message TEXT,

                    -- Image info
                    image_path TEXT,
                    image_filename TEXT,

                    -- Timestamps
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    processed_at TEXT,

                    -- Metadata
                    metadata TEXT  -- JSON for additional data
                )
            """
            )

            # Check ID mappings for quick lookups
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS check_id_mappings (
                    check_id TEXT PRIMARY KEY,
                    routing_number TEXT NOT NULL,
                    account_number TEXT NOT NULL,
                    check_number TEXT NOT NULL,
                    amount REAL NOT NULL,
                    date TEXT NOT NULL,
                    customer_id TEXT,
                    created_at TEXT NOT NULL
                )
            """
            )

            # Processing logs for audit trail
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_id TEXT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,  -- JSON
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (check_id) REFERENCES checks(check_id)
                )
            """
            )

            # OAuth tokens storage
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS oauth_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service TEXT NOT NULL,  -- 'quickbooks'
                    access_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    realm_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Create indexes for better query performance
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checks_check_number
                ON checks(check_number)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checks_status
                ON checks(status)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checks_created_at
                ON checks(created_at)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processing_logs_check_id
                ON processing_logs(check_id)
            """
            )

            logger.info(f"Database initialized at {self.db_path}")

    # ===== Check Operations =====

    def insert_check(self, check_data: Dict[str, Any]) -> int:
        """
        Insert a new check record.

        Args:
            check_data: Dictionary containing check information

        Returns:
            Row ID of inserted check
        """
        now = datetime.utcnow().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO checks (
                    check_id, shareable_id, check_number, routing_number,
                    account_number, amount, check_date, customer_name,
                    customer_id, email_uid, email_from, email_subject,
                    email_received_at, ocr_confidence,
                    status, image_path, image_filename,
                    created_at, updated_at, metadata
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
                (
                    check_data.get("check_id"),
                    check_data.get("shareable_id"),
                    check_data.get("check_number"),
                    check_data.get("routing_number"),
                    check_data.get("account_number"),
                    check_data.get("amount"),
                    check_data.get("check_date"),
                    check_data.get("customer_name"),
                    check_data.get("customer_id"),
                    check_data.get("email_uid"),
                    check_data.get("email_from"),
                    check_data.get("email_subject"),
                    check_data.get("email_received_at"),
                    check_data.get("ocr_confidence"),
                    check_data.get("status", "pending"),
                    check_data.get("image_path"),
                    check_data.get("image_filename"),
                    now,
                    now,
                    check_data.get("metadata"),
                ),
            )

            row_id = cursor.lastrowid
            logger.info(
                f"Inserted check {check_data.get('check_id')} with row ID {row_id}"
            )
            return row_id

    def update_check(self, check_id: str, updates: Dict[str, Any]) -> None:
        """
        Update a check record.

        Args:
            check_id: Unique check identifier
            updates: Dictionary of fields to update
        """
        updates["updated_at"] = datetime.utcnow().isoformat()

        # Build SET clause dynamically
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [check_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE checks
                SET {set_clause}
                WHERE check_id = ?
            """,
                values,
            )

            logger.info(f"Updated check {check_id}: {list(updates.keys())}")

    def get_check(self, check_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a check by ID.

        Args:
            check_id: Unique check identifier

        Returns:
            Check data as dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM checks WHERE check_id = ?", (check_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def check_duplicate(
        self, routing_number: str, account_number: str, check_number: str, date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a check with same details already exists.

        Args:
            routing_number: Bank routing number
            account_number: Account number
            check_number: Check number
            date: Check date

        Returns:
            Existing check data if duplicate found, None otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM checks
                WHERE routing_number = ?
                AND account_number = ?
                AND check_number = ?
                AND check_date = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (routing_number, account_number, check_number, date),
            )

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_checks_by_status(
        self, status: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get checks by status.

        Args:
            status: Check status
            limit: Maximum number of results

        Returns:
            List of check records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM checks
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (status, limit),
            )

            return [dict(row) for row in cursor.fetchall()]

    # ===== ID Mapping Operations =====

    def insert_check_id_mapping(self, mapping_data: Dict[str, Any]) -> None:
        """
        Insert check ID mapping for reverse lookup.

        Args:
            mapping_data: Dictionary containing mapping information
        """
        now = datetime.utcnow().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO check_id_mappings (
                    check_id, routing_number, account_number, check_number,
                    amount, date, customer_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    mapping_data["check_id"],
                    mapping_data["routing_number"],
                    mapping_data["account_number"],
                    mapping_data["check_number"],
                    mapping_data["amount"],
                    mapping_data["date"],
                    mapping_data.get("customer_id"),
                    now,
                ),
            )

    def get_check_by_mapping(self, check_id: str) -> Optional[Dict[str, Any]]:
        """
        Get check data from ID mapping.

        Args:
            check_id: Check ID to lookup

        Returns:
            Mapping data or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM check_id_mappings WHERE check_id = ?
            """,
                (check_id,),
            )

            row = cursor.fetchone()
            return dict(row) if row else None

    # ===== Logging Operations =====

    def log_event(
        self, check_id: str, event_type: str, event_data: Optional[str] = None
    ) -> None:
        """
        Log a processing event.

        Args:
            check_id: Check ID this event relates to
            event_type: Type of event
            event_data: Optional JSON data
        """
        now = datetime.utcnow().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO processing_logs (check_id, event_type, event_data, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (check_id, event_type, event_data, now),
            )

    def get_check_logs(self, check_id: str) -> List[Dict[str, Any]]:
        """
        Get all processing logs for a check.

        Args:
            check_id: Check ID

        Returns:
            List of log entries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM processing_logs
                WHERE check_id = ?
                ORDER BY timestamp ASC
            """,
                (check_id,),
            )

            return [dict(row) for row in cursor.fetchall()]

    # ===== OAuth Token Operations =====

    def store_oauth_token(
        self,
        service: str,
        access_token: str,
        refresh_token: str,
        expires_at: str,
        realm_id: Optional[str] = None,
    ) -> None:
        """
        Store or update OAuth tokens.

        Args:
            service: Service name (e.g., 'quickbooks')
            access_token: OAuth access token
            refresh_token: OAuth refresh token
            expires_at: Token expiration timestamp
            realm_id: Optional realm/company ID
        """
        now = datetime.utcnow().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if token exists
            cursor.execute("SELECT id FROM oauth_tokens WHERE service = ?", (service,))
            existing = cursor.fetchone()

            if existing:
                # Update existing token
                cursor.execute(
                    """
                    UPDATE oauth_tokens
                    SET access_token = ?, refresh_token = ?, expires_at = ?,
                        realm_id = ?, updated_at = ?
                    WHERE service = ?
                """,
                    (access_token, refresh_token, expires_at, realm_id, now, service),
                )
            else:
                # Insert new token
                cursor.execute(
                    """
                    INSERT INTO oauth_tokens (
                        service, access_token, refresh_token, expires_at,
                        realm_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        service,
                        access_token,
                        refresh_token,
                        expires_at,
                        realm_id,
                        now,
                        now,
                    ),
                )

            logger.info(f"Stored OAuth token for {service}")

    def get_oauth_token(self, service: str) -> Optional[Dict[str, Any]]:
        """
        Get stored OAuth token.

        Args:
            service: Service name

        Returns:
            Token data or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM oauth_tokens WHERE service = ?
                ORDER BY updated_at DESC LIMIT 1
            """,
                (service,),
            )

            row = cursor.fetchone()
            return dict(row) if row else None

    # ===== Statistics =====

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with various statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total checks
            cursor.execute("SELECT COUNT(*) as count FROM checks")
            stats["total_checks"] = cursor.fetchone()["count"]

            # By status
            cursor.execute(
                """
                SELECT status, COUNT(*) as count
                FROM checks
                GROUP BY status
            """
            )
            stats["by_status"] = {
                row["status"]: row["count"] for row in cursor.fetchall()
            }

            # Average confidence
            cursor.execute(
                """
                SELECT AVG(ocr_confidence) as avg_confidence
                FROM checks
                WHERE ocr_confidence IS NOT NULL
            """
            )
            result = cursor.fetchone()
            stats["avg_confidence"] = result["avg_confidence"] if result else 0

            # Cloud fallback usage
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM checks
                WHERE used_cloud_fallback = 1
            """
            )
            stats["cloud_fallback_count"] = cursor.fetchone()["count"]

            return stats
