"""
Configuration settings loaded from environment variables.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Email Configuration
    imap_server: str = Field(..., description="IMAP server address")
    imap_port: int = Field(993, description="IMAP port (default: 993 for SSL)")
    imap_use_ssl: bool = Field(True, description="Use SSL for IMAP")

    smtp_server: str = Field(..., description="SMTP server address")
    smtp_port: int = Field(465, description="SMTP port (default: 465 for SSL)")
    smtp_use_ssl: bool = Field(True, description="Use SSL for SMTP")

    email_username: str = Field(..., description="Email username")
    email_password: str = Field(..., description="Email password")

    email_inbox_folder: str = Field("INBOX", description="Inbox folder name")
    email_processed_folder: str = Field("Processed", description="Processed folder")
    email_failed_folder: str = Field("Failed", description="Failed folder")
    email_manual_review_folder: str = Field(
        "ManualReview", description="Manual review folder"
    )
    email_duplicates_folder: str = Field("Duplicates", description="Duplicates folder")

    imap_poll_interval: int = Field(
        120, description="Poll interval in seconds (default: 120)"
    )

    # QuickBooks Configuration
    qbo_client_id: str = Field(..., description="QBO OAuth client ID")
    qbo_client_secret: str = Field(..., description="QBO OAuth client secret")
    qbo_redirect_uri: str = Field(..., description="OAuth redirect URI")
    qbo_environment: str = Field(
        "production", description="QBO environment (production/sandbox)"
    )
    qbo_realm_id: Optional[str] = Field(None, description="QBO Company/Realm ID")
    qbo_refresh_token: Optional[str] = Field(None, description="Stored refresh token")

    qbo_payment_method_id: str = Field("1", description="Payment method ID for checks")
    qbo_deposit_account_id: str = Field("35", description="Deposit account ID")
    qbo_income_item_id: str = Field("1", description="Income item ID")

    # Google Cloud / Vertex AI Configuration
    google_application_credentials: Optional[str] = Field(
        None, description="Path to Google service account JSON"
    )
    vertex_ai_project_id: str = Field(
        ..., description="Google Cloud project ID for Vertex AI"
    )
    vertex_ai_location: str = Field(
        "us-central1", description="Vertex AI location/region (default: us-central1)"
    )
    vertex_ai_model: str = Field(
        "gemini-2.5-pro", description="Vertex AI model name (default: gemini-2.5-pro)"
    )
    vertex_ai_retry_attempts: int = Field(
        2,
        description="Number of times to retry Vertex AI extractions (including first attempt)",
    )
    vertex_ai_retry_delay: float = Field(
        2.0, description="Initial delay between Vertex AI retry attempts in seconds"
    )
    vertex_ai_retry_backoff_multiplier: float = Field(
        2.0,
        description="Multiplier applied to the retry delay after each failed Vertex attempt",
    )

    # OCR Configuration
    ocr_confidence_threshold: float = Field(
        85.0, description="Minimum confidence for auto-process (default: 85.0)"
    )
    manual_review_threshold: float = Field(
        70.0, description="Threshold for manual review (default: 70.0)"
    )

    # Database
    database_path: str = Field("./data/checks.db", description="SQLite database path")
    database_echo: bool = Field(False, description="Echo SQL queries")

    # Application Settings
    log_level: str = Field("INFO", description="Logging level")
    log_file: str = Field("./logs/check_processor.log", description="Log file path")
    log_rotation: str = Field("10 MB", description="Log rotation size")
    log_retention: str = Field("30 days", description="Log retention period")

    max_retries: int = Field(3, description="Maximum retry attempts (default: 3)")
    retry_delay: int = Field(
        5, description="Initial retry delay in seconds (default: 5)"
    )
    retry_backoff_multiplier: float = Field(
        2.0, description="Retry backoff multiplier (default: 2.0)"
    )

    max_concurrent_processing: int = Field(
        5, description="Max concurrent check processing"
    )
    image_temp_dir: str = Field("./data/temp", description="Temporary image directory")

    # Check Detection Algorithm Parameters (all configurable via .env!)
    check_detector_dpi: int = Field(
        300, description="DPI for PDF rendering and detection (default: 300)"
    )
    check_detector_iqr_factor: float = Field(
        1.5, description="IQR multiplier for gap detection (default: 1.5)"
    )
    check_detector_outlier_threshold: float = Field(
        2.5, description="Z-score threshold for outlier filtering (default: 2.5)"
    )
    check_detector_expand_percent: float = Field(
        4.0, description="Percentage to expand bounding boxes (default: 4.0)"
    )
    min_check_dimension_px: int = Field(
        300, description="Minimum check dimension in pixels (default: 300)"
    )
    min_component_area_px: int = Field(
        80, description="Minimum component area in pixels (default: 80)"
    )
    micr_line_max_height_px: int = Field(
        450, description="Maximum MICR line height in pixels (default: 450)"
    )
    micr_line_max_gap_px: int = Field(
        600, description="Maximum gap to MICR line in pixels (default: 600)"
    )
    base_detection_confidence: float = Field(
        50.0, description="Base confidence for check detection (default: 50.0)"
    )

    # Security
    encryption_key: Optional[str] = Field(
        None, description="Encryption key for sensitive data"
    )

    # Development
    debug: bool = Field(False, description="Enable debug mode")
    dry_run: bool = Field(False, description="Dry run mode (no actual changes)")
    # Primary mailbox where processed/check notifications should be delivered.
    # Set this to the checks inbox so threaded replies land there by default.
    notification_recipient: str = Field(
        "checks@mail.phldems.org",
        description="Primary notification recipient (checks inbox)",
    )

    # Optional test override (kept for compatibility but not used by default flows)
    test_email_recipient: Optional[str] = Field(
        None, description="Test email recipient"
    )

    @field_validator("qbo_environment")
    @classmethod
    def validate_qbo_environment(cls, v: str) -> str:
        """Validate QBO environment value."""
        if v not in ["production", "sandbox"]:
            raise ValueError("qbo_environment must be 'production' or 'sandbox'")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level value."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of: {', '.join(valid_levels)}")
        return v_upper

    def get_qbo_base_url(self) -> str:
        """Get QuickBooks API base URL based on environment."""
        if self.qbo_environment == "sandbox":
            return "https://sandbox-quickbooks.api.intuit.com/v3"
        return "https://quickbooks.api.intuit.com/v3"

    def ensure_directories(self):
        """Ensure required directories exist."""
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.image_temp_dir).mkdir(parents=True, exist_ok=True)


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
