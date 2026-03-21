# Email Extractor

A lightweight Python script tool designed to connect to any IMAP server (like Gmail) and extract emails into a beautifully structured, date-grouped JSON file. 

## Features
- **Date-based Searching:** Automatically filter and fetch emails for a specific target date or an entire date range.
- **Structured JSON Output:** Neatly parses complex email headers into organized sender and recipient dictionaries instead of raw strings.
- **Date Grouping:** Automatically aggregates fetched emails using the date they were received as the primary JSON key.
- **Zero External Dependencies (For Python 3.11+):** Uses built-in `imaplib`, `email`, and `tomllib` libraries.

## Prerequisites
- Python 3.11 or newer (to use the built-in `tomllib` configuration parser).
  - *Note: If you are using Python 3.10 or older, you must install the `tomli` backport by running `pip install tomli`.*
- An App Password (if using Gmail or an account with 2FA secured).

## Installation & Setup

1. **Clone or Download** the repository to your machine.
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Configure your settings:**
   Edit the `config.toml` file with your IMAP credentials and search constraints.
   
   ```toml
   [imap]
   server = "imap.gmail.com"
   username = "YOUR_EMAIL_ADDRESS"
   password = "YOUR_APP_PASSWORD"

   [search]
   # You can provide a specific date (Format: YYYY-MM-DD):
   target_date = "2026-03-21"

   # Or a date range (will override target_date if both are set):
   start_date = "2026-03-15"
   end_date = "2026-03-22"

   output_file = "email_data.json"
   ```

   **Important Note for Gmail Users:**
   Google no longer supports signing into third-party IMAP applications with your standard Google account password. You must generate an **App Password**:
   1. Go to your Google Account -> **Security**.
   2. Ensure **2-Step Verification** is enabled.
   3. Search for **App passwords** in your Google account and create a new one.
   4. Take the 16-character generated password and place it in the `password` field in `config.toml`.

## Usage

Once your `config.toml` is filled out, simply run the script:
```bash
python extract_emails.py
```

The script will connect to your provider, search the inbox based on the target dates provided, and extract their `Date`, `Subject`, `Sender`, `Recipients`, and plain-text `Body`.

## Output Format (`email_data.json`)

The output is saved as a dictionary, clustering lists of emails under the specific date they were received:

```json
{
    "emails": {
        "2026-03-21": [
            {
                "time": "2026-03-21T18:45:00+00:00",
                "subject": "Hello World",
                "sender": {
                    "name": "Jane Doe",
                    "email": "jane@example.com"
                },
                "recipients": [
                    {
                        "name": "John Smith",
                        "email": "john@example.com"
                    }
                ],
                "body": "The email content goes here..."
            }
        ]
    }
}
```
