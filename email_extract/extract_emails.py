import imaplib
import email
from email.header import decode_header
from email.utils import getaddresses, parsedate_to_datetime
from collections import defaultdict
import datetime
import json
import sys

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11
    except ImportError:
        print("Please install 'tomli' to use config.toml on Python < 3.11 (e.g., pip install tomli)")
        sys.exit(1)

# Load configuration from config.toml
try:
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
except FileNotFoundError:
    print("Error: config.toml not found. Please create it first.")
    sys.exit(1)

# Base IMAP server configuration
IMAP_SERVER = config.get("imap", {}).get("server", "imap.gmail.com")
USERNAME = config.get("imap", {}).get("username", "your_email@gmail.com")
PASSWORD = config.get("imap", {}).get("password", "your_app_password")

def clean_text(text):
    """Utility to clean up newline characters in the body text."""
    if text:
        return text.replace('\r', '').replace('\n', ' ').strip()
    return ""

def get_body(msg):
    """Extract the plain text body from the email message."""
    # If the email message is multipart
    if msg.is_multipart():
        for part in msg.walk():
            # Get content type of the part
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            # The email body is usually in text/plain and not an attachment
            if content_type == "text/plain" and "attachment" not in content_disposition:
                try:
                    return part.get_payload(decode=True).decode(errors="ignore")
                except Exception as e:
                    print(f"Error decoding multipart body: {e}")
                    pass
    else:
        # If the email is single part (not multipart)
        try:
            return msg.get_payload(decode=True).decode(errors="ignore")
        except Exception as e:
            print(f"Error decoding body: {e}")
            pass
    return ""

def get_decoded_header(header_value):
    """Decode an email header value."""
    if not header_value:
        return ""
        
    decoded_fragments = decode_header(header_value)
    header_str = ""
    for fragment, encoding in decoded_fragments:
        if isinstance(fragment, bytes):
            # If it's a bytes object, decode it to string
            header_str += fragment.decode(encoding if encoding else "utf-8", errors="ignore")
        else:
            # If it's already a string, just append
            header_str += fragment
    return header_str

def extract_emails(target_date_str=None, start_date_str=None, end_date_str=None, output_file="extracted_emails.json"):
    """
    Connect to IMAP server, fetch the emails matching a date or date range and save to a JSON file.
    """
    if USERNAME == "your_email@gmail.com" or PASSWORD == "your_app_password":
        print("Warning: Please set your EMAIL_USERNAME and EMAIL_PASSWORD (preferably using environment variables).")

    print(f"Connecting to {IMAP_SERVER}...")
    try:
        # Create an IMAP client instance and connect to the server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"Failed to connect or login: {e}")
        return

    # Select the mailbox (usually 'inbox')
    mail.select("inbox")

    search_query = "ALL"
    if start_date_str and end_date_str:
        try:
            start_dt = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
            # For inclusive end date behavior in IMAP, we can add 1 day to end_date since BEFORE is exclusive
            end_dt = datetime.datetime.strptime(end_date_str, "%Y-%m-%d") + datetime.timedelta(days=1)
            
            imap_start = start_dt.strftime("%d-%b-%Y")
            imap_end = end_dt.strftime("%d-%b-%Y")
            
            search_query = f'(SINCE "{imap_start}" BEFORE "{imap_end}")'
            print(f"Fetching emails from {start_date_str} up to {end_date_str}...")
        except ValueError:
            print("Invalid date format in range. Expected YYYY-MM-DD.")
    elif target_date_str:
        try:
            dt = datetime.datetime.strptime(target_date_str, "%Y-%m-%d")
            imap_date = dt.strftime("%d-%b-%Y")
             # Use IMAP ON keyword to filter by exact date
            search_query = f'ON "{imap_date}"'
            print(f"Fetching emails specifically for date: {target_date_str}...")
        except ValueError:
            print(f"Invalid date format: '{target_date_str}'. Expected YYYY-MM-DD. Fetching ALL instead.")

    status, messages = mail.search(None, search_query)
    
    if status != "OK":
        print("Search failed or no messages found!")
        mail.close()
        mail.logout()
        return

    # Convert the messages string into a list of email IDs
    email_ids = messages[0].split()
    
    if not email_ids:
        print(f"No emails found for query: {search_query}")
        mail.close()
        mail.logout()
        return

    extracted_data = defaultdict(list)
    print(f"Found {len(email_ids)} emails. Extracting contents...")

    for num in email_ids:
        # Fetch the email message by ID # RFC822 is a standard format for email messages, Don't just give me the snippets; give me the full, raw source code of this email.
        status, msg_data = mail.fetch(num, "(RFC822)")
        
        if status != "OK":
            print(f"Failed to fetch email ID: {num}")
            continue

        for response_part in msg_data:
            if isinstance(response_part, tuple):
                # Parse the raw email bytes into a message object
                msg = email.message_from_bytes(response_part[1])
                
                # Extract and decode headers
                subject = get_decoded_header(msg.get("Subject"))
                
                # Parse From
                from_raw = get_decoded_header(msg.get("From"))
                from_list = getaddresses([from_raw])
                sender = {"name": from_list[0][0], "email": from_list[0][1]} if from_list else {"name": "", "email": from_raw}
                
                # Parse To
                to_raw = get_decoded_header(msg.get("To"))
                recipients = [{"name": name, "email": email_addr} for name, email_addr in getaddresses([to_raw])] if to_raw else []
                
                # Parse Date into ISO format
                date_raw = get_decoded_header(msg.get("Date"))
                iso_date = date_raw
                date_key = "Unknown Date"
                if date_raw:
                    try:
                        dt = parsedate_to_datetime(date_raw)
                        iso_date = dt.isoformat()
                        date_key = dt.strftime("%Y-%m-%d")
                    except (TypeError, ValueError):
                        pass  # Fallback to raw date if parsing fails
                
                # Extract body
                body = get_body(msg)
                
                # Append to our collected data dict
                email_info = {
                    "time": iso_date,
                    "subject": subject,
                    "sender": sender,
                    "recipients": recipients,
                    "body": clean_text(body)
                }
                extracted_data[date_key].append(email_info)

    # Dump the extracted data to a JSON file as a dictionary
    print(f"Saving extracted data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        # Save as standard dict to JSON
        json.dump({"emails": dict(extracted_data)}, f, ensure_ascii=False, indent=4)
        
    # Count total emails processed
    total_extracted = sum(len(emails) for emails in extracted_data.values())
    print(f"Successfully saved {total_extracted} emails to {output_file}")
    
    # Clean up the connection
    mail.close()
    mail.logout()

if __name__ == "__main__":
    # Load constraints from config
    target_date = config.get("search", {}).get("target_date")
    start_date = config.get("search", {}).get("start_date")
    end_date = config.get("search", {}).get("end_date")
    output_filename = config.get("search", {}).get("output_file", "email_data.json")
    
    extract_emails(
        target_date_str=target_date, 
        start_date_str=start_date, 
        end_date_str=end_date, 
        output_file=output_filename
    )
