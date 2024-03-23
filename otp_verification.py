import smtplib
import ssl
import secrets
from email.mime.text import MIMEText
import re
import time
import random
import string

# Function to send OTP email
def send_otp_email(receiver_email, otp):
    sender_email = "samyuktapathak1131@gmail.com"  # Sender email address
    password = "mauvtjaqluphqrzl"  # Sender email password
    subject = "OTP for Verification"
    message = f"Your OTP for Login Verification is: {otp}"

    msg = MIMEText(message)
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

otp_storage = {}

# Function to generate a random OTP
def generate_otp():
    otp = ''.join(random.choices(string.digits, k=6))  # Generate a 6-digit OTP
    timestamp = time.time() 
    otp_storage[otp] = timestamp  # Store the OTP along with its creation time
    return otp

# Function to verify the OTP within a specified time limit
def verify_otp(input_otp, generated_otp):
    if input_otp in otp_storage:
        otp_timestamp = otp_storage[input_otp]
        if time.time() - otp_timestamp <= 300: 
            del otp_storage[input_otp]  # Remove the OTP from storage
            return True
    return False

# Function to validate email format
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email)
