import argparse
import base64
import os
import threading
import time
import hashlib
import pickle
from encryption import Encryptor
from recognize import Recognizer
from otp_verification import send_otp_email, generate_otp, verify_otp, validate_email
from password_verification import verify_password

# User-label mapping
user_mapping = {
    "Aryan": "user",
    "Samyukta": "user0",
    "Shivangi": "user2",
    "Vinithra": "user3",
    # Add more users as needed
}

# Function to retrieve embeddings for a user
def retrieve_embeddings_for_user(username):
    embeddings_file = "./pickle/embeddings.pickle"
    try:
        with open(embeddings_file, "rb") as f:
            data = pickle.load(f)
            user_embeddings = data.get("embeddings")
            user_names = data.get("names")
            if user_names and user_embeddings:
                user_index = [i for i, name in enumerate(user_names) if name == username]
                if user_index:
                    return user_embeddings[user_index[0]]
                else:
                    print(f"No embeddings found for user: {username}")
            else:
                print("Error: No data found in embeddings file.")
    except FileNotFoundError:
        print(f"Error: Embeddings file '{embeddings_file}' not found.")
    except Exception as e:
        print(f"Error retrieving embeddings: {e}")
    return None

# Function to generate AES key based on hashed embeddings
def generate_aes_key_from_embeddings(embeddings):
    # Hash the concatenated embeddings using SHA-256
    concatenated_embeddings = b''.join(embeddings)
    hashed_embeddings = hashlib.sha256(concatenated_embeddings).digest()
    return hashed_embeddings[:32]  # Use first 32 bytes as AES key (256 bits)

# Function to authenticate user based on detected faces
def authenticate_user_face():
    recog = Recognizer(user_mapping)
    user_name = recog.recognize()
    return user_name

# Function to update time left for OTP verification
def update_time_left(start_time, event):
    while not event.is_set():
        time_left = int(300 - (time.time() - start_time))  # Calculate time left
        if time_left <= 0:
            break
        print(f"\rTime left: {time_left} seconds", end='', flush=True)
        time.sleep(1)

# Function to authenticate user using OTP
def authenticate_user_otp():
    attempts = 3
    while attempts > 0:
        receiver_email = input("Enter your email address: ")
        if not validate_email(receiver_email):
            print("Invalid email format. Please enter a valid email address.")
            continue
        
        generated_otp = generate_otp()
        send_otp_email(receiver_email, generated_otp)
        
        start_time = time.time()
        
        timer_event = threading.Event()
        time_thread = threading.Thread(target=update_time_left, args=(start_time, timer_event))
        time_thread.start()
        print("\n")
        timer_event.set()
        
        input_otp = input("Enter the OTP received: ")
    
        time_thread.join() 
        
        if verify_otp(input_otp, generated_otp):  # Pass the generated OTP here
            print("\nOTP Verified.")
            print("Look at the Camera")
            return True
        else:
            print("\nInvalid OTP. Please try again.")
        
        attempts -= 1
    print("OTP verification failed.")
    return False

# Main functionality
def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="Path to file")
    ap.add_argument("-m", "--mode", required=True, help="Enter 'encrypt' or 'decrypt'")
    args = vars(ap.parse_args())

    # Extract arguments
    file_path = args['file']
    mode = args['mode']

    # Main functionality
    if mode == "encrypt":
        # Authenticate user using face detection
        authenticated_user = authenticate_user_face()
        if authenticated_user:
            # Retrieve embeddings for authenticated user
            embeddings = retrieve_embeddings_for_user(authenticated_user)
            if embeddings:
                # Generate AES key based on hashed embeddings
                aes_key = generate_aes_key_from_embeddings(embeddings)
                enc = Encryptor(aes_key)  # Initialize Encryptor object with AES key
                try:
                    enc.encrypt_file(file_path)
                    print("File Encrypted")
                except Exception as e:
                    print(f"Encryption failed: {e}")
            else:
                print("No embeddings found for authenticated user.")
        else:
            print("Face Authentication Failed. Exiting.")
        exit()

    elif mode == "decrypt":
        # Add decryption logic here
        pass

    else:
        print("Incorrect Mode")

if __name__ == "__main__":
    main()


