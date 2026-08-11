import os
import sys
import sqlite3
import argparse
import shutil
from typing import Optional

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from security.infrastructure.encryption_service import PIIEncryptionService, EncryptionKeyMissingError

def main():
    parser = argparse.ArgumentParser(description="PII Database Encryption/Decryption Migration Tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--encrypt", action="store_true", help="Encrypt PII patient fields (name, age, gender)")
    group.add_argument("--rollback", action="store_true", help="Rollback (decrypt) PII patient fields back to plaintext")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry-run without writing to the database")
    parser.add_argument("--db", type=str, default="outputs/clinical_reports.db", help="Path to SQLite database")
    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        print(f"Error: Database file does not exist at '{db_path}'")
        sys.exit(1)

    # Initialize encryption service
    try:
        encryption_service = PIIEncryptionService()
    except EncryptionKeyMissingError as e:
        print(f"Error: {e}")
        print("Please configure the PII_ENCRYPTION_KEY environment variable.")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing encryption service: {e}")
        sys.exit(1)

    print(f"Database: {os.path.abspath(db_path)}")
    print(f"Dry-run: {args.dry_run}")
    
    if not args.dry_run and args.encrypt:
        # Prompt for backup or notify
        backup_path = db_path + ".backup"
        print(f"Creating a temporary backup at '{backup_path}'...")
        try:
            shutil.copy2(db_path, backup_path)
            print("Backup created successfully.")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}. Proceeding at your own risk...")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Load all patients
        cursor.execute("SELECT patient_id, name, age, gender FROM patients;")
        patients = cursor.fetchall()
    except Exception as e:
        print(f"Error reading patients table: {e}")
        conn.close()
        sys.exit(1)

    total = len(patients)
    already_encrypted = 0
    already_plaintext = 0
    migrated = 0
    failed = 0

    updates = []

    for p in patients:
        p_id = p["patient_id"]
        name = p["name"]
        age = p["age"]
        gender = p["gender"]

        # Determine encryption status of each field
        name_is_enc = encryption_service.is_encrypted(name)
        age_is_enc = encryption_service.is_encrypted(age)
        gender_is_enc = encryption_service.is_encrypted(gender)

        if args.encrypt:
            # We want to encrypt. If all fields are already encrypted, skip
            if name_is_enc and age_is_enc and gender_is_enc:
                already_encrypted += 1
                continue
            
            # Encrypt fields that are plaintext
            try:
                new_name = encryption_service.encrypt(name) if not name_is_enc else name
                new_age = encryption_service.encrypt(age) if not age_is_enc else age
                new_gender = encryption_service.encrypt(gender) if not gender_is_enc else gender
                updates.append((new_name, new_age, new_gender, p_id))
                migrated += 1
            except Exception as e:
                print(f"Failed to encrypt patient {p_id}: {e}")
                failed += 1

        elif args.rollback:
            # We want to rollback to plaintext. If all fields are already plaintext, skip
            if not name_is_enc and not age_is_enc and not gender_is_enc:
                already_plaintext += 1
                continue

            try:
                # Decrypt
                new_name = encryption_service.decrypt(name) if name_is_enc else name
                new_age = encryption_service.decrypt(age) if age_is_enc else age
                
                # Cast age back to integer if it represents an integer
                if new_age is not None:
                    try:
                        new_age = int(new_age)
                    except ValueError:
                        pass

                new_gender = encryption_service.decrypt(gender) if gender_is_enc else gender
                updates.append((new_name, new_age, new_gender, p_id))
                migrated += 1
            except Exception as e:
                print(f"Failed to decrypt/rollback patient {p_id}: {e}")
                failed += 1

    # Perform updates if not dry-run
    if not args.dry_run and updates:
        try:
            with conn:
                conn.executemany(
                    "UPDATE patients SET name = ?, age = ?, gender = ? WHERE patient_id = ?;",
                    updates
                )
            print("Database updates written successfully.")
        except Exception as e:
            print(f"Error executing database updates: {e}")
            conn.close()
            sys.exit(1)

    conn.close()

    print("\nMigration Results:")
    print(f"Patients scanned: {total}")
    if args.encrypt:
        print(f"Already encrypted: {already_encrypted}")
        print(f"Migrated (encrypted): {migrated}")
    else:
        print(f"Already plaintext: {already_plaintext}")
        print(f"Migrated (decrypted): {migrated}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
