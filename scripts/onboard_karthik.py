import os
import sys
import sqlite3
import datetime
import shutil
import argparse
from security.infrastructure.encryption_service import PIIEncryptionService

def main():
    parser = argparse.ArgumentParser(description="Onboard and assign patient Karthik Kumar to Doctor 4")
    parser.add_argument("--age", type=int, help="Patient age")
    parser.add_argument("--gender", type=str, choices=["Female", "Male", "Other"], help="Patient gender")
    args = parser.parse_args()

    db_path = "outputs/clinical_reports.db"
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    # Get interactive input if not supplied in args
    age = args.age
    while age is None:
        val = input("Enter age for Karthik Kumar (0-120): ").strip()
        if val.isdigit() and 0 <= int(val) <= 120:
            age = int(val)
        else:
            print("Invalid age. Please enter an integer between 0 and 120.")

    gender = args.gender
    while gender is None:
        val = input("Enter gender for Karthik Kumar (Male/Female/Other): ").strip().capitalize()
        if val in ["Male", "Female", "Other"]:
            gender = val
        else:
            print("Invalid gender. Please enter Male, Female, or Other.")

    print(f"\nTarget database: {db_path}")

    # 1. Create a timestamped backup of the production database before changes
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"outputs/clinical_reports.db.backup_onboard_{timestamp}"
    shutil.copy2(db_path, backup_path)
    print(f"Created database backup at: {backup_path}")

    # 2. Check current counts
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        patients_count_before = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        assignments_count_before = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments").fetchone()[0]

        # Verify user exists
        patient_uuid = "67d5923b-9680-4af5-8e02-70af8c09b1de"
        user_row = conn.execute("SELECT full_name FROM users WHERE uuid = ? AND role = 'patient'", (patient_uuid,)).fetchone()
        if not user_row:
            print(f"Error: Patient user {patient_uuid} not found in users table.", file=sys.stderr)
            sys.exit(1)

        # Verify doctor exists
        doctor_id = 4
        doc_row = conn.execute("SELECT email FROM users WHERE id = ? AND role = 'doctor'", (doctor_id,)).fetchone()
        if not doc_row:
            # Maybe look by email doctor@aurascan.ai
            doc_row = conn.execute("SELECT id, email FROM users WHERE email = 'doctor@aurascan.ai' AND role = 'doctor'").fetchone()
            if not doc_row:
                print("Error: Doctor doctor@aurascan.ai not found.", file=sys.stderr)
                sys.exit(1)
            doctor_id = doc_row["id"]

        print(f"Onboarding patient: {user_row['full_name']} ({patient_uuid})")
        print(f"Assigning to doctor: {doc_row['email']} (ID {doctor_id})")

        # 3. Perform changes transactionally
        encryption_service = PIIEncryptionService()
        enc_name = encryption_service.encrypt(user_row["full_name"])
        enc_age = encryption_service.encrypt(str(age))
        enc_gender = encryption_service.encrypt(gender)
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with conn:
            # Check if demographics exist
            demo_exists = conn.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_uuid,)).fetchone()
            if not demo_exists:
                conn.execute(
                    """
                    INSERT INTO patients (patient_id, name, age, gender, created_at)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (patient_uuid, enc_name, enc_age, enc_gender, now_str)
                )
                print("Patient demographics row created.")
            else:
                conn.execute(
                    """
                    UPDATE patients
                    SET name = ?, age = ?, gender = ?
                    WHERE patient_id = ?;
                    """,
                    (enc_name, enc_age, enc_gender, patient_uuid)
                )
                print("Patient demographics row updated.")

            # Create assignment
            conn.execute(
                """
                INSERT OR IGNORE INTO doctor_patient_assignments (doctor_id, patient_id, created_at)
                VALUES (?, ?, ?);
                """,
                (doctor_id, patient_uuid, now_str)
            )
            print("Doctor-patient assignment created.")

        # 4. Verification checks
        fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]

        patients_count_after = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        assignments_count_after = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments").fetchone()[0]

        print("\n=== POST-ONBOARDING VERIFICATION ===")
        print("FK violations:", len(fk_violations))
        print("Integrity check:", integrity)
        print(f"Patients count: {patients_count_before} -> {patients_count_after}")
        print(f"Assignments count: {assignments_count_before} -> {assignments_count_after}")

        if fk_violations:
            raise RuntimeError(f"Foreign key violations found: {fk_violations}")
        if integrity != "ok":
            raise RuntimeError(f"Integrity check failed: {integrity}")

        print("\n==========================================")
        print("PATIENT ONBOARDING REPAIR: SUCCESS")
        print("==========================================")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        # Restore backup
        print("Restoring database from backup...")
        conn.close()
        shutil.copy2(backup_path, db_path)
        print("Database restored successfully.")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
