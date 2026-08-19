import time
import os
import re
import urllib.parse
import requests
import sqlite3
from playwright.sync_api import sync_playwright

def handle_login(page, email, password, target_url):
    page.fill("#login-email", email)
    page.fill("#login-password", password)
    page.click("button[type='submit']")

    # Wait for either redirection or 2FA modal
    page.wait_for_function(
        f"() => window.location.href.includes('{target_url}') || !document.getElementById('otp-modal').classList.contains('hidden')",
        timeout=10000
    )

    if target_url in page.url:
        print(f"Direct login success for {email} to {page.url}")
    else:
        # OTP Modal is visible
        otp_text = page.text_content("#otp-hint")
        print(f"OTP Hint visible for {email}: {otp_text}")
        otp_code = re.search(r"\b\d{6}\b", otp_text).group(0)
        print(f"Extracted OTP code: {otp_code}")

        page.fill("#otp-code", otp_code)
        page.click("#otp-modal button[type='submit']")
        page.wait_for_url(f"**{target_url}", timeout=10000)
        print(f"Login success via 2FA for {email} to {page.url}")

def run_e2e_test():
    # Clean up test patients from the database to ensure a pristine starting state
    db_path = "outputs/clinical_reports.db"
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM doctor_patient_assignments WHERE patient_id IN ('pat_verify_e2e_browser_1', 'pat_verify_e2e_browser_2', 'pat_verify_e2e_browser_3')")
            conn.execute("DELETE FROM patients WHERE patient_id IN ('pat_verify_e2e_browser_1', 'pat_verify_e2e_browser_2', 'pat_verify_e2e_browser_3')")
            conn.execute("DELETE FROM reports WHERE patient_id IN ('pat_verify_e2e_browser_1', 'pat_verify_e2e_browser_2', 'pat_verify_e2e_browser_3')")
        print("Test database cleaned successfully before E2E run.")
    except Exception as e:
        print(f"Error cleaning test database: {e}")
    finally:
        conn.close()

    scan_file = os.path.abspath("outputs/clinical_reports/temp_scan.png")
    # Generate a high-contrast mock brain MRI image to satisfy quality gate
    import cv2
    import numpy as np
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.ellipse(img, (128, 128), (110, 110), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (128, 128), (80, 80), 0, 0, 360, (0, 0, 0), -1)
    cv2.ellipse(img, (128, 128), (50, 50), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (128, 128), (20, 20), 0, 0, 360, (0, 0, 0), -1)
    os.makedirs(os.path.dirname(scan_file), exist_ok=True)
    cv2.imwrite(scan_file, img)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # ----------------------------------------------------
        # 1. DOCTOR A LOG IN & REPORT GENERATION FOR PATIENT X
        # ----------------------------------------------------
        print("\n=== E2E TEST: DOCTOR A FLOW ===")
        context_a = browser.new_context(viewport={"width": 1920, "height": 1080})
        page_a = context_a.new_page()

        # Print browser console logs
        page_a.on("console", lambda msg: print(f"[BROWSER CONSOLE] {msg.text}"))

        try:
            page_a.goto("http://127.0.0.1:5000/")
            handle_login(page_a, "doctor@aurascan.ai", "Password@123", "/doctor")

            # Wait for page scripts to load
            page_a.wait_for_timeout(2000)

            # Explicitly onboard/assign Patient X (pat_verify_e2e_browser_1) to Doctor A
            token_a = page_a.evaluate("localStorage.getItem('auth_token')")
            print(f"Doctor A auth token retrieved. Onboarding pat_verify_e2e_browser_1...")
            resp_onboard = requests.post(
                "http://127.0.0.1:8000/api/doctor/assign-patient?patient_id=pat_verify_e2e_browser_1",
                headers={"Authorization": f"Bearer {token_a}"}
            )
            print(f"Onboarding API status: {resp_onboard.status_code}, response: {resp_onboard.text}")

            # Switch tab to 'generate' so elements become visible
            print("Switching to generate report tab...")
            page_a.click("#nav-generate")
            page_a.wait_for_selector("#tab-generate:not(.hidden)", timeout=5000)

            # Generate diagnostic report for patient X (pat_verify_e2e_browser_1)
            print("Doctor A is filling report generation form for Patient X (pat_verify_e2e_browser_1)...")
            page_a.fill("#gen-patient-id", "pat_verify_e2e_browser_1")
            page_a.fill("#gen-patient-name", "Jane E2E Doe")
            page_a.fill("#gen-patient-age", "30")
            page_a.select_option("#gen-patient-gender", "Female")
            page_a.fill("#gen-ref-physician", "Dr. Default Doctor")
            page_a.set_input_files("#gen-file", scan_file)

            # Start diagnostic pipeline
            page_a.click("#submit-btn")

            # Wait for either details modal or Quality override modal
            print("Waiting for diagnostic pipeline completion or quality override modal...")
            page_a.wait_for_function(
                "() => !document.getElementById('details-modal').classList.contains('hidden') || !document.getElementById('mri-quality-override-modal').classList.contains('hidden')",
                timeout=45000
            )

            if not page_a.locator("#mri-quality-override-modal").is_hidden():
                print("Quality gate triggered. Bypassing with clinical reason...")
                page_a.fill("#mri-override-reason", "Slight movement but structure is clear and acceptable.")
                page_a.click("#mri-override-confirm-btn")

            page_a.wait_for_selector("#details-modal:not(.hidden)", timeout=45000)
            print("Diagnostic report generated successfully. Details modal visible.")

            # Close details modal using specific button
            page_a.click("button[onclick*='details-modal']")

            # Reload page and go to Patients registry tab
            page_a.reload()
            page_a.wait_for_timeout(2000)
            page_a.click("#nav-database")

            # Verify patient X appears in doctor dashboard list
            print("Verifying Jane E2E Doe is visible in the registry...")
            page_a.wait_for_selector("#registry-table-body >> text=Jane E2E Doe", timeout=10000)
            print("Jane E2E Doe successfully verified in Doctor A's patient list.")

            # Logout Doctor A
            page_a.click("button:has-text('Sign Out'), button:has-text('Log Out'), button[onclick*='handleLogout']")
            page_a.wait_for_url("**/", timeout=5000)
            context_a.close()
        except Exception as e:
            # Capture diagnostic screenshot in the artifacts directory
            screenshot_path = "C:/Users/Asus/.gemini/antigravity-ide/brain/b6a93975-cd67-4774-b42f-3ed64385928c/browser_error.png"
            page_a.screenshot(path=screenshot_path)
            print(f"E2E Verification failed. Diagnostic screenshot saved to {screenshot_path}")
            browser.close()
            raise e

        # ----------------------------------------------------
        # 2. DOCTOR B ATTEMPTS TO ACCESS PATIENT X DATA
        # ----------------------------------------------------
        print("\n=== E2E TEST: DOCTOR B FLOW ===")
        context_b = browser.new_context(viewport={"width": 1920, "height": 1080})
        page_b = context_b.new_page()
        page_b.goto("http://127.0.0.1:5000/")

        handle_login(page_b, "h32_auth_test_direct@aurascan.ai", "Password@123", "/doctor")

        # Verify Patient X is NOT present in Doctor B's dashboard list
        dashboard_content = page_b.content()
        if "Jane E2E Doe" in dashboard_content or "pat_verify_e2e_browser_1" in dashboard_content:
            print("FAILURE: Patient X is visible to Doctor B!")
            exit(1)
        else:
            print("SUCCESS: Patient X is NOT visible in Doctor B's patient list.")

        # Get authentication token of Doctor B
        auth_token_b = page_b.evaluate("localStorage.getItem('auth_token')")
        print(f"Doctor B auth token retrieved.")

        # Direct API request to get Patient X profile using Doctor B's credentials
        api_url = "http://127.0.0.1:8000/api/doctor/patients/pat_verify_e2e_browser_1"
        resp = requests.get(api_url, headers={"Authorization": f"Bearer {auth_token_b}"})
        print(f"Doctor B direct API request to Patient X profile status: {resp.status_code}")
        if resp.status_code == 403:
            print("SUCCESS: Doctor B direct API access to Patient X is forbidden (403).")
        else:
            print(f"FAILURE: Doctor B direct API access returned: {resp.status_code}")
            exit(1)

        # Logout Doctor B
        page_b.click("button:has-text('Sign Out'), button:has-text('Log Out'), button[onclick*='handleLogout']")
        page_b.wait_for_url("**/", timeout=5000)
        context_b.close()

        # ----------------------------------------------------
        # 3. PATIENT X ATTEMPTS OWN RECORDS AND ADMIN CHECKS
        # ----------------------------------------------------
        print("\n=== E2E TEST: PATIENT X & PATIENT Y ISOLATION ===")
        context_px = browser.new_context(viewport={"width": 1920, "height": 1080})
        page_px = context_px.new_page()
        page_px.goto("http://127.0.0.1:5000/")

        handle_login(page_px, "pat_verify_e2e_browser_1@aurascan.ai", "Password@123", "/patient")

        # Patient X should see their own demographics
        page_px.wait_for_selector("text=Jane E2E Doe", timeout=5000)
        print("SUCCESS: Patient X successfully accesses own profile data.")

        # Verify Referring Physician displays assigned doctor details
        page_px.wait_for_selector("#stat-physician:has-text('doctor@aurascan.ai')", timeout=5000)
        print("SUCCESS: Patient X dashboard displays assigned doctor details.")

        # Get Patient X auth token
        token_px = page_px.evaluate("localStorage.getItem('auth_token')")
        context_px.close()

        # Patient Y tries to access Patient X profile
        print("\nChecking cross-patient access (Patient Y accessing Patient X)...")
        context_py = browser.new_context(viewport={"width": 1920, "height": 1080})
        page_py = context_py.new_page()
        page_py.goto("http://127.0.0.1:5000/")
        handle_login(page_py, "pat_verify_e2e_browser_2@aurascan.ai", "Password@123", "/patient")

        # Verify Patient Y Referring Physician displays "Not Assigned"
        page_py.wait_for_selector("#stat-physician:has-text('Not Assigned')", timeout=5000)
        print("SUCCESS: Unassigned Patient Y dashboard displays 'Not Assigned'.")

        token_py = page_py.evaluate("localStorage.getItem('auth_token')")
        context_py.close()

        # Patient Y tries to request Patient X's profile via direct API
        resp_py = requests.get(
            "http://127.0.0.1:8000/api/doctor/patients/pat_verify_e2e_browser_1",
            headers={"Authorization": f"Bearer {token_py}"}
        )
        print(f"Patient Y direct API request to Patient X profile status: {resp_py.status_code}")
        if resp_py.status_code == 403:
            print("SUCCESS: Cross-patient access is forbidden (403).")
        else:
            print(f"FAILURE: Cross-patient access was allowed: {resp_py.status_code}")
            exit(1)

        # ----------------------------------------------------
        # 4. ADMIN ACCESS TO PATIENT X DATA
        # ----------------------------------------------------
        print("\n=== E2E TEST: ADMIN FLOW ===")
        context_admin = browser.new_context(viewport={"width": 1920, "height": 1080})
        page_admin = context_admin.new_page()
        page_admin.goto("http://127.0.0.1:5000/")

        handle_login(page_admin, "admin@aurascan.ai", "Password@123", "/admin")

        token_admin = page_admin.evaluate("localStorage.getItem('auth_token')")
        context_admin.close()

        # Admin requests Patient X's profile via direct API
        resp_admin = requests.get(
            "http://127.0.0.1:8000/api/doctor/patients/pat_verify_e2e_browser_1",
            headers={"Authorization": f"Bearer {token_admin}"}
        )
        print(f"Admin direct API request to Patient X profile status: {resp_admin.status_code}")
        if resp_admin.status_code == 200:
            print("SUCCESS: Admin can access Patient X profile (200).")
        else:
            print(f"FAILURE: Admin request returned status: {resp_admin.status_code}")
            exit(1)

        print("\n=== E2E PLAYWRIGHT BROWSER VERIFICATION PASSED ===")
        browser.close()

if __name__ == "__main__":
    run_e2e_test()
