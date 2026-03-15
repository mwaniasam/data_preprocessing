"""
pipeline.py — Multimodal Authentication & Product Recommendation CLI
======================================================================
Group: Data Preprocessing Team
Author: Kelvin (Task 6 — System Demonstration)

Usage
-----
Run a single authorized user:
    python pipeline.py --mode authorized --member Kelvin

Run all authorized users:
    python pipeline.py --mode all

Run an unauthorized attempt (face/voice mismatch):
    python pipeline.py --mode unauthorized --face Kelvin --voice Samuel

Run all simulations at once (full demo):
    python pipeline.py --mode demo

Show help:
    python pipeline.py --help

Pipeline Flow (matches assignment diagram)
------------------------------------------
  User Input
      ↓
  [STAGE 1] Face Recognition       → FAIL → Access Denied
      ↓ PASS
  [STAGE 2] Product Recommendation (computed, held until voice passes)
      ↓
  [STAGE 3] Voice Verification     → FAIL → Access Denied
      ↓ PASS
  Display: Welcome + Recommended Product

Requirements
------------
    pip install numpy pandas scikit-learn joblib
"""

import argparse
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Directory where .pkl model files are saved (Task 4 output)
MODELS_DIR = "saved_models"

# Directory where processed data CSVs live (Task 1 + Task 2 + Task 3 output)
DATA_DIR     = os.path.join("data", "processed")
FEATURES_DIR = "features"

# File paths
IMAGE_CSV  = os.path.join(DATA_DIR,     "/home/school/ML_ALU/alu-machine_learning/formatives/data_preprocessing/models/data/processed/image_features.csv")
AUDIO_CSV  = os.path.join(FEATURES_DIR, "/home/school/ML_ALU/alu-machine_learning/formatives/data_preprocessing/features/audio_features.csv")
MERGED_CSV = os.path.join(DATA_DIR,     "/home/school/ML_ALU/alu-machine_learning/formatives/data_preprocessing/models/data/processed/merged_dataset.csv")

# Member → Customer ID mapping
# Each team member is assigned a real customer ID from the merged dataset.
# This is the bridge between biometric identity and purchase history.
# image_features.csv  → Member_1, Member_2, Member_3, Member_4
# face model trained   → on Member_1/2/3/4  → returns Member_1/2/3/4
# audio_features.csv  → David, Kelvin, Michael Kimani, Samuel
# voice model trained  → on real names       → returns real names
#
# Confirmed member assignments:
#   Member_1 = David  | Member_2 = Kelvin
#   Member_3 = Michael Kimani | Member_4 = Samuel

# Step 1 maps: Member_X → Customer ID  (face model returns Member_X)
MEMBER_TO_CUSTOMER_ID = {
    "Member_1": "A192",   # David
    "Member_2": "A190",   # Kelvin
    "Member_3": "A150",   # Michael Kimani
    "Member_4": "A103",   # Samuel
}

# Step 2 maps: Member_X → Real name  (needed to look up audio_df)
FOLDER_TO_REAL_NAME = {
    "Member_1": "David",
    "Member_2": "Kelvin",
    "Member_3": "Michael Kimani",
    "Member_4": "Samuel",
}

# All members use folder names (matches image_df and face model output)
ALL_MEMBERS = list(FOLDER_TO_REAL_NAME.keys())

# Face confidence threshold — below this we reject as unknown
FACE_CONFIDENCE_THRESHOLD = 0.50

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✅  {msg}{RESET}")
def fail(msg):  print(f"  {RED}❌  {msg}{RESET}")
def info(msg):  print(f"  {CYAN}ℹ   {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠   {msg}{RESET}")
def header(msg):
    bar = "═" * 62
    print(f"\n{BOLD}{bar}")
    print(f"  {msg}")
    print(f"{bar}{RESET}")
def stage(num, name):
    print(f"\n{BOLD}  [STAGE {num}]  {name}{RESET}")
    print(f"  {'─' * 50}")

def denied():
    print(f"\n  {RED}{BOLD}{'─'*50}")
    print(f"  🔒  ACCESS DENIED")
    print(f"  {'─'*50}{RESET}\n")

def approved(member, customer_id, product):
    print(f"\n  {GREEN}{BOLD}{'─'*50}")
    print(f"  🎉  AUTHENTICATION SUCCESSFUL")
    print(f"  {'─'*50}{RESET}")
    print(f"  {BOLD}Welcome, {member}!  (Customer ID: {customer_id}){RESET}")
    print(f"  Recommended product category: {BOLD}{CYAN}{product}{RESET}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    """Load all three trained models and their supporting objects."""
    required = [
        "facial_recognition_model.pkl",
        "face_label_encoder.pkl",
        "voiceprint_model.pkl",
        "voice_label_encoder.pkl",
        "audio_scaler.pkl",
        "product_recommendation_model.pkl",
        "product_label_encoder.pkl",
        "product_feature_columns.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        print(f"{RED}ERROR: Missing model files in '{MODELS_DIR}/':{RESET}")
        for m in missing:
            print(f"  • {m}")
        print(f"\n{YELLOW}Run task_4_model_creation.ipynb first to generate the saved_models/ folder.{RESET}\n")
        sys.exit(1)

    models = {
        "face_model"   : joblib.load(os.path.join(MODELS_DIR, "facial_recognition_model.pkl")),
        "face_enc"     : joblib.load(os.path.join(MODELS_DIR, "face_label_encoder.pkl")),
        "voice_model"  : joblib.load(os.path.join(MODELS_DIR, "voiceprint_model.pkl")),
        "voice_enc"    : joblib.load(os.path.join(MODELS_DIR, "voice_label_encoder.pkl")),
        "audio_scaler" : joblib.load(os.path.join(MODELS_DIR, "audio_scaler.pkl")),
        "prod_model"   : joblib.load(os.path.join(MODELS_DIR, "product_recommendation_model.pkl")),
        "prod_enc"     : joblib.load(os.path.join(MODELS_DIR, "product_label_encoder.pkl")),
        "prod_cols"    : joblib.load(os.path.join(MODELS_DIR, "product_feature_columns.pkl")),
    }
    return models

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load feature datasets. Falls back to synthetic audio if CSV not found."""

    # Image features (Task 2)
    if not os.path.exists(IMAGE_CSV):
        print(f"{RED}ERROR: {IMAGE_CSV} not found. Run Task 2 notebook first.{RESET}")
        sys.exit(1)
    image_df = pd.read_csv(IMAGE_CSV)

    # Merged dataset (Task 1)
    if not os.path.exists(MERGED_CSV):
        print(f"{RED}ERROR: {MERGED_CSV} not found. Run Task 1 notebook first.{RESET}")
        sys.exit(1)
    merged_df = pd.read_csv(MERGED_CSV)
    bool_cols = merged_df.select_dtypes(include="bool").columns
    merged_df[bool_cols] = merged_df[bool_cols].astype(int)

    # Audio features (Task 3) — use real file if available, else synthetic
    if os.path.exists(AUDIO_CSV):
        audio_df = pd.read_csv(AUDIO_CSV)
        audio_source = "real"
    else:
        warn(f"audio_features.csv not found in {FEATURES_DIR}/")
        warn("Using synthetic audio features. Place audio_features.csv in features/ for real data.")
        audio_df = _synthetic_audio()
        audio_source = "synthetic"

    # Feature column sets
    face_feature_cols  = [c for c in image_df.columns
                          if c not in ["member", "expression", "augmentation"]]
    audio_feature_cols = [c for c in audio_df.columns
                          if c not in ["member", "phrase_label", "sample_type", "file_name"]]

    return image_df, audio_df, merged_df, face_feature_cols, audio_feature_cols, audio_source


def _synthetic_audio():
    """Generate reproducible synthetic audio features for demo purposes."""
    AUDIO_COLS = [f"mfcc_{i}" for i in range(1, 14)] + ["spectral_rolloff", "rms_energy", "zcr"]
    np.random.seed(42)
    # Synthetic audio must use REAL names (David, Kelvin etc.)
    # because voice model was trained on real names
    REAL_MEMBERS = list(FOLDER_TO_REAL_NAME.values())
    member_centers = {m: np.random.randn(len(AUDIO_COLS)) * 10 for m in REAL_MEMBERS}
    rows = []
    for member in REAL_MEMBERS:
        center = member_centers[member]
        for phrase in ["yes_approve", "confirm_transaction"]:
            for stype in ["original", "augmented"]:
                noise = np.random.randn(len(AUDIO_COLS)) * 0.5
                row = {"member": member, "phrase_label": phrase,
                       "sample_type": stype, "file_name": f"{member}_{phrase}_{stype}.wav"}
                for j, col in enumerate(AUDIO_COLS):
                    row[col] = center[j] + noise[j]
                rows.append(row)
            for k in range(4):
                noise = np.random.randn(len(AUDIO_COLS)) * 0.5
                row = {"member": member, "phrase_label": phrase,
                       "sample_type": "augmented",
                       "file_name": f"{member}_{phrase}_aug{k}.wav"}
                for j, col in enumerate(AUDIO_COLS):
                    row[col] = center[j] + noise[j]
                rows.append(row)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def get_customer_profile(customer_id, merged_df):
    """Fetch the most recent transaction row for a customer ID."""
    subset = merged_df[merged_df["customer_id_new"] == customer_id]
    if subset.empty:
        return None
    return subset.sort_values("purchase_month", ascending=False).iloc[0].to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(face_member, audio_member, models, data, label=None):
    """
    Run the full 3-stage authentication + recommendation pipeline.

    Parameters
    ----------
    face_member  : str  Member whose face features to use as input
    audio_member : str  Member whose voice features to use as input
    models       : dict All loaded model objects
    data         : tuple (image_df, audio_df, merged_df, face_cols, audio_cols, audio_source)
    label        : str  Optional display label for this run
    """
    image_df, audio_df, merged_df, face_cols, audio_cols, _ = data

    if label is None:
        label = f"Face: {face_member}  |  Voice: {audio_member}"

    header(label)

    # ── STAGE 1: Face Recognition ─────────────────────────────────────────
    stage(1, "FACE RECOGNITION")
    info(f"Scanning face input for: {face_member}")
    time.sleep(0.3)

    # image_df uses Member_1/2/3/4 — use face_member directly, no translation
    face_rows = image_df[
        (image_df["member"] == face_member) &
        (image_df["augmentation"] == "original")
    ]
    if face_rows.empty:
        fail(f"No image data for '{face_member}'")
        fail(f"Available in image_df: {list(image_df['member'].unique())}")
        denied()
        return False

    face_row   = face_rows.iloc[0]
    face_input = face_row[face_cols].values.reshape(1, -1)
    face_pred  = models["face_model"].predict(face_input)[0]
    face_proba = models["face_model"].predict_proba(face_input).max()
    face_name  = models["face_enc"].inverse_transform([face_pred])[0]
    # face_name is Member_X (what the model was trained to return)

    print(f"  Member     : {face_member}")
    print(f"  Detected   : {BOLD}{face_name}{RESET}")
    print(f"  Confidence : {face_proba:.1%}")

    if face_proba < FACE_CONFIDENCE_THRESHOLD:
        fail(f"Confidence {face_proba:.1%} below threshold ({FACE_CONFIDENCE_THRESHOLD:.0%}).")
        fail("Face does not match any registered member.")
        denied()
        return False

    ok("Face recognized. Proceeding to product recommendation...")

    # ── STAGE 2: Product Recommendation ──────────────────────────────────
    stage(2, "PRODUCT RECOMMENDATION")
    info(f"Looking up customer profile for: {face_name}")
    time.sleep(0.3)

    customer_id = MEMBER_TO_CUSTOMER_ID.get(face_name)
    if customer_id is None:
        fail(f"No customer ID mapped for member: {face_name}")
        denied()
        return False

    profile = get_customer_profile(customer_id, merged_df)
    if profile is None:
        fail(f"No purchase history found for customer: {customer_id}")
        denied()
        return False

    prod_input  = pd.DataFrame([profile])[models["prod_cols"]].values.astype(float)
    prod_pred   = models["prod_model"].predict(prod_input)[0]
    prod_proba  = models["prod_model"].predict_proba(prod_input).max()
    recommended = models["prod_enc"].inverse_transform([prod_pred])[0]

    print(f"  Customer ID        : {BOLD}{customer_id}{RESET}")
    print(f"  Predicted category : {BOLD}{CYAN}{recommended}{RESET}")
    print(f"  Confidence         : {prod_proba:.1%}")
    ok("Recommendation ready. Proceeding to voice verification...")

    # ── STAGE 3: Voice Verification ───────────────────────────────────────
    stage(3, "VOICE VERIFICATION")
    info(f"Scanning voice input for: {audio_member}")
    time.sleep(0.3)

    # audio_df uses real names — translate Member_X → real name
    real_audio_name = FOLDER_TO_REAL_NAME.get(audio_member, audio_member)

    audio_rows = audio_df[
        (audio_df["member"] == real_audio_name) &
        (audio_df["sample_type"] == "original")
    ]
    if audio_rows.empty:
        fail(f"No audio data for '{audio_member}' (looking for '{real_audio_name}')")
        fail(f"Available in audio_df: {list(audio_df['member'].unique())}")
        denied()
        return False

    audio_row   = audio_rows.iloc[0]
    audio_input = models["audio_scaler"].transform(
        audio_row[audio_cols].values.reshape(1, -1)
    )
    voice_pred  = models["voice_model"].predict(audio_input)[0]
    voice_proba = models["voice_model"].predict_proba(audio_input).max()
    voice_name  = models["voice_enc"].inverse_transform([voice_pred])[0]
    # voice_name is a real name (David/Kelvin/etc.)

    # Translate face_name (Member_X) to real name for comparison
    real_face_name = FOLDER_TO_REAL_NAME.get(face_name, face_name)

    print(f"  Member     : {audio_member}")
    print(f"  Detected   : {BOLD}{voice_name}{RESET}")
    print(f"  Confidence : {voice_proba:.1%}")

    if voice_name != real_face_name:
        fail(f"Voice ({voice_name}) does not match face ({real_face_name}) [{face_name}].")
        denied()
        return False

    ok("Proceed with Transaction.")
    approved(real_face_name, customer_id, recommended)
    return True

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODES
# ─────────────────────────────────────────────────────────────────────────────

def sim_authorized(member, models, data):
    """Run a single authorized user through the full pipeline."""
    if member not in ALL_MEMBERS:
        print(f"{RED}Unknown member '{member}'. Valid options: {ALL_MEMBERS}{RESET}")
        sys.exit(1)
    run_pipeline(member, member, models, data,
                 label=f"AUTHORIZED USER — {member}")

def sim_unauthorized(face_member, voice_member, models, data):
    """Run an unauthorized attempt: face and voice belong to different people."""
    if face_member not in ALL_MEMBERS:
        print(f"{RED}Unknown face member '{face_member}'.{RESET}")
        sys.exit(1)
    if voice_member not in ALL_MEMBERS:
        print(f"{RED}Unknown voice member '{voice_member}'.{RESET}")
        sys.exit(1)
    if face_member == voice_member:
        print(f"{YELLOW}Warning: face and voice are the same member — this is an authorized scenario.{RESET}")
    run_pipeline(face_member, voice_member, models, data,
                 label=f"UNAUTHORIZED ATTEMPT — {face_member} face + {voice_member} voice")

def sim_all_authorized(models, data):
    """Run all 4 members as authorized users."""
    print(f"\n{BOLD}{'▓'*62}")
    print(f"  SIMULATION — ALL AUTHORIZED USERS  ({len(ALL_MEMBERS)} members)")
    print(f"{'▓'*62}{RESET}")
    results = {}
    for member in ALL_MEMBERS:
        result = run_pipeline(member, member, models, data,
                              label=f"Authorized — {member}")
        results[member] = "✅ APPROVED" if result else "❌ DENIED"
    _print_summary_table(results, "All Authorized Users")

def sim_all_unauthorized(models, data):
    """Run multiple unauthorized mismatch scenarios."""
    scenarios = [
        ("Member_2", "Member_4", "Member_2 (Kelvin) face  + Member_4 (Samuel) voice"),
        ("Member_1", "Member_3", "Member_1 (David) face   + Member_3 (Michael) voice"),
        ("Member_3", "Member_1", "Member_3 (Michael) face + Member_1 (David) voice"),
        ("Member_4", "Member_2", "Member_4 (Samuel) face  + Member_2 (Kelvin) voice"),
    ]
    print(f"\n{BOLD}{'▓'*62}")
    print(f"  SIMULATION — UNAUTHORIZED ATTEMPTS  ({len(scenarios)} scenarios)")
    print(f"{'▓'*62}{RESET}")
    results = {}
    for face, voice, label in scenarios:
        result = run_pipeline(face, voice, models, data,
                              label=f"UNAUTHORIZED — {label}")
        results[label] = "✅ APPROVED" if result else "❌ DENIED (correct)"

    _print_summary_table(results, "Unauthorized Attempts")

def sim_full_demo(models, data):
    """Run the complete demo: all authorized + unauthorized scenarios."""
    print(f"\n{BOLD}{CYAN}{'█'*62}")
    print(f"  FULL SYSTEM DEMONSTRATION")
    print(f"  Multimodal Authentication & Product Recommendation")
    print(f"{'█'*62}{RESET}")

    print(f"\n{BOLD}  PART 1 — AUTHORIZED USERS{RESET}")
    print(f"  Each member uses their own face AND their own voice")
    sim_all_authorized(models, data)

    print(f"\n{BOLD}  PART 2 — UNAUTHORIZED ATTEMPTS{RESET}")
    print(f"  Attacker submits one person's face but a different voice")
    sim_all_unauthorized(models, data)

    _print_system_summary(models, data)

def _print_summary_table(results, title):
    """Print a formatted results table."""
    print(f"\n  {'─'*50}")
    print(f"  {BOLD}Results — {title}{RESET}")
    print(f"  {'─'*50}")
    for scenario, outcome in results.items():
        color = GREEN if "APPROVED" in outcome else RED
        print(f"  {color}{scenario:<40} {outcome}{RESET}")
    print(f"  {'─'*50}\n")

def _print_system_summary(models, data):
    """Print the full system architecture summary."""
    _, _, _, face_cols, audio_cols, audio_source = data
    print(f"\n{BOLD}{'═'*62}")
    print(f"  SYSTEM ARCHITECTURE SUMMARY")
    print(f"{'═'*62}{RESET}")
    print(f"""
  Models loaded from: {MODELS_DIR}/
  ┌─────────────────────────────────────────────────────┐
  │  Model                   Algorithm    Features       │
  │  ─────────────────────── ────────── ───────────────  │
  │  Facial Recognition      Rnd Forest  {len(face_cols)} image cols  │
  │  Voiceprint Verification Rnd Forest  {len(audio_cols)} audio cols  │
  │  Product Recommendation  Rnd Forest  13 tabular cols │
  └─────────────────────────────────────────────────────┘

  Member → Customer ID mapping:""")
    for folder, cid in MEMBER_TO_CUSTOMER_ID.items():
        real = FOLDER_TO_REAL_NAME.get(folder, folder)
        print(f"    {folder} ({real:<15}) → {cid}")
    print(f"""
  Audio features source  : {audio_source.upper()}
  Face confidence gate   : {FACE_CONFIDENCE_THRESHOLD:.0%}

  Pipeline order (per assignment diagram):
    Face Recognition → Product Recommendation → Voice Verification
{'═'*62}
""")

# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description=(
            "Multimodal Authentication & Product Recommendation CLI\n"
            "Simulates face recognition → product recommendation → voice verification"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --mode demo
  python pipeline.py --mode authorized --member Kelvin
  python pipeline.py --mode all
  python pipeline.py --mode unauthorized --face Kelvin --voice Samuel
  python pipeline.py --mode unauthorized --face David --voice Michael Kimani

Valid members: David, Kelvin, "Michael Kimani", Samuel
        """
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["demo", "authorized", "unauthorized", "all"],
        help=(
            "demo         : Run full demo (all authorized + unauthorized scenarios)\n"
            "authorized   : Run one authorized user (requires --member)\n"
            "unauthorized : Run a mismatch attack (requires --face and --voice)\n"
            "all          : Run all 4 authorized members"
        )
    )
    parser.add_argument(
        "--member",
        type=str,
        default=None,
        help="Member name for authorized mode. E.g. --member Kelvin"
    )
    parser.add_argument(
        "--face",
        type=str,
        default=None,
        help="Face member for unauthorized mode. E.g. --face Kelvin"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice member for unauthorized mode. E.g. --voice Samuel"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Startup banner
    print(f"\n{BOLD}{CYAN}{'═'*62}")
    print(f"  🔐  Multimodal Authentication System")
    print(f"  📦  Pipeline v1.0  |  Task 6 — System Demonstration")
    print(f"{'═'*62}{RESET}")

    # Load models and data
    print(f"\n  Loading models from {BOLD}{MODELS_DIR}/{RESET}...")
    models = load_models()
    print(f"  {GREEN}All models loaded.{RESET}")

    print(f"  Loading feature data...")
    data = load_data()
    _, _, _, _, _, audio_source = data
    if audio_source == "synthetic":
        warn("Audio data is synthetic. Place audio_features.csv in features/ for real data.")
    print(f"  {GREEN}Data loaded.{RESET}\n")

    # Route to the right simulation
    if args.mode == "demo":
        sim_full_demo(models, data)

    elif args.mode == "authorized":
        if not args.member:
            print(f"{RED}--member is required for authorized mode.{RESET}")
            print(f"Valid members: {ALL_MEMBERS}")
            sys.exit(1)
        sim_authorized(args.member, models, data)

    elif args.mode == "unauthorized":
        if not args.face or not args.voice:
            print(f"{RED}--face and --voice are both required for unauthorized mode.{RESET}")
            print(f"Valid members: {ALL_MEMBERS}")
            sys.exit(1)
        sim_unauthorized(args.face, args.voice, models, data)

    elif args.mode == "all":
        sim_all_authorized(models, data)


if __name__ == "__main__":
    main()
