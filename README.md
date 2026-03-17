# Data Preprocessing — Formative 2

Multimodal authentication and product recommendation system built for our Formative 2 assignment. The idea is straightforward: we verify a user through their face and voice, and once both checks pass, we recommend a product category based on their purchase history.

## Group Members

| Name | What they worked on |
|------|-------------------|
| Samuel Mwania | Task 1 & 2  — Merging the datasets and doing the EDA, Image Data Collection and Feature Extraction |
| Michael Kimani | Task 3 — Audio/sound data collection and processing |
| David Akachi | Task 4 — Training all three models |
| Kelvin Rwihimba | Task 6 — System demonstration and the pipeline script |

## How the pipeline works

The system follows three stages in sequence. If any stage fails, the user gets denied immediately.

```
User walks in
     |
     v
[Stage 1] Face Recognition
     |--- fails --> Access Denied
     |
     v (passes)
[Stage 2] Product Recommendation (runs in background)
     |
     v
[Stage 3] Voice Verification
     |--- fails --> Access Denied
     |
     v (passes)
Show welcome message + recommended product
```

We used three separate models for this:

| What it does | Input data | Algorithm |
|-------------|-----------|-----------|
| Facial recognition | Image histogram + intensity features | Random Forest |
| Voiceprint verification | MFCC coefficients, spectral rolloff, RMS energy | Random Forest |
| Product recommendation | Purchase history, engagement scores, sentiment | Random Forest |

## Folder structure

Here is what each folder contains:

```
data_preprocessing/
|
|-- pipeline.py                              <-- main demo script, run this
|-- task_6_system_demonstration (1).ipynb    <-- notebook version of the demo
|
|-- models/
|   |-- formaive2_data_preprocessing.ipynb   <-- Task 1: EDA and merging
|   |-- task_3_audio_Sound_Data_Collection_and_Processing.ipynb  <-- Task 3
|   |-- task_4_model_creation.ipynb          <-- Task 4: model training
|   |-- requirements.txt
|   |-- data/
|       |-- raw/          <-- original CSV files (social profiles + transactions)
|       |-- processed/    <-- merged dataset, image features, audio features
|       |-- images/       <-- facial images for each member
|
|-- saved_models/         <-- all the trained .pkl model files
|
|-- features/
|   |-- audio_features.csv
|
|-- .gitignore
|-- README.md
```

## Setting up the project

You need Python 3.8 or newer. We tested everything on 3.10.

1. Clone the repository:

```
git clone https://github.com/mwaniasam/data_preprocessing.git
cd data_preprocessing
```

2. Set up a virtual environment (optional but helps avoid version conflicts):

```
python -m venv .venv
.venv\Scripts\activate
```

On Mac or Linux that last line would be `source .venv/bin/activate` instead.

3. Install the packages:

```
pip install -r requirements.txt
```

There is also a requirements file inside the `models/` folder if you only want the dependencies for running the notebooks.

4. Check that everything loaded properly:

```
python pipeline.py --help
```

If you see the help text with the available options, you are good to go.

## Running the demonstration

The `pipeline.py` script handles the full demo from the command line. We set it up so you can either run everything at once or test individual scenarios.

**Full demo (recommended for presentation):**

```
python pipeline.py --mode demo
```

This goes through all four authorized members first, then runs several unauthorized scenarios where a person tries to use someone else's voice. The output is color-coded so it is easy to follow.

**Just one member:**

```
python pipeline.py --mode authorized --member Member_1
```

**All authorized members:**

```
python pipeline.py --mode all
```

**Testing unauthorized access (face and voice from different people):**

```
python pipeline.py --mode unauthorized --face Member_1 --voice Member_3
```

In an authorized run you should see all three stages pass and a product recommendation at the end. In an unauthorized run the voice stage will fail because the voice does not match the face, and the system blocks the transaction.

## Running the notebooks from scratch

If you want to rerun the entire pipeline from data to trained models:

1. Start Jupyter with `jupyter notebook`
2. Open and run the notebooks in this order:
   - `models/formaive2_data_preprocessing.ipynb` (merges the raw data, does EDA)
   - `models/task_3_audio_Sound_Data_Collection_and_Processing.ipynb` (processes audio samples)
   - `models/task_4_model_creation.ipynb` (trains all three models)
3. After step 3 completes, the `saved_models/` folder gets populated with the .pkl files
4. Then you can run `python pipeline.py --mode demo`

You do not need to rerun the notebooks just to do the demo since the trained models are already included in `saved_models/`.

## Common issues

If you get a `ModuleNotFoundError` — run `pip install -r requirements.txt` first.

If the pipeline says it cannot find model files — the `saved_models/` folder might be missing. Run the `task_4_model_creation.ipynb` notebook to regenerate them.

If you see a warning about synthetic audio — that just means `audio_features.csv` is not in the `features/` folder. The pipeline will fall back to synthetic features for the demo, but ideally you want the real file there.
