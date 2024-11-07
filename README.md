# Dense Paraphrasing for Multimodal Dialogue Interpretation

## Source Data

Source files of the Weight Task Dataset, and processed data files for the experiments.

```
source-data
├── actions         # manually created annotations for actions
├── cga_originals   # Commonground Annotation 
├── cga_new         # Commonground Annotation (automatically added ACCEPTs for all STATEMEMTs, except those that were later DOUBTed)
├── oracle          # Manually segmented and transcribed utterances
├── dped-oracle     # Full DPed transcript sentences
├── frames          # Video frames
├── gamr            # GAMR annotation
└── videos          # Dialogue video
```

## Data Ingestion

Scripts for ingesting necessary data files (GAMR, Action, etc.) into Python data structures.
No need to run these scripts separately.

```
ingest
├── common_ground.py
├── data.py
├── file_path.py
├── generate_history_from_cga.py
├── ingest_action.py
├── ingest_cps.py
├── ingest_dialogue.py
└── ingest_gamr.py
```

## Scripts

Utility files

```
scripts
├── align_action_gamr_utterance.py      # Align action and GAMR with utterances
├── generate_full_dp_transcripts.py     # Generate full DPed transcript sentences in dped-oracle/
└── video_processing.py                 # Extract frames from video
```

## Common Ground Tracking Experiment

```
cgqa
├── prepare_cg_input.py     # Process data to GPT input format
├── mmdp_gpt.py             # CGT with unimodal textual GPT
├── mmdp_vision_gpt.py      # CGT with multimodal (text + images) GPT
├── output_postprocess.py   # Postprocess GPT output
└── eval_cgqa_output.py     # Evaluate GPT output in F1 and DSC
```

### Running Instructions

Edit the data and model setting in file as needed.

```
# Run the CGT with unimodal textual GPT
python -m cgqa.mmdp_gpt

# Or run the CGT with multimodal (text + images) GPT
python -m cgqa.mmdp_vision_gpt

# Postprocess the GPT output
python -m cgqa.output_postprocess

# Evaluate the GPT output in F1 and DSC
python -m cgqa.eval_cgqa_output
```

## Model Output

Post-processed output files from the Common Ground Tracking experiment under different settings.

```
cgqa_output
├── dp-gpt3.5       # DP-Decont. with GPT-3.5
├── dp-gpt4o        # DP-Decont. with GPT-4o
├── dp-mm-gpt3.5    # MMDP-Decont. with GPT-3.5
├── dp-mm-gpt4o     # MMDP-Decont. with GPT-4o
├── text-gpt3.5     # DP-Utt. with GPT-3.5
├── text-gpt4o      # DP-Utt. with GPT-4o
├── text-mm-gpt3.5  # MMDP-Utt. with GPT-3.5
├── text-mm-gpt4o   # MMDP-Utt. with GPT-4o
├── vision-4o       # Multimodal baseline with GPT-4o
└── vision-4o-mini  # Multimodal baseline with GPT-4o-mini
```