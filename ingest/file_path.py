from pathlib import Path

DATA_FOLDER = Path(__file__).parent.parent.joinpath("source-data")
ORACLE_PATH = DATA_FOLDER.joinpath("oracle")
DP_PATH = DATA_FOLDER.joinpath("dped-transcripts")
DP_ORACLE_PATH = DATA_FOLDER.joinpath("dped-oracle")
CPS_CSV_PATH = DATA_FOLDER / "cps" / "All_Groups_CPS.csv"
ACTION_PATH = DATA_FOLDER.joinpath("actions")
GAMR_PATH = DATA_FOLDER.joinpath("gamr")
CGA_NEW_PATH = DATA_FOLDER.joinpath("cga_new")
CGA_ORI_PATH = DATA_FOLDER.joinpath("cga_originals")
CGA_OUTPUT_PATH = DATA_FOLDER.joinpath("cgqa_output")
VIDEO_PATH = DATA_FOLDER.joinpath("videos")
FRAME_PATH = DATA_FOLDER.joinpath("video_frames")
