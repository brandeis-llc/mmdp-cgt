import cv2
from ingest.file_path import VIDEO_PATH, FRAME_PATH
from cgqa.prepare_cg_input import get_evidences_for_cg_statement, utterance_to_input


def extract_frames_from_video(group_id, evi_id, utt_id, video_cap, start_time, end_time, num_frames):
    # Get frames per second of the video
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    duration = end_time - start_time
    # Calculate the interval (in seconds) between the frames
    interval = duration / (num_frames - 1)
    # Generate the list of timestamps at which frames will be captured
    timestamps = [start_time + i * interval for i in range(num_frames)]
    frame_index = 0
    images = []
    for i, timestamp in enumerate(timestamps):
        # Calculate the corresponding frame number
        frame_number = int(round(timestamp * fps))
        # Set the video position to the frame corresponding to the timestamp
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # Read the frame
        ret, frame = video_cap.read()
        # get the pixel height and width of the frame
        # height, width, _ = frame.shape
        # print(f"Frame {frame_index} at timestamp {timestamp}s (frame {frame_number}) has dimensions {width}x{height}.")
        if not ret:
            print(f"Warning: Could not read frame at timestamp {timestamp}s (frame {frame_number}).")
            continue
        images.append(frame)
        # Save the frame as an image file (Optional)
        # frame_file = FRAME_PATH / f"frame_at_{timestamp:.2f}s.jpg"
        frame_file = FRAME_PATH / str(group_id) / f"{evi_id}-{utt_id}-{i}.jpg"

        cv2.imwrite(frame_file, frame)
        frame_index += 1
    return images


if __name__ == '__main__':
    longest = []
    for group_id in range(1, 11):
        evidence, history = get_evidences_for_cg_statement(group_id, use_dp=False, text_only=True, use_new=False)
        # Open the video file
        video_path = VIDEO_PATH.joinpath(f"Group_{str(group_id).rjust(2, '0')}-master-audio.mp4")
        cap = cv2.VideoCapture(video_path)
        (FRAME_PATH / str(group_id)).mkdir(exist_ok=True)
        for e in evidence:
            # print(e, history[e].fbank)
            for u in evidence[e]:
                utt_id = u.id
                utt_start = u.start
                utt_end = u.end
                longest.append(utt_end - utt_start)
                extract_frames_from_video(group_id, e, utt_id, cap, utt_start, utt_end, 10)
            print(f"Extracted frames for evidence {e}, total {len(evidence[e])} utterances.")
        cap.release()
        # break
    # print(sorted(longest, reverse=True))
    # print(sum(longest) / len(longest))
