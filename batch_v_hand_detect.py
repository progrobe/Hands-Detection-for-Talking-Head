import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import cv2
import numpy as np
from tqdm import tqdm
import json
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--video_dir', type=str, default='example')
parser.add_argument('-o', '--out_video_dir', type=str, default='output')
parser.add_argument('-m', '--model_path', type=str, default='weights')
parser.add_argument('--cut', action='store_true')
parser.add_argument('--min_frame_num', type=int, default=15)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--box_threshold', type=float, default=0.2)
parser.add_argument('--text_threshold', type=float, default=0.1)
parser.add_argument('--skip', type=int, default=3)
parser.add_argument('--demo', action='store_true')
parser.add_argument('-d', '--demo_dir', type=str, default='demo')
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()
video_dir = args.video_dir
out_video_dir = args.out_video_dir
print(f'processing {video_dir} to {out_video_dir}')

# model_id = "IDEA-Research/grounding-dino-base"
model_path = args.model_path
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device)
print('model loaded')

video_list = [video_name for video_name in os.listdir(video_dir) if video_name.endswith('.mp4')]

for video_name in video_list:
    print(f'processing {video_name}')
    video_path = os.path.join(video_dir, video_name)
    out_subclip_dir = os.path.join(out_video_dir, video_name.replace('.mp4', '/'))
    os.makedirs(out_subclip_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if args.demo:
        out_demo_path = os.path.join(args.demo_dir, video_name)
        os.makedirs(args.demo_dir, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_demo_path, fourcc, fps/args.skip, (width, height))

    # VERY important: text queries need to be lowercased + end with a dot
    text = "hand. fingers. finger. thumb."
    text = text.lower()

    # frame_count = 200

    bs = args.batch_size
    skip = args.skip # apply detection every 3 frames to save time
    hand_frames = []
    frame_idx = -1
    for i in tqdm(range(0,frame_count//bs//skip)):
        image_list = []
        frame_list = []
        for j in range(bs*skip): # for batch inference
            ret, frame = cap.read()
            if not ret:
                print(f'error in frame {frame_idx}')
                break
            frame_idx += 1
            frame_list.append(frame_idx)
            if frame_idx % skip != 0:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_list.append(image)
        text_list = [text] * len(image_list)

        inputs = processor(images=image_list, text=text_list, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1] for image in image_list],
        )
        for j in range(len(results)):
            result = results[j]
            image = image_list[j]
            is_hand = False
            for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
                is_hand = True
                if args.verbose:
                    box = [round(x, 2) for x in box.tolist()]
                    print(f"Detected {labels} in frame {frame_list[j*skip]} with confidence {round(score.item(), 3)} at location {box}")
            if is_hand:
                hand_frames.append(frame_list[j*skip])

            if args.demo:
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                for box in result["boxes"]:
                    box = [round(x, 2) for x in box.tolist()]
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                out.write(image)
    print('detection done')

    if args.verbose:
        print(f'hands frame idexs:{hand_frames}')

    min_clip = args.min_frame_num
    no_hand_frames = []
    start_idx = 0
    for frame_idx in tqdm(range(frame_count)):
        if frame_idx not in hand_frames:
            if frame_idx == frame_count - 1:
                end_idx = frame_idx - args.skip
                if end_idx - start_idx > min_clip:
                    no_hand_frames.append([start_idx, end_idx])
            else:
                continue
        else:
            end_idx = frame_idx - 1 - args.skip
            if end_idx - start_idx > min_clip:
                no_hand_frames.append([start_idx, end_idx])
            start_idx = frame_idx + args.skip

    if args.verbose:
        no_hand_frame_num = 0
        for no_hand_frame in no_hand_frames:
            no_hand_frame_num += no_hand_frame[1] - no_hand_frame[0] + 1
        no_hand_ratio = no_hand_frame_num / frame_count *100//1/100
        print(f'no hand frames:{no_hand_frames}')
        print(f'no hand ratio:{no_hand_ratio}')

    if args.cut:
        print('cutting video')
        for start_idx, end_idx in no_hand_frames:
            subclip_path = os.path.join(out_subclip_dir, f'{start_idx}_{end_idx}.mp4')
            cmd = f'ffmpeg -i {video_path} -ss {start_idx/fps} -to {end_idx/fps} -crf 14 -preset veryslow -c:a aac {subclip_path} -y -loglevel error'
            os.system(cmd)
        print('done')
