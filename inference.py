import torch
import cv2
import os
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_jpeg, read_image, VideoReader, write_video
from metrics import calculate_mse, calculate_mAP, calculate_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import PrecisionRecallCurve
from torchdata.datapipes.iter import IterableWrapper
import numpy as np

# tentatively maybe not use this?
from datasets2 import (
    get_model_instance_segmentation,
    collate_fn,
    get_transform,
    Dataset
)

# constants

MODEL_PATH = 'option_model_best.pth'
IMAGES_PATH = 'custom_data/valid/'
CONFIDENCE_THRESHOLD = 0.8
IOU_THRESHOLDS = [0.5, 0.8, 0.90, 0.99]
REC_THRESHOLDS = [0.5, 0.8, 0.90, 0.99]
BATCH_LOAD_SIZE = 2
MODEL_NUM_CLASSES = 5
CALCULATE_STATISTICS = False
VIDEO_PATH = '/data/crusher_bin_bridge2.mkv'
VIDEO_PATH_2 = '/app/video/(10.114.237.108) - TV401C PC1 ROM Bin-2024.04.16-04.00.00-15m00s.mkv'
VIDEO_PATH_3 = '/app/video/TV401C PC1 ROM Bin_urn-uuid-00075fbe-43fb-fb43-be5f-0700075fbe5f_2024-01-27_20-20-00.mp4'
OUTPUT_VIDEO_PATH = '/app/inference/output.mkv'


def object_detection(model, image_tensor, confidence_threshold=CONFIDENCE_THRESHOLD):
    with torch.no_grad():
        y_pred = model([image_tensor])
        bbox, scores, labels = y_pred[0]['boxes'], y_pred[0]['scores'], y_pred[0]['labels']
        indices = torch.nonzero(scores > confidence_threshold).squeeze(1)

        return bbox[indices], scores[indices], labels[indices]


def draw_boxes_and_labels(image, bbox, labels, class_names, scores):
    img_copy = image.copy()

    for i in range(len(bbox)):
        x, y, w, h = bbox[i].cpu().numpy().astype('int')
        cv2.rectangle(img_copy, (x, y), (w, h), (0, 0, 255), 5)

        class_index = labels[i].cpu().numpy().astype('int')
        class_detected = f"{class_names[class_index - 1]}: {scores[i]}"

        cv2.putText(img_copy, class_detected, (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    return img_copy


def load_data(device, image_path_iterator, batch_size=BATCH_LOAD_SIZE):
    """
    put a batch of images as torch tensors on torch device, return
    cpu bound opencv image for human review
    """
    images = []
    tfmr = get_transform()
    for i, item in enumerate(image_path_iterator):
        if i >= batch_size:
            break
        else:
            cv2i = read_image(item)
            cudai = convert_image_dtype(cv2i, torch.float32).to(device)

            images.append(cudai)

    return images


def get_image_iterator(image_folder_path: str):
    """
    generator of images to perform inference once there are no more images the
    generator will not wait for more
    """
    with os.scandir(image_folder_path) as iter:
        for item in iter:
            if item.name.endswith('.jpg') and item.is_file():
                yield item.path


# may not need this if mAP and other statistics accept iou thresholds
def confidence_threshold_filter(y_pred, confidence_threshold=CONFIDENCE_THRESHOLD):
    threshold = []
    for i_pred in y_pred:
        indices = torch.nonzero(i_pred['scores'] > confidence_threshold).squeeze(1)
        threshold.append(indices)

    return [{k: v[threshold[i]] for k, v in x.items()} for i, x in enumerate(y_pred)]


def get_cv2_videowriter(in_filename, out_filename):
    cap = cv2.VideoCapture(in_filename)

    print(cv2.VideoWriter_fourcc(*'H264'),
          cap.get(cv2.CAP_PROP_FPS),
          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), sep='\n')

    vw = cv2.VideoWriter(out_filename,
                         cv2.VideoWriter_fourcc(*'H264'),
                         cap.get(cv2.CAP_PROP_FPS),
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    return vw


def video_run(video_file, model, device, confidence_threshold=CONFIDENCE_THRESHOLD):
    # frames, audio, meta, we don't care about audio and meta at this point
    #    frames, _, meta =
    writer = get_cv2_videowriter(str(video_file), OUTPUT_VIDEO_PATH)
    # setup torch video reader
    reader = VideoReader(str(video_file))
    reader.set_current_stream('video')
    # setup cv2 videowriter
    #  print(reader.get_metadata())
    dp = IterableWrapper(reader)
    fbatcher = dp.batch(batch_size=BATCH_LOAD_SIZE)
    tfmr = get_transform()
    for i, frames in enumerate(fbatcher):
        # print("Loading frames to gpu")
        imgs = [convert_image_dtype(f['data'], torch.float32).to(device) for f in frames]
        y_pred = confidence_threshold_filter(model(imgs), confidence_threshold)

        write_video_frames(imgs, y_pred, writer)


def write_video_frames(imgs, preds, writer):
    labels = {0: 'rocks', 1: 'Potential Blockage', 2: 'blockage', 3: 'roi_b', 4: 'roi_t'}
    colors = {0: 'green', 1: 'yellow', 2: 'red', 3: 'magenta', 4: 'cyan'}
    video_array = []

    for i, img in enumerate(imgs):
        i_lbl_names = [str(labels[int(x)]) for x in preds[i]['labels']]
        i_cols = [str(colors[int(x)]) for x in preds[i]['labels']]
        i2 = draw_bounding_boxes(convert_image_dtype(img, torch.uint8),
                                 preds[i]['boxes'],
                                 labels=i_lbl_names,
                                 colors=i_cols)
        # bring the image back to cpu land
        cpu = i2.to('cpu').numpy(force=True)
        cv2_image = np.transpose(cpu, (1, 2, 0))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        writer.write(cv2_image)


#    write_video('/app/inference/output_video.mkv',tensor,12.0, video_codec='h264')


def eval_run(model, data_loader, device):
    # for each data_loader batch
    mses = []  # for iscrowd label?

    mAP = MeanAveragePrecision(iou_type="bbox", iou_thresholds=IOU_THRESHOLDS, rec_thresholds=REC_THRESHOLDS)
    mAP.to(device)

    prC = PrecisionRecallCurve(thresholds=IOU_THRESHOLDS, task='binary')
    prC.to(device)
    for i, (imgs, annotations) in enumerate(data_loader):
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # get predictions compared to labeled data
        # y_pred = confidence_threshold_filter(model(imgs), CONFIDENCE_THRESHOLD)
        y_pred = model(imgs)
        # print(f"Annotations {i}")
        # print(*[x for x in annotations], sep="\n")
        # print(f"Prediction {i}")
        # print(*[x for x in y_pred], sep="\n")

        # statistic
        mAP.update(y_pred, annotations)
        for i in range(len(y_pred)):
            p_len = len(y_pred[i]['scores'])
            a_len = len(annotations[i]['labels'])

            if (p_len > a_len):
                zeroes = torch.zeros((p_len - a_len), dtype=torch.int32).to(device)
                print((annotations[i]['labels'], zeroes))
                zero_padded = torch.cat((annotations[i]['labels'], zeroes), 0)
                prC.update(y_pred[i]['scores'], zero_padded)
            else:
                prC.update(y_pred[i]['scores'], annotations[i]['labels'])

        # create batch visualisations

    # res = mAP.compute()
    fig, ax = prC.plot()
    # fig.set_title(f"Precision recall curve mar_medium={res['mar_medium']}")
    fig.savefig('/app/inference/inference_mAP_plot.png')
    print(mAP.compute())


if __name__ == "__main__":
    eval_coco = "custom_data/valid/_annotations.coco.json"
    eval_dir = "custom_data/valid"

    dataset = Dataset(
        root=eval_dir, annotation=eval_coco, transforms=get_transform()
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_LOAD_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # prepare pre-trained model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(MODEL_NUM_CLASSES)
    trained_state = torch.load(MODEL_PATH)
    model.load_state_dict(trained_state)  # optionally turn strict off
    model.eval()
    model.to(device)
    torch.no_grad()

    # path_iter = get_image_iterator(IMAGES_PATH)

    try:
        os.mkdir('/app/inference')
    except:
        pass

    # eval_run(model, data_loader, device)
    print("Start video run")
    video_run(VIDEO_PATH, model, device, confidence_threshold=CONFIDENCE_THRESHOLD)
