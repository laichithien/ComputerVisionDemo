import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent
import cv2
from pprint import pprint

def take_pic():

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "test.jpg"
            cv2.imwrite(img_name, frame)
            img_counter += 1
            break

    cam.release()

    cv2.destroyAllWindows()


def IoU(bbox1, bbox2):

    x1_left = bbox1[0]
    y1_top = bbox1[1]
    x1_right = bbox1[0] + bbox1[2]
    y1_bot = bbox1[1] + bbox1[3]

    x2_left = bbox2[0]
    y2_top = bbox2[1]
    x2_right = bbox2[0] + bbox2[2]
    y2_bot = bbox2[1] + bbox2[3]

    x_left = max(x1_left, x2_left)
    x_right = min(x1_right, x2_right)
    y_top = max(y1_top, y2_top)
    y_bot = min(y1_bot, y2_bot)

    inter = (x_right-x_left) * (y_bot - y_top)
    area1 = (x1_right-x1_left) * (y1_bot - y1_top)
    area2 = (x2_right-x2_left) * (y2_bot - y2_top)
    union = area1 + area2 - inter

    IoU = inter / union
    return IoU

async def main():
    config_file = './configs/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = './models/Faster_RCNN_epoch_24.pth'
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    take_pic()


    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # test a single image and show the results
    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once

    async with concurrent(streamqueue):
        result = await async_inference_detector(model, img)

    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='result.jpg')
    print(type(result))
    pprint(result[1]) # Mask weared incorrect 
    pprint(result[2]) # With mask
    pprint(result[3]) # Without mask
    
    

    total_people = 0
    incorrect = 0
    withmask = 0
    withoutmask = 0
    for i in result[1]:
        if i[4] >=0.4:
            total_people += 1
            pprint(total_people)
            pprint("In")
            incorrect += 1
    for i in result[2]:
        if i[4] >= 0.4:
            total_people += 1
            pprint(total_people)
            pprint("With")
            withmask += 1
    for i in result[3]:
        if i[4] >= 0.4:
            total_people += 1
            pprint(total_people)
            pprint("Without")
            withoutmask += 1
    pprint("Tong so nguoi: " + str(total_people))
    pprint("Ti le so nguoi deo khau trang khong dung cach: "+ str(incorrect / total_people))
    pprint("Ti le so nguoi deo khau trang: "+ str(withmask / total_people))
    pprint("Ti le so nguoi khong deo khau trang : "+ str(withoutmask / total_people))
    window_name = 'result'
    cv2.imshow(window_name, cv2.imread('result.jpg'))
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

asyncio.run(main())
