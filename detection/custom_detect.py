import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.custom_plots import plot_one_box, plot_line_betwen_boxes, plot_av_V, plot_av_C, plot_av_S
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import os
from os.path import exists

coef = 1.0
with open("coef.txt") as f:
        lines = f.readlines()
        coef = float(lines[0]) 


def save_to_file(write_log, im):
    file_exists = os.path.exists('log.txt')
    if file_exists:
        if check_count():
            with open('log.txt', 'a') as f:
                f.write('\n')
                f.write(write_log)
            replace_num()
        else:
            work_with_speed(im)
    else:
        with open('log.txt', 'w') as f:
            f.write('1')
            f.write('\n')
            f.write(write_log)

def replace_num():
    with open("log.txt") as f:
        lines = f.readlines()
        lines[0] = str(int(lines[0]) + 1) + '\n'
        print (lines)
    with open("log.txt", "w") as f:
        f.writelines(lines)

def check_count():
    with open("log.txt") as f:
        lines = f.readlines()
        if int(lines[0]) > 3:
            return False
        else:
            return True

        
def work_with_speed(im):
    global coef
    summ = 0
    with open("log.txt") as f:
        lines = f.readlines()
        summ += abs((int(lines[2]) - int(lines[1])) * coef)
        #y = abs(int(lines[2]) - int(lines[1])) * coef
        
        
        summ += abs((int(lines[3]) - int(lines[2])) * coef)
        
        summ += abs((int(lines[4]) - int(lines[3])) * coef)
        #print ('SPEED11 ' + str(summ))

    summ = float(summ / 3)
    fps = 30
    speed = summ * fps
    os.remove("log.txt")
    if speed < 5.0:
        plot_av_S(speed, im)
    else:
        print(speed)
            
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        #print(pred)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                pixels = []
                first_B = False
                for *xyxy, conf, cls in reversed(det):
                    if conf > 0.6:
                        y = xyxy.clone() if isinstance(xyxy, torch.Tensor) else np.copy(xyxy)
                        #print(y[0].item())
                        #print('BBBBBB')
                        pixels.append([int(y[0].item()),int(y[1].item()),int(y[2].item()),int(y[3].item()), conf, cls])
                        #if first_B == False:
                            
                           # first_B = True
                      
                plot_av_S(-1, im0)
                temp_p = pixels
                temp_p.sort(key = lambda x: x[0])
                for num in temp_p:
                    if num[4] > 0.6:
                        save_to_file(str(num[0]), im0)  
                        break
                
                colors_view = []
                for counter in pixels:
                    temp_color = []
                    a = int(((counter[3] - counter[1])/ 2) + counter[1])
                    b = int(((counter[2] - counter[0])/ 2) + counter[0])
                    print (a,b)
                    print (counter)
                    temp_color.append(im0[a][b])
                    #im0[a][b] = [255,0,0]
                    temp_color.append(im0[a-1][b])
                    #im0[a-1][b] = [255,0,0]
                    temp_color.append(im0[a+1][b])
                    #im0[a+1][b] = [255,0,0]
                    temp_color.append(im0[a][b-1])
                    #im0[a][b-1] = [255,0,0]
                    temp_color.append(im0[a][b+1])
                    #im0[a][b+1] = [255,0,0]
                    temp_color.append(im0[a-1][b+1])
                    #im0[a-1][b+1] = [255,0,0]
                    temp_color.append(im0[a-1][b-1])
                    #im0[a-1][b-1] = [255,0,0]
                    temp_color.append(im0[a+1][b+1])
                    #im0[a+1][b+1] = [255,0,0]
                    temp_color.append(im0[a+1][b-1])
                    #im0[a+1][b-1] = [255,0,0]
                    summa = 0
                    summb = 0
                    summc = 0
                    
                    for i in temp_color:
                        summa += i[0]
                        summb += i[1]
                        summc += i[2]
                    colors_view.append([int(summa/len(temp_color)), int(summb/len(temp_color)), int(summc/len(temp_color))])
                    
                print(colors_view)   
                
                global coef
                
                #test = pixels
                measures = []
                pixels.sort(key = lambda x: x[0])
                for measure in pixels:
                    if measure[5] == 1:
                        a = float(((measure[3] - measure[1])/ 2) * coef)
                        b = float(((measure[2] - measure[0])/ 2) * coef)      
                        c = a
                        V = 4/3*math.pi*a*b*c
                    elif measure[5] == 0:
                        r = float(((measure[3] - measure[1])/ 2) * coef)
                        V = 4/3*math.pi*r*r*r
                        
                    measure.append(V)
                    measures.append(V)
                    
                plot_av_V(measures, im0)
                plot_av_C(colors_view, im0)
                pixels = np.asarray(pixels)
                print (pixels)
                print ("Pix")
                #test.sort(key = lambda x: x[0])
                #test = np.asarray(test)
                #print (test)
                #print ("test")
                #print(pixels)
#                 bubles = []
#                 for i in range(0, len(pixels-1)):
#                     for j in range(i+1, len(pixels)):
#                         if abs(pixels[i][2] - pixels[j][0]) < 30:
#                             bubles.append(pixels[i])
#                             bubles.append(pixels[j])
#                             break
                #print(bubles)
                #bubles = np.asarray(bubles)
                #print(bubles)
                plot_line_betwen_boxes(pixels, im0, coef, line_thickness=3)
                
                
                
                
                    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #print(xyxy)
                    #print(det[0][0])
                    #print("A")
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        #print(line)
                        #print("YOLO YOLO YOLO YOLO YOLO YOLO")
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        #y = xyxy.clone() if isinstance(xyxy, torch.Tensor) else np.copy(xyxy)
                        #print (y)
                        #if  np.array_equal(np.asarray(xyxy), bubles[0]) or np.array_equal(np.asarray(xyxy), bubles[1]):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                            
                        #xyxy1 = 
                        #print(xyxy1 == bubles[0])
                        #print("YOLO YOLO YOLO YOLO YOLO YOLO")

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:            
                detect()
                strip_optimizer(opt.weights)
        else: 
            detect()
