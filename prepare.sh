mv guangshan/ ./YOLO2COCO/dataset/YOLOV5/
cd YOLO2COCO/
mv dataset/YOLOV5/guangshan/images/ dataset/YOLOV5/guangshan/images_temp
mv dataset/YOLOV5/guangshan/labels/ dataset/YOLOV5/guangshan/labels_temp
mkdir dataset/YOLOV5/guangshan/images
cp -r dataset/YOLOV5/guangshan/images_temp/train/* dataset/YOLOV5/guangshan/images
ls -R dataset/YOLOV5/guangshan/images/* > dataset/YOLOV5/train.txt
rm -rf dataset/YOLOV5/guangshan/images/*
cp -r dataset/YOLOV5/guangshan/images_temp/val/* dataset/YOLOV5/guangshan/images
ls -R dataset/YOLOV5/guangshan/images/* > dataset/YOLOV5/val.txt
cp -r dataset/YOLOV5/guangshan/images_temp/train/* dataset/YOLOV5/guangshan/images
mkdir dataset/YOLOV5/guangshan/labels
cp -r dataset/YOLOV5/guangshan/labels_temp/train/* dataset/YOLOV5/guangshan/labels
cp -r dataset/YOLOV5/guangshan/labels_temp/val/* dataset/YOLOV5/guangshan/labels
rm -rf dataset/YOLOV5/guangshan/labels_temp
rm -rf dataset/YOLOV5/guangshan/images_temp
python yolov5_2_coco.py --dir_path dataset/YOLOV5
