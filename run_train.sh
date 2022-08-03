# python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-custom.yaml --weights 'pretrained/yolov7-w6_training.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml

python train.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 512 512 --cfg cfg/training/yolov7x.yaml --weights 'pretrained/yolov7x_training.pt' --name yolov7x_4classes_watermark --hyp data/hyp.scratch.custom.yaml

