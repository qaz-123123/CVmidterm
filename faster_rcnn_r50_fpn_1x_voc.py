_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


dataset_type = 'VOCDataset'
data_root = 'D:/pycharm/Pycharm 2024.1/1/1/pythonProject1/CVmidterm2/VOCdevkit/'

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
            loss_bbox=dict(type='L2Loss', loss_weight=0.5))
    )
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2012/'
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2012/'
    )
)

runner = dict(type='EpochBasedRunner', max_epochs=200)
