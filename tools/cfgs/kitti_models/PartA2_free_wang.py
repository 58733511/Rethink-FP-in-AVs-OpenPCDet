CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']

DATA_CONFIG:
    _BASE_CONFIG_: /home/leichen/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml


MODEL:
    NAME: PointRCNN

    VFE:
        NAME: MeanVFE

    BACKBONE_3D:
        NAME: UNetV2
        RETURN_ENCODED_TENSOR: False

    POINT_HEAD:
        NAME: PointIntraPartOffsetHead
        CLS_FC: [128, 128]
        PART_FC: [128, 128]
        REG_FC: [128, 128]
        CLASS_AGNOSTIC: False
        USE_POINT_FEATURES_BEFORE_FUSION: False
        TARGET_CONFIG:
            GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
            BOX_CODER: PointResidualCoder
            BOX_CODER_CONFIG: {
                'use_mean_size': True,
                'mean_size': [
                    [3.9, 1.6, 1.56],
                    [0.8, 0.6, 1.73],
                    [1.76, 0.6, 1.73]
                ]
            }

        LOSS_CONFIG:
            LOSS_REG: WeightedSmoothL1Loss
            LOSS_WEIGHTS: {
                'point_cls_weight': 1.0,
                'point_box_weight': 1.0,
                'point_part_weight': 1.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

    ROI_HEAD:
        NAME: PVRCNNHead
        CLASS_AGNOSTIC: True

        SHARED_FC: [256, 256]
        CLS_FC: [256, 256]
        REG_FC: [256, 256]
        DP_RATIO: 0.3

        NMS_CONFIG:
            TRAIN:
                NMS_TYPE: nms_gpu
                MULTI_CLASSES_NMS: False
                NMS_PRE_MAXSIZE: 9000
                NMS_POST_MAXSIZE: 512
                NMS_THRESH: 0.8
            TEST:
                NMS_TYPE: nms_gpu
                MULTI_CLASSES_NMS: False
                NMS_PRE_MAXSIZE: 1024
                NMS_POST_MAXSIZE: 100
                NMS_THRESH: 0.7

        ROI_GRID_POOL:
            GRID_SIZE: 6
            MLPS: [[64, 64], [64, 64]]
            POOL_RADIUS: [0.8, 1.6]
            NSAMPLE: [16, 16]
            POOL_METHOD: max_pool
            
        ROI_GRID_POOL_ADD:
            GRID_SIZE: 6
            MLPS: [[16, 16], [16, 16]]
            POOL_RADIUS: [0.4, 0.8]
            NSAMPLE: [16, 16]
            POOL_METHOD: max_pool
            
        TARGET_CONFIG:
            BOX_CODER: ResidualCoder
            ROI_PER_IMAGE: 128
            FG_RATIO: 0.5

            SAMPLE_ROI_BY_EACH_CLASS: True
            CLS_SCORE_TYPE: roi_iou

            CLS_FG_THRESH: 0.75
            CLS_BG_THRESH: 0.25
            CLS_BG_THRESH_LO: 0.1
            HARD_BG_RATIO: 0.8

            REG_FG_THRESH: 0.65

        LOSS_CONFIG:
            CLS_LOSS: BinaryCrossEntropy
            REG_LOSS: smooth-l1
            CORNER_LOSS_REGULARIZATION: True
            LOSS_WEIGHTS: {
                'rcnn_cls_weight': 1.0,
                'rcnn_reg_weight': 1.0,
                'rcnn_corner_weight': 1.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        SCORE_THRESH: 0.1
        OUTPUT_RAW_SCORE: False

        EVAL_METRIC: kitti

        NMS_CONFIG:
            MULTI_CLASSES_NMS: False
            NMS_TYPE: nms_gpu
            NMS_THRESH: 0.1
            NMS_PRE_MAXSIZE: 4096
            NMS_POST_MAXSIZE: 500


OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 4
    NUM_EPOCHS: 80

    OPTIMIZER: adam_onecycle
    LR: 0.003
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9

    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001

    LR_WARMUP: False
    WARMUP_EPOCH: 1

    GRAD_NORM_CLIP: 10
