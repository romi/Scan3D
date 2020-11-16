#!/usr/bin/env python
# -*- coding: utf-8 -*-

MODULES = {
    # Scanning modules:
    "Scan": "romiscanner.tasks.scan",
    "VirtualPlant": "romiscanner.tasks.lpy",
    "VirtualScan": "romiscanner.tasks.scan",
    "CalibrationScan": "romiscanner.tasks.scan",
    # Geometric reconstruction modules:
    "Colmap": "romiscan.tasks.colmap",
    "Undistorted": "romiscan.tasks.proc2d",
    "Masks": "romiscan.tasks.proc2d",
    "Voxels": "romiscan.tasks.cl",
    "PointCloud": "romiscan.tasks.proc3d",
    "TriangleMesh": "romiscan.tasks.proc3d",
    "CurveSkeleton": "romiscan.tasks.proc3d",
    # Machine learning reconstruction modules:
    "Segmentation2D": "romiscan.tasks.proc2d",
    "SegmentedPointCloud": "romiscan.tasks.proc3d",
    "ClusteredMesh": "romiscan.tasks.proc3d",
    # Quantification modules:
    "TreeGraph": "romiscan.tasks.arabidopsis",
    "AnglesAndInternodes": "romiscan.tasks.arabidopsis",
    "CylinderRadius": "romiscan.tasks.calibration",
    # Visu modules:
    "Visualization": "romiscan.tasks.visualization",
    # Database modules:
    "Clean": "romidata.task"
}

TASKS = list(MODULES.keys())

EVAL_MODULES = {
    "VoxelGroundTruth": "romiscan.tasks.evaluation",
    "VoxelsEvaluation": "romiscan.tasks.evaluation",
    "PointCloudGroundTruth": "romiscan.tasks.evaluation",
    "PointCloudEvaluation": "romiscan.tasks.evaluation",
    "ClusteredMeshGroundTruth": "romiscan.tasks.evaluation",
    "PointCloudSegmentationEvaluation": "romiscan.tasks.evaluation",
    "Segmentation2DEvaluation": "romiscan.tasks.evaluation",
    "CylinderRadiusGroundTruth": "romiscan.tasks.evaluation",
    "CylinderRadiusEvaluation": "romiscan.tasks.evaluation"
}

EVAL_TASKS = list(EVAL_MODULES.keys())