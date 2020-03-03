import luigi
import numpy as np
import tempfile
import os
from scipy.ndimage.filters import gaussian_filter

from romidata import io
from romidata.task import FilesetExists, RomiTask, FilesetTarget, DatabaseConfig, ImagesFilesetExists

from romiscan.log import logger
from romiscan.tasks import proc2d, proc3d
from romiscanner.lpy import VirtualPlant
from romiscan.tasks.cl import Voxels

class EvaluationTask(RomiTask):
    upstream_task = luigi.TaskParameter()
    ground_truth = luigi.TaskParameter()

    def requires(self):
        return [self.upstream_task(), self.ground_truth()]

    def output(self):
        fileset_id = self.task_family #self.upstream_task().task_id + "Evaluation"
        return FilesetTarget(DatabaseConfig().scan, fileset_id)


    def evaluate(self):
        raise NotImplementedError

    def run(self):
        res = self.evaluate()
        io.write_json(self.output_file(), res)

class VoxelGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlant)

    def run(self):
        import pywavefront
        import open3d
        import trimesh
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"), collect_faces=True, create_materials=True)
            res = {}
            min= np.min(x.vertices, axis=0)
            max = np.max(x.vertices, axis=0)
            arr_size = np.asarray((max-min) / Voxels().voxel_size + 1, dtype=np.int)+ 1
            for k in x.meshes.keys():
                t = open3d.geometry.TriangleMesh()
                t.triangles = open3d.utility.Vector3iVector(np.asarray(x.meshes[k].faces))
                t.vertices = open3d.utility.Vector3dVector(np.asarray(x.vertices))
                t.compute_triangle_normals()
                open3d.io.write_triangle_mesh(os.path.join(tmpdir, "tmp.stl"), t)
                m = trimesh.load(os.path.join(tmpdir, "tmp.stl"))
                v = m.voxelized(Voxels().voxel_size)
                

                class_name = x.meshes[k].materials[0].name
                arr = np.zeros(arr_size)
                voxel_size = Voxels().voxel_size
                origin_idx = np.asarray((v.origin - min) / voxel_size, dtype=np.int) 
                arr[origin_idx[0]:origin_idx[0] + v.matrix.shape[0],origin_idx[1]:origin_idx[1] + v.matrix.shape[1], origin_idx[2]:origin_idx[2] + v.matrix.shape[2]] = v.matrix
                res[class_name] = gaussian_filter(arr, voxel_size)
                
            # The following is needed because there are rotation in lpy's output...
            arr = np.swapaxes(arr, 2,1)
            arr = np.flip(arr, 1)
            bg = np.ones((arr_size))
            for k in res.keys():
                bg = np.minimum(bg, 1-res[k])
            res["background"] = bg
            io.write_npz(self.output_file(), res)
              
class PointCloudGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlant)
    pcd_size = luigi.IntParameter(default=100000)

    def run(self):
        import pywavefront
        import open3d
        import trimesh
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"), collect_faces=True, create_materials=True)
            res = open3d.geometry.PointCloud()
            point_labels = []

            for i, k in enumerate(x.meshes.keys()):
                t = open3d.geometry.TriangleMesh()
                t.triangles = open3d.utility.Vector3iVector(np.asarray(x.meshes[k].faces))
                t.vertices = open3d.utility.Vector3dVector(np.asarray(x.vertices))
                t.compute_triangle_normals()

                # The following is needed because there are rotation in lpy's output...
                pcd = t.sample_points_poisson_disk(self.pcd_size)
                pcd_pts = np.asarray(pcd.points)
                pcd_pts = pcd_pts[:,[0,2,1]]
                pcd_pts[:, 1] *= -1
                pcd.points = open3d.utility.Vector3dVector(pcd_pts)

                res = res + pcd
                class_name = x.meshes[k].materials[0].name
                point_labels += [k] * len(pcd.points)

            io.write_point_cloud(self.output_file(), res)
            self.output_file().set_metadata({'labels' : point_labels})        

class PointCloudEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc3d.PointCloud)
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)
    max_distance = luigi.FloatParameter(default=2)

    def evaluate(self):
        import open3d
        source = io.read_point_cloud(self.upstream_task().output_file())
        target = io.read_point_cloud(self.ground_truth().output_file())
        labels = self.upstream_task().output_file().get_metadata('labels')
        labels_gt = self.ground_truth().output_file().get_metadata('labels')
        if labels is not None:
            eval = { "id": self.upstream_task().task_id }
            for l in set(labels_gt):
                idx = [i for i in range(len(labels)) if labels[i] == l]
                idx_gt = [i for i in range(len(labels_gt)) if labels_gt[i] == l]

                subpcd_source = open3d.geometry.PointCloud()
                subpcd_source.points = open3d.utility.Vector3dVector(np.asarray(source.points)[idx])
                subpcd_target = open3d.geometry.PointCloud()
                subpcd_target.points = open3d.utility.Vector3dVector(np.asarray(target.points)[idx_gt])

                if len(subpcd_target.points) == 0:
                    continue

                logger.debug("label : %s"%l)
                logger.debug("gt points: %i"%len(subpcd_target.points))
                logger.debug("pcd points: %i"%len(subpcd_source.points))
                res = open3d.registration.evaluate_registration(subpcd_source, subpcd_target, self.max_distance)
                eval[l] = {
                    "fitness": res.fitness,
                    "inlier_rmse": res.inlier_rmse
                }
        res = open3d.registration.evaluate_registration(source, target, self.max_distance)
        eval["all"] = {
            "fitness": res.fitness,
            "inlier_rmse": res.inlier_rmse
        }
        return eval

class Segmentation2DEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default = proc2d.Segmentation2D)
    ground_truth = luigi.TaskParameter(default = ImagesFilesetExists)
    hist_bins = luigi.IntParameter(default = 100)

    def evaluate(self):
        
        prediction_fileset = self.upstream_task().output().get()
        gt_fileset = self.ground_truth().output().get()
        channels = gt_fileset.get_metadata('channels')
        channels.remove('rgb')

        
        histograms = {}

        for c in channels:

            logger.debug(prediction_fileset)
            preds = prediction_fileset.get_files(query = {'channel' : c})
            logger.debug(preds)
            hist_high, disc = np.histogram(np.array([]), self.hist_bins, range=(0,1))
            hist_low, disc = np.histogram(np.array([]), self.hist_bins, range=(0,1))

            
            for pred in preds:
                        ground_truth = gt_fileset.get_files(query = {'channel': c, 'shot_id': pred.get_metadata('shot_id')})[0]
                        im_pred = io.read_image(pred)
                        im_gt = io.read_image(ground_truth)
                
                        
                        im_gt_high = im_pred[im_gt > 255//2]/255



                        im_gt_low = im_pred[im_gt < 255//2]/255	


            hist_high_pred, bins_high = np.histogram(im_gt_high, self.hist_bins, range=(0,1))
            hist_low_pred, bins_low = np.histogram(im_gt_low, self.hist_bins, range=(0,1))
            hist_high += hist_high_pred
            hist_low += hist_low_pred
            histograms[c] = {"hist_high": hist_high.tolist(), "bins_high": bins_high.tolist(), "hist_low": hist_low.tolist(), "bins_low": bins_low.tolist()}
        return histograms

class VoxelsEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default = Voxels)
    ground_truth = luigi.TaskParameter(default = VoxelGroundTruth)
    hist_bins = luigi.IntParameter(default = 100)

    def requires(self):
        return [self.upstream_task(), self.ground_truth()]

    def evaluate(self):
        prediction_file = self.upstream_task().output().get().get_files()[0]
        gt_file = self.ground_truth().output().get().get_files()[0]

        predictions = io.read_npz(prediction_file)
        gts = io.read_npz(gt_file)

        channels = list(gts.keys())
        histograms = {}
        from matplotlib import pyplot as plt

        for c in channels:

            prediction_c = predictions[c]
            gt_c = gts[c]


            gt_c = gt_c[0:prediction_c.shape[0],0:prediction_c.shape[1] ,0:prediction_c.shape[2]]
            im_gt_high = prediction_c[gt_c > 0.5]
            im_gt_low = prediction_c[gt_c < 0.5]


            hist_high, bins_high = np.histogram(im_gt_high, self.hist_bins)
            hist_low, bins_low = np.histogram(im_gt_low, self.hist_bins)
            plt.figure()
            plt.plot(bins_high[:-1], hist_high)
            plt.savefig("high%s.png"%c)

            plt.figure()
            plt.plot(bins_low[:-1], hist_low)
            plt.savefig("low%s.png"%c)

            plt.imshow(gt_c.max(0))
            plt.savefig("gt%s.png"%c)

            plt.imshow(prediction_c.max(0))
            plt.savefig("prediction%s.png"%c)

            histograms[c] = {"hist_high": hist_high.tolist(), "bins_high": bins_high.tolist(), "hist_low": hist_low.tolist(), "bins_low": bins_low.tolist()}
        return histograms


