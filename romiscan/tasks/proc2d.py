import luigi
import numpy as np
from romidata import io
from romidata.task import FileByFileTask
from romidata.task import FilesetExists
from romidata.task import ImagesFilesetExists
from romiscan.log import logger
from romiscan.proc2d import vesselness
from romiscan.tasks.colmap import Colmap
from skimage.exposure import rescale_intensity


class Undistorted(FileByFileTask):
    """ Undistorts images using computed intrinsic camera parameters

    Module: romiscan.tasks.proc2d
    Default upstream tasks: Scan, Colmap
    Upstream task format: Fileset with image files
    Output fileset format: Fileset with image files

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)

    def input(self):
        return self.upstream_task().output()

    def requires(self):
        # return [Colmap(), Scan(), self.upstream_task()]
        return [Colmap(), self.upstream_task()]

    def f(self, fi, outfs):
        from romiscan import proc2d
        camera = fi.get_metadata('colmap_camera')
        if camera is not None:
            camera_model = camera['camera_model']
            x = io.read_image(fi)
            x = proc2d.undistort(x, camera_model)
            outfi = outfs.create_file(fi.id)
            io.write_image(outfi, x)
            return outfi


class Masks(FileByFileTask):
    """ Compute masks using several functions

    Module: romiscan.tasks.proc2d
    Default upstream tasks: Undistorted
    Upstream task format: Fileset with image files
    Output fileset format: Fileset with grayscale or binary image files
    
    Parameters
    ----------
    type : luigi.Parameter
        "linear", "excess_green", "vesselness", "invert" (TODO: see segmentation documentation)
    parameters : luigi.ListParameter
        list of scalar parmeters, depends on type
    dilation : luigi.IntParameter
        by how much to dilate masks if binary
    binarize : luigi.BoolParameter, optional
        binarize the masks, default=True
    threshold : luigi.FloatParameter, optional
        threshold for binarization, default=0.0

    """
    upstream_task = luigi.TaskParameter(default=Undistorted)

    type = luigi.Parameter("linear")
    parameters = luigi.ListParameter(default=[0,1,0])
    logger.debug(f"Parameters: {parameters}")
    dilation = luigi.IntParameter(default=0)

    binarize = luigi.BoolParameter(default=True)
    threshold = luigi.FloatParameter(default=0.3)

    def f_raw(self, x):
        from romiscan import proc2d
        x = np.asarray(x, dtype=np.float)
        logger.debug(f"x shape: {x.shape}")
        x = rescale_intensity(x, out_range=(0, 1))
        logger.debug(f"x shape: {x.shape}")
        if self.type == "linear":
            coefs = self.parameters
            return (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                    coefs[2] * x[:, :, 2])
        elif self.type == "excess_green":
            return proc2d.excess_green(x)
        elif self.type == "vesselness":
            scale = self.parameters['scale']
            channel = self.parameters['channel']
            return vesselness(x, scale, channel=channel)
        elif self.type == "invert":
            return 1 - x
        else:
            raise Exception("Unknown masking type")

    def f(self, fi, outfs):
        from romiscan import proc2d
        logger.debug(f"Loading file: {fi.filename}")
        x = io.read_image(fi)
        x = self.f_raw(x)
        if self.binarize:
            x = x > self.threshold
            if self.dilation > 0:
                x = proc2d.dilation(x, self.dilation)
        else:
            x[x < self.threshold] = 0
            x = rescale_intensity(x, out_range=(0, 1))
        x = np.array(255 * x, dtype=np.uint8)

        outfi = outfs.create_file(fi.id)
        io.write_image(outfi, x)
        return outfi


class ModelFileset(FilesetExists):
    # scan_id = luigi.Parameter()
    fileset_id = "models"

class Segmentation2D(Masks):
    """ Compute masks using trained deep learning models.

    Module: romiscan.tasks.proc2d
    Description: compute masks using trained deep learning models
    Default upstream tasks: Undistorted
    Upstream task format: Fileset with image files
    Output fileset format: Fileset with grayscale image files, each corresponding to a given input image and class

    Parameters
    ----------
    query : DictParameter
        query to pass to upstream fileset. It filters file by metadata, e.g
        {"channel": "rgb"} will process only input files such that "channel"
        metadata is equal to "rgb".
    labels : Parameter
        string of the form "a,b,c" such that a, b, c are the identifiers of the
        labels produced by the neural network
    Sx, Sy : IntParametr
        size of the input of the neural network.
        Input pictures are cropped in the center to this size.
    model_segmentation_name : str??
        name of ".pt" file. Can be found at `https://db.romi-project.eu/models`
    lighter_gpu_computation: luigi.BoolParameter
        if true, gives the files one by one to the segmentation (for GPU memory)

    """
    type = None
    parameters = None

    upstream_task = luigi.TaskParameter(default=Undistorted)
    model_fileset = luigi.TaskParameter(default=ModelFileset)

    model_id = luigi.Parameter()
    query = luigi.DictParameter(default={})

    Sx = luigi.IntParameter(default=896)
    Sy = luigi.IntParameter(default=896)

    labels = luigi.ListParameter(default=[])
    inverted_labels = luigi.ListParameter(default=["background"])

    resize = luigi.BoolParameter(default=False)

    lighter_gpu_computation = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            "images": self.upstream_task(),
            "model": self.model_fileset()
        }

    def run(self):
        from romiseg.Segmentation2D import segmentation
        from romiscan import proc2d

        images_fileset = self.input()["images"].get().get_files(
            query=self.query)
        model_file = self.input()["model"].get().get_file(self.model_id)
        if model_file is None:
            raise IOError("unable to find model: %s" % self.model_id)

        labels = model_file.get_metadata("label_names")
        if len(self.labels) > 0:
            label_range = [labels.index(x) for x in self.labels]
        else:
            label_range = range(len(labels))

        output_fileset = self.output().get()

        if self.lighter_gpu_computation:
            # ugly stuff, apply segmentation file by file and not directly on a full fileset to allocate less gpu memory
            for i, image_file in enumerate(images_fileset):
                images_segmented, id_im = segmentation(self.Sx, self.Sy, [image_file],
                                                       model_file, self.resize)

                for j in label_range:
                    f = output_fileset.create_file('%03d_%s' % (i, labels[j]))
                    im = images_segmented[0, j, :, :].cpu().numpy()

                    if labels[j] in self.inverted_labels:
                        im = 1.0 - im
                    if self.binarize:
                        im = im > self.threshold
                        if self.dilation > 0:
                            im = proc2d.dilation(im, self.dilation)
                    im = (im * 255).astype(np.uint8)
                    if labels[j] in self.inverted_labels:
                        im = 255 - im
                    io.write_image(f, im, 'png')
                    orig_metadata = images_fileset[i].get_metadata()
                    f.set_metadata({'image_id': id_im[0][0], **orig_metadata,
                                    'channel': labels[j]})

                del images_segmented
                output_fileset.set_metadata("label_names",
                                            [labels[j] for j in label_range])
        else:
            # APPLY SEGMENTATION
            images_segmented, id_im = segmentation(self.Sx, self.Sy, images_fileset,
                                                   model_file, self.resize)

            # Save class prediction as images, one by one, class per class
            logger.debug(
                "Saving the .astype(np.uint8)segmented images, takes around 15 s")
            for i in range(images_segmented.shape[0]):
                for j in label_range:
                    f = output_fileset.create_file('%03d_%s' % (i, labels[j]))
                    im = images_segmented[i, j, :, :].cpu().numpy()
                    if labels[j] in self.inverted_labels:
                        im = 1.0 - im
                    if self.binarize:
                        im = im > self.threshold
                        if self.dilation > 0:
                            im = proc2d.dilation(im, self.dilation)
                    im = (im * 255).astype(np.uint8)
                    if labels[j] in self.inverted_labels:
                        im = 255 - im
                    io.write_image(f, im, 'png')
                    orig_metadata = images_fileset[i].get_metadata()
                    f.set_metadata({'image_id': id_im[i][0], **orig_metadata,
                                    'channel': labels[j]})
                output_fileset.set_metadata("label_names",
                                            [labels[j] for j in label_range])
