__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 06, 2015 15:07:46 EDT$"


import json
import operator
import os
import shutil
import tempfile

import h5py
import numpy

import nanshe.util
import nanshe.flow.learner


class TestLearner(object):
    def setup(self):
        self.config_a_block = {
            "debug" : True,
            "generate_neurons" : {
                "postprocess_data" : {
                    "wavelet_denoising" : {
                        "remove_low_intensity_local_maxima" : {
                            "percentage_pixels_below_max" : 0
                        },
                        "wavelet.transform" : {
                            "scale" : 4
                        },
                        "accepted_region_shape_constraints" : {
                            "major_axis_length" : {
                                "max" : 25.0,
                                "min" : 0.0
                            }
                        },
                        "accepted_neuron_shape_constraints" : {
                            "eccentricity" : {
                                "max" : 0.9,
                                "min" : 0.0
                            },
                            "area" : {
                                "max" : 600,
                                "min" : 30
                            }
                        },
                        "estimate_noise" : {
                            "significance_threshold" : 3.0
                        },
                        "significant_mask" : {
                            "noise_threshold" : 3.0
                        },
                        "remove_too_close_local_maxima" : {
                            "min_local_max_distance" : 100.0
                        },
                        "use_watershed" : True
                    },
                    "merge_neuron_sets" : {
                        "alignment_min_threshold" : 0.6,
                        "fuse_neurons" : {
                            "fraction_mean_neuron_max_threshold" : 0.01
                        },
                        "overlap_min_threshold" : 0.6
                    }
                },
                "run_stage" : "all",
                "preprocess_data" : {
                    "normalize_data" : {
                        "renormalized_images" : {
                            "ord" : 2
                        }
                    },
                    "extract_f0" : {
                        "spatial_smoothing_gaussian_filter_stdev" : 5.0,
                        "spatial_smoothing_gaussian_filter_window_size" : 5.0,
                        "which_quantile" : 0.5,
                        "temporal_smoothing_gaussian_filter_stdev" : 5.0,
                        "temporal_smoothing_gaussian_filter_window_size" : 5.0,
                        "half_window_size" : 1,
                        "bias" : 100
                    },
                    "remove_zeroed_lines" : {
                        "erosion_shape" : [
                            21,
                            1
                        ],
                        "dilation_shape" : [
                            1,
                            3
                        ]
                    },
                    "wavelet.transform" : {
                        "scale" : [
                            3,
                            4,
                            4
                        ]
                    }
                },
                "generate_dictionary" : {
                    "spams.trainDL" : {
                        "gamma2" : 0,
                        "gamma1" : 0,
                        "numThreads" : 1,
                        "K" : 10,
                        "iter" : 100,
                        "modeD" : 0,
                        "posAlpha" : True,
                        "clean" : True,
                        "posD" : True,
                        "batchsize" : 256,
                        "lambda1" : 0.2,
                        "lambda2" : 0,
                        "mode" : 2
                    }
                }
            }
        }

        self.temp_dir = tempfile.mkdtemp(dir=os.environ.get("TEMP", None))
        self.temp_dir = os.path.abspath(self.temp_dir)

        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir = os.path.abspath(self.temp_dir)

        self.ruffus_sqlite_filename = os.path.join(self.temp_dir, ".ruffus_history.sqlite")
        self.hdf5_input_filename = os.path.join(self.temp_dir, "data.raw.h5")
        self.hdf5_input_filepath = self.hdf5_input_filename + "/" + "images"
        self.hdf5_output_filename = os.path.join(self.temp_dir, "data.post.h5")
        self.hdf5_output_filepath = self.hdf5_output_filename + "/"

        self.config_a_block_filename = os.path.join(self.temp_dir, "config_a_block.json")

        self.space = numpy.array([110, 110])
        self.radii = numpy.array([6, 6, 6, 6, 7, 6])
        self.magnitudes = numpy.array([15, 16, 15, 17, 16, 16])
        self.points = numpy.array([[30, 24],
                                   [59, 65],
                                   [21, 65],
                                   [80, 78],
                                   [72, 16],
                                   [45, 32]])

        self.bases_indices = [[1, 3, 4], [0, 2], [5]]
        self.linspace_length = 25

        self.masks = nanshe.syn.data.generate_hypersphere_masks(self.space, self.points, self.radii)
        self.images = nanshe.syn.data.generate_gaussian_images(self.space, self.points, self.radii/3.0, self.magnitudes) * self.masks

        self.bases_masks = numpy.zeros((len(self.bases_indices),) + self.masks.shape[1:] , dtype=self.masks.dtype)
        self.bases_images = numpy.zeros((len(self.bases_indices),) + self.images.shape[1:] , dtype=self.images.dtype)

        for i, each_basis_indices in enumerate(self.bases_indices):
            self.bases_masks[i] = self.masks[list(each_basis_indices)].max(axis = 0)
            self.bases_images[i] = self.images[list(each_basis_indices)].max(axis = 0)

        self.image_stack = None
        ramp = numpy.concatenate([numpy.linspace(0, 1, self.linspace_length), numpy.linspace(1, 0, self.linspace_length)])

        self.image_stack = numpy.zeros((self.bases_images.shape[0] * len(ramp),) + self.bases_images.shape[1:],
                                       dtype = self.bases_images.dtype)
        for i in xrange(len(self.bases_images)):
            image_stack_slice = slice(i * len(ramp), (i+1) * len(ramp), 1)

            self.image_stack[image_stack_slice] = nanshe.util.xnumpy.all_permutations_operation(operator.mul,
                                                                                                   ramp,
                                                                                                   self.bases_images[i])

        with h5py.File(self.hdf5_input_filename, "w") as fid:
            fid["images"] = self.image_stack

        with open(self.config_a_block_filename, "w") as fid:
            json.dump(self.config_a_block, fid)
            fid.write("\n")

        print os.getcwd()


    def test_main_1(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)

            with open(os.path.expanduser("~/test.log"), "w") as f:
                f.write(str(os.listdir(self.temp_dir)))

            exit_code = nanshe.flow.learner.main(
                nanshe.flow.learner.main.__name__.replace(".pyc", ".py"),
                "--forced_tasks", "postprocess_data",
                "-c",
                os.path.basename(self.config_a_block_filename)
            )


            assert exit_code == 0
        finally:
            os.chdir(cwd)

        print os.listdir(self.temp_dir)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("images" in fid)

            neurons = fid["images"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis = 0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)


    def teardown(self):
        try:
            os.remove(self.ruffus_sqlite_filename)
        except OSError:
            pass
        self.ruffus_sqlite_filename = ""

        try:
            os.remove(self.config_a_block_filename)
        except OSError:
            pass
        self.config_a_block_filename = ""

        try:
            os.remove(self.hdf5_input_filename)
        except OSError:
            pass
        self.hdf5_input_filename = ""

        try:
            os.remove(self.hdf5_output_filename)
        except OSError:
            pass
        self.hdf5_output_filename = ""

        shutil.rmtree(self.temp_dir)
        self.temp_dir = ""

        tempfile.tempdir = None
