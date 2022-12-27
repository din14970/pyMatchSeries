import numpy as np
from dateclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from skimage.transform import pyramid_gaussian, resize
from tqdm import tqdm


@dataclass
class JNNRConfig:
    pass


class JNRR:
    def __init__(self):
        pass


def main():
    # im1 and im2 are two images (float32 dtype) of the same size assumed to be available
    im1 = np.zeros((128, 128), dtype=np.float32)
    im2 = np.zeros((128, 128), dtype=np.float32)

    im1[2 * 10: 2 * 30, 2 * 15: 2 * 45] = 1
    im2[2 * 25: 2 * 45, 2 * 25: 2 * 55] = 1

    # Regularization parameter
    L = 0.1
    num_levels = 5

    # Create an image hierarchy for both of our images
    pyramid_tem = tuple(
        pyramid_gaussian(im1, max_layer=num_levels - 1, downscale=2, channel_axis=None)
    )
    pyramid_ref = tuple(
        pyramid_gaussian(im2, max_layer=num_levels - 1, downscale=2, channel_axis=None)
    )

    disp_new = None

    for i in reversed(range(num_levels)):
        image_tem = pyramid_tem[i]
        image_ref = pyramid_ref[i]

        objective = RegistrationObjectiveFunction(image_tem, image_ref, L)

        # initialize displacement as zero if we no guess from a previous level
        if disp_new is None:
            disp = np.zeros_like(objective.identity)
        # initialize displacement by upsampling the one from the previous level
        else:
            disp = np.stack(
                [
                    resize(disp_new[0, ...], image_tem.shape),
                    resize(disp_new[1, ...], image_tem.shape),
                ]
            )

        disp_new = GaussNewtonAlgorithm(
            disp.ravel(),
            objective.evaluate_residual,
            objective.evaluate_residual_gradient,
        ).reshape(disp.shape)
        # res = minimize(objective.evaluate_energy, disp.ravel(), jac=objective.evaluate_energy_gradient, method="BFGS", options={"disp": True, "maxiter": 1000})
        # disp_new = res.x.reshape(disp.shape)

        mpl.rcParams["image.cmap"] = "gray"
        _, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].title.set_text("tem")
        ax[0].imshow(image_tem)
        ax[1].title.set_text("ref")
        ax[1].imshow(image_ref)
        ax[2].title.set_text("tem(phi)")
        ax[2].imshow(
            map_coordinates(
                image_tem,
                [
                    disp_new[0, ...] / objective.grid_h + objective.identity[0, ...],
                    disp_new[1, ...] / objective.grid_h + objective.identity[1, ...],
                ],
            )
        )
        plt.suptitle(f"Result for resolution {image_tem.shape}")
        plt.show()


if __name__ == "__main__":
    main()
