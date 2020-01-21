from imageio import imwrite
from io import IOBytes
import numpy as np


class Artist(object):
    """
    Generates aurora-like training curves from batch accuracy histories.
    """

    def __init__(self, samples_per_timestep=8, num_scatters=3, scatter_spread=64,
                 scatter_group_dim=8, strictness=5, target_heat=0.1):
        """
        Initialize.

        - samples_per_timestep  Number of samples kept per timestep.  Uses
                                distributions instead of single values for balance.

        - num_scatters          Number of scatter/mean iterations.

        - scatter_spread        The scattering breadth.  Higher values increasingly
                                iron out the training curves.  Lower stays jagged.

        - scatter_group_dim     Size of the groups that are averaged over.  Higher
                                values result in clean, narrow beams.  Lower values
                                normalize less, leaving more artifacts.

        - strictness            How aligned accuracy/height needs to be to light up.

        - target_heat           Desired mean that heatmap activations are scaled to.
        """
        self.samples_per_timestep = samples_per_timestep
        self.num_scatters = num_scatters
        self.scatter_group_dim = scatter_group_dim
        self.scatter_spread = scatter_spread
        self.strictness = strictness
        self.target_heat = target_heat

    @classmethod
    def scatter_and_mean(cls, accuracies, spread=64, group_dim=8):
        """
        Scatter the values around a bit, then return the means of groups of them.
        """
        size = accuracies.shape[0]

        # Start with indices unchanged.
        indices = np.arange(size).astype(np.int32)

        # Then add rounded Gaussian noise to scatter.
        indices = np.expand_dims(indices, 0)
        indices = np.tile(indices, (group_dim, 1))
        indices += np.random.normal(0, spread, indices.shape).astype(np.int32)

        # Avoid weirdness by reflecting intrusions off both ends of the array.
        indices = np.abs(indices)
        indices = size - 1 - indices
        indices = np.abs(indices)
        indices = size - 1 - indices

        # Now do the scatter.
        accuracies = accuracies[indices]

        # Blend values the fell on the same slot, smoothing the result.
        return accuracies.mean(0)

    @classmethod
    def history_to_heatmap(cls, accuracies, height, samples_per_timestep=8,
                           num_scatters=3, scatter_spread=64, scatter_group_dim=8,
                           strictness=5, target_heat=0.1):
        """
        Convert the historical accuracy distribution into a heatmap.
        """
        # Truncate incomplete last timestep.
        size = accuracies.shape[0]
        size = size - size % samples_per_timestep
        accuracies = accuracies[:size]

        # Narrow and straighten the accuracy distribution.
        for i in range(num_scatters):
            accuracies = cls.scatter_and_mean(accuracies, scatter_spread,
                                              scatter_group_dim)

        # Squash the normalized accuracy values into groups of size
        # samples_per_timestep so that we have a distribution of values to compare
        # against each timestep.
        accuracies = accuracies.reshape(1, -1, samples_per_timestep)
        #width = accuracies.shape[1]

        # Batch accuracies fall between 0 and 1, so the grid heights are laid out the
        # same way.
        grid = np.arange(height).astype(np.float32) / height

        # Construct the output image grid of shape (height/accuracy, width/timestep,
        # channels/samples_per_timestep).
        grid = grid.reshape(height, 1, 1)

        # Get height-accuracy alignment (from 1 to 0), then raise that to a large
        # power to only light up if close by.  Then even it out by reducing on the
        # sample dimension.
        heat = 1 - np.abs(x - grid)
        heat **= strictness
        heat = heat.mean(2)

        # Then scale it so that it's not too dark, not too light.  Taper values that
        # are over the limit.
        heat /= heat.sum()
        heat *= target_heat
        heat = np.tanh(heat / 2) * 2

        # Convert to a grayscale image.
        heat = heat.clip(max=255)
        heat = heat.flip(0)
        return heat.astype(np.uint8)

    @classmethod
    def compose_heatmaps(cls, train, val):
        """
        Overlay the two heatmaps into a color + alpha png image.

        Crimson training curve, bright green validation curve, alpha the rest.
        Designed for a featureless dark blue background.
        """
        red = train
        green = val
        blue = red // 4
        alpha = red // 2 + green // 2
        arr = np.stack([red, green, blue, alpha], 2)
        buf = IOBytes()
        imwrite(buf, arr, format='png')
        return buf.getvalue()

    def __call__(self, train_accs, val_accs, height):
        """
        Generate an image given the batch accuracy histories.
        """
        train_heatmap = self.history_to_heatmap(
            train_accs, height, self.sample_per_timestep, self.num_scatters,
            self.scatter_spread, self.scatter_group_dim, self.strictness,
            self.target_heat)

        val_heatmap = self.history_to_heatmap(
            val_accs, height, self.sample_per_timestep, self.num_scatters,
            self.scatter_spread, self.scatter_group_dim, self.strictness,
            self.target_heat)

        return self.compose_heatmaps(train_heatmap, val_heatmap)
