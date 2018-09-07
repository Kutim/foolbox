import logging

import numpy as np
from scipy.stats import entropy as kl_divergence

from .base import Attack
from .base import call_decorator
from ..utils import softmax


class VirtualAdversarialAttack(Attack):
    """ Determines the adversarial direction from the model distribution alone,
    without needing the label information.

    Implements Virtual Adversarial Attacks introduced in [1]_.

    References
    ----------
    .. [1] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae,
           Shin Ishii, "Distributional Smoothing with Virtual Adversarial
           Training", https://arxiv.org/abs/1507.00677

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 max_iter=100, epsilon=.1, finite_difference=1e-6):

        """Simple and close to optimal gradient-based
        adversarial attack.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        max_iter : int
            Maximum number of steps to perform.
        epsilon : float
            Factor for maximum input variation
        finite_difference : float
            Value for finite difference method to approximate
            the virtual adversarial pertubation
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            return

        if a.target_class() is not None:
            logging.fatal('VirtualAdversarialAttack is an untargeted adversarial attack.')
            return

        min_, max_ = a.bounds()
        perturbed_image = a.original_image.copy()
        dimensions = a.original_image.shape
        direction = np.random.randn(*dimensions)  # random direction

        logits_original = softmax(a.predictions(a.original_image)[0])

        for _ in range(max_iter):
            direction = np.linalg.norm(direction)

            perturbed_image += direction
            perturbed_image = np.clip(perturbed_image, min_, max_)  # TODO: otherwise out of bounds?

            logits_new = softmax(a.predictions(perturbed_image)[0])

            kl_divergence_1 = kl_divergence(logits_original, logits_new)

            new_direction = direction.copy()

            for i in range(*dimensions):  # TODO: *dimensions?
                direction[i] += finite_difference
                perturbed_image += direction
                logits_new = softmax(a.predictions(perturbed_image)[0])
                kl_divergence_2 = kl_divergence(logits_original, logits_new)
                new_direction[i] = (kl_divergence_2 - kl_divergence_1) / finite_difference
                direction[i] -= finite_difference

            direction = new_direction.copy()

            perturbed_image = perturbed_image + epsilon * np.linalg.norm(direction)
            perturbed_image = np.clip(perturbed_image, min_, max_)
