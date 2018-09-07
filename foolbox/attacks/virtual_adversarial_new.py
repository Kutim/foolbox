import logging

import numpy as np
from foolbox import Adversarial
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
                 max_iter=100, epsilon=.1, xi=1e-6):

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
        xi : float
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

        probs_original, gradients = self._predict(a, a.original_image)

        for _ in range(max_iter):
            direction = xi * np.linalg.norm(direction)  # l2 norm
            perturbed_image += direction
            perturbed_image = np.clip(perturbed_image, min_, max_)

            # probs_new = self._predict(a, perturbed_image)

            '''kl_div = kl_divergence(softmax(logits_original),
                                   softmax(logits_new))'''
            direction = self._gradient(a, probs_original, perturbed_image)

        perturbed_image += epsilon * np.linalg.norm(direction)
        perturbed_image = np.clip(perturbed_image, min_, max_)

    def _predict(self, a, image):
        logits, gradients, _ = a.predictions_and_gradient(image)
        probs = softmax(logits)
        return probs, gradients

    def _gradient(self, a, probs_original, perturbed_image):

        gradients = 0
        for i in range(a.num_classes()):
            gradients += probs_original[i] * a.gradient(perturbed_image, i)

        return gradients
