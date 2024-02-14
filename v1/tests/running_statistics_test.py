# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for running statistics utilities.

This file was taken from acme and modified to simplify dependencies:

https://github.com/deepmind/acme/blob/master/acme/jax/running_statistics_test.py
"""

import functools
import math

from absl.testing import absltest
from brax.training.acme import running_statistics
from brax.training.acme import specs
from jax import config as jax_config
import jax.numpy as jnp
import numpy as np

update_and_validate = functools.partial(
    running_statistics.update, validate_shapes=True)


class RunningStatisticsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    jax_config.update('jax_enable_x64', False)

  def assert_allclose(self,
                      actual: jnp.ndarray,
                      desired: jnp.ndarray,
                      err_msg: str = '') -> None:
    np.testing.assert_allclose(
        actual, desired, atol=1e-5, rtol=1e-5, err_msg=err_msg)

  def test_normalize(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.dtype('float32')))

    x = jnp.arange(200, dtype=jnp.float32).reshape(20, 2, 5)
    x1, x2, x3, x4 = jnp.split(x, 4, axis=0)

    state = update_and_validate(state, x1)
    state = update_and_validate(state, x2)
    state = update_and_validate(state, x3)
    state = update_and_validate(state, x4)
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized)
    std = jnp.std(normalized)
    self.assert_allclose(mean, jnp.zeros_like(mean))
    self.assert_allclose(std, jnp.ones_like(std))

  def test_init_normalize(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.dtype('float32')))

    x = jnp.arange(200, dtype=jnp.float32).reshape(20, 2, 5)
    normalized = running_statistics.normalize(x, state)

    self.assert_allclose(normalized, x)

  def test_one_batch_dim(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.dtype('float32')))

    x = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized, axis=0)
    std = jnp.std(normalized, axis=0)
    self.assert_allclose(mean, jnp.zeros_like(mean))
    self.assert_allclose(std, jnp.ones_like(std))

  def test_clip(self):
    state = running_statistics.init_state(specs.Array((), jnp.dtype('float32')))

    x = jnp.arange(5, dtype=jnp.float32)

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state, max_abs_value=1.0)

    mean = jnp.mean(normalized)
    std = jnp.std(normalized)
    self.assert_allclose(mean, jnp.zeros_like(mean))
    self.assert_allclose(std, jnp.ones_like(std) * math.sqrt(0.6))

  def test_validation(self):
    state = running_statistics.init_state(specs.Array((1, 2, 3), jnp.dtype('float32')))

    x = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)
    with self.assertRaises(AssertionError):
      update_and_validate(state, x)

    x = jnp.arange(3, dtype=jnp.float32).reshape(1, 1, 3)
    with self.assertRaises(AssertionError):
      update_and_validate(state, x)

  def test_int_not_normalized(self):
    state = running_statistics.init_state(specs.Array((), jnp.dtype('int32')))

    x = jnp.arange(5, dtype=jnp.int32)

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state)

    np.testing.assert_array_equal(normalized, x)

  def test_weights(self):
    state = running_statistics.init_state(specs.Array((), jnp.dtype('float32')))

    x = jnp.arange(5, dtype=jnp.float32)
    x_weights = jnp.ones_like(x)
    y = 2 * x + 5
    y_weights = 2 * x_weights
    z = jnp.concatenate([x, y])
    weights = jnp.concatenate([x_weights, y_weights])

    state = update_and_validate(state, z, weights=weights)

    self.assertEqual(state.mean, (jnp.mean(x) + 2 * jnp.mean(y)) / 3)
    big_z = jnp.concatenate([x, y, y])
    normalized = running_statistics.normalize(big_z, state)
    self.assertAlmostEqual(jnp.mean(normalized), 0., places=6)
    self.assertAlmostEqual(jnp.std(normalized), 1., places=6)

  def test_denormalize(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.dtype('float32')))

    x = jnp.arange(100, dtype=jnp.float32).reshape(10, 2, 5)
    x1, x2 = jnp.split(x, 2, axis=0)

    state = update_and_validate(state, x1)
    state = update_and_validate(state, x2)
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized)
    std = jnp.std(normalized)
    self.assert_allclose(mean, jnp.zeros_like(mean))
    self.assert_allclose(std, jnp.ones_like(std))

    denormalized = running_statistics.denormalize(normalized, state)
    self.assert_allclose(denormalized, x)

  def test_nest(self):
    state = running_statistics.init_state(
        dict(dummy=specs.Array((5,), jnp.dtype('float32'))))

    x = dict(dummy=jnp.arange(10, dtype=jnp.float32).reshape(2, 5))

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized['dummy'], axis=0)
    std = jnp.std(normalized['dummy'], axis=0)

    self.assert_allclose(mean, jnp.zeros_like(mean))
    self.assert_allclose(std, jnp.ones_like(std))


if __name__ == '__main__':
  absltest.main()
