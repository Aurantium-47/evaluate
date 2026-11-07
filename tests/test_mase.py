# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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
import numpy as np
import pytest

from evaluate import load


@pytest.fixture
def mase():
    return load("./metrics/mase", config_name="multilist", keep_in_memory=True)


@pytest.mark.parametrize("multioutput", ["uniform_average", [0.3, 0.7], np.array([0.3, 0.7])])
@pytest.mark.parametrize("sample_weight", [None, [1.0, 3.0]])
@pytest.mark.parametrize("periodicity", [1, 2])
def test_mase_averages_scaled_output_errors(mase, multioutput, sample_weight, periodicity):
    inputs = dict(
        predictions=[[1.0, 10.0], [3.0, 20.0]],
        references=[[0.0, 0.0], [0.0, 0.0]],
        training=[[0.0, 0.0], [1.0, 100.0], [2.0, 200.0]],
        periodicity=periodicity,
        sample_weight=sample_weight,
    )
    raw = mase.compute(**inputs, multioutput="raw_values")["mase"]
    expected_raw = np.average([[1.0, 0.1], [3.0, 0.2]], axis=0, weights=sample_weight) / periodicity
    np.testing.assert_allclose(raw, expected_raw)
    weights = None if isinstance(multioutput, str) else multioutput
    actual = mase.compute(**inputs, multioutput=multioutput)["mase"]
    assert actual == pytest.approx(np.average(expected_raw, weights=weights))


@pytest.mark.parametrize("scale", [0.001, 1.0, 100.0])
@pytest.mark.parametrize("multioutput, expected", [("uniform_average", 0.55), ([0.3, 0.7], 0.37)])
def test_mase_independent_output_units(mase, scale, multioutput, expected):
    result = mase.compute(
        predictions=[[1.0, 10.0 * scale], [1.0, 10.0 * scale]],
        references=[[0.0, 0.0], [0.0, 0.0]],
        training=[[0.0, 0.0], [1.0, 100.0 * scale], [2.0, 200.0 * scale]],
        multioutput=multioutput,
    )
    assert result["mase"] == pytest.approx(expected)


@pytest.mark.parametrize("step", [0.0, np.finfo(np.float64).eps / 2])
def test_mase_scales_constant_outputs_before_averaging(mase, step):
    inputs = dict(
        predictions=[[1.0, 10.0]],
        references=[[0.0, 0.0]],
        training=[[0.0, 0.0], [step, 100.0]],
    )
    raw = mase.compute(**inputs, multioutput="raw_values")["mase"]
    np.testing.assert_allclose(raw, [1 / np.finfo(np.float64).eps, 0.1])
    assert mase.compute(**inputs)["mase"] == pytest.approx(np.mean(raw))


@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
def test_mase_univariate_unchanged(multioutput):
    metric = load("./metrics/mase", keep_in_memory=True)
    result = metric.compute(
        predictions=[2.5, 0.0, 2.0, 8.0, 1.25],
        references=[3.0, -0.5, 2.0, 7.0, 2.0],
        training=[5.0, 0.5, 4.0, 6.0, 3.0, 5.0, 2.0],
        multioutput=multioutput,
    )
    np.testing.assert_allclose(result["mase"], 0.18333333333333335)


def test_mase_incremental_matches_direct(mase):
    mase.add(prediction=[1.0, 10.0], reference=[0.0, 0.0])
    mase.add_batch(predictions=[[3.0, 20.0]], references=[[0.0, 0.0]])
    result = mase.compute(training=[[0.0, 0.0], [1.0, 100.0]])
    assert result["mase"] == pytest.approx(1.075)


@pytest.mark.parametrize("multioutput", ["invalid", [1.0, 2.0, 3.0], [[1.0, 2.0]], None])
def test_mase_rejects_invalid_output_weights(mase, multioutput):
    with pytest.raises(ValueError):
        mase.compute(
            predictions=[[1.0, 10.0]],
            references=[[0.0, 0.0]],
            training=[[0.0, 0.0], [1.0, 100.0]],
            multioutput=multioutput,
        )


def test_mase_rejects_custom_weights_for_univariate_output():
    metric = load("./metrics/mase", keep_in_memory=True)
    with pytest.raises(ValueError, match="Custom weights"):
        metric.compute(predictions=[1.0], references=[0.0], training=[0.0, 1.0], multioutput=[1.0])
