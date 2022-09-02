import pandas as pd
import pytest

pytest.importorskip("lithops")

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.runtime.executors.lithops import LithopsDagExecutor
from cubed.tests.utils import LITHOPS_LOCAL_CONFIG


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, max_mem=2_000_000_000)


@pytest.fixture()
def reserved_mem():
    executor = LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG)
    return cubed.measure_reserved_memory(executor)


# Array Object


@pytest.mark.slow
def test_index(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = a[1:, :]
    run_operation("index", b, reserved_mem)


# Creation Functions


@pytest.mark.slow
def test_eye(spec, reserved_mem):
    a = xp.eye(10000, 10000, chunks=(5000, 5000), spec=spec)
    run_operation("eye", a, reserved_mem)


@pytest.mark.slow
def test_tril(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.tril(a)
    run_operation("tril", b, reserved_mem)


# Elementwise Functions


@pytest.mark.slow
def test_add(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    run_operation("add", c, reserved_mem)


@pytest.mark.slow
def test_negative(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.negative(a)
    run_operation("negative", b, reserved_mem)


# Linear Algebra Functions


@pytest.mark.slow
def test_matmul(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.matmul(c, d)
    run_operation("matmul", e, reserved_mem)


@pytest.mark.slow
def test_matrix_transpose(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.matrix_transpose(a)
    run_operation("matrix_transpose", b, reserved_mem)


@pytest.mark.slow
def test_tensordot(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.tensordot(c, d, axes=1)
    run_operation("tensordot", e, reserved_mem)


# Manipulation Functions


@pytest.mark.slow
def test_concat(spec, reserved_mem):
    # Note 'a' has one fewer element in axis=0 to force chunking to cross array boundaries
    a = cubed.random.random(
        (9999, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.concat((a, b), axis=0)
    run_operation("concat", c, reserved_mem)


@pytest.mark.slow
def test_reshape(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    # need intermediate reshape due to limitations in Dask's reshape_rechunk
    b = xp.reshape(a, (5000, 2, 10000))
    c = xp.reshape(b, (5000, 20000))
    run_operation("reshape", c, reserved_mem)


@pytest.mark.slow
def test_stack(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.stack((a, b), axis=0)
    run_operation("stack", c, reserved_mem)


# Searching Functions


@pytest.mark.slow
def test_argmax(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.argmax(a, axis=0)
    run_operation("argmax", b, reserved_mem)


# Statistical Functions


@pytest.mark.slow
def test_max(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.max(a, axis=0)
    run_operation("max", b, reserved_mem)


@pytest.mark.slow
def test_mean(spec, reserved_mem):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.mean(a, axis=0)
    run_operation("mean", b, reserved_mem)


# Internal functions


def run_operation(name, result_array, reserved_mem):
    # result_array.visualize(f"cubed-{name}-unoptimized", optimize_graph=False)
    # result_array.visualize(f"cubed-{name}")
    executor = LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG)
    hist = HistoryCallback()
    # use store=None to write to temporary zarr
    cubed.to_zarr(result_array, store=None, executor=executor, callbacks=[hist])

    plan_df = pd.read_csv(hist.plan_df_path)
    stats_df = pd.read_csv(hist.stats_df_path)
    df = analyze(plan_df, stats_df, reserved_mem)
    print(df)

    # check utilization does not exceed 1
    assert (df["utilization"] <= 1.0).all()


def analyze(plan_df, stats_df, reserved_mem):

    reserved_mem_mb = reserved_mem / 1_000_000
    reserved_mem_mb *= 1.05  # add some wiggle room

    # convert memory to MB
    plan_df["required_mem_mb"] = plan_df["required_mem"] / 1_000_000
    plan_df["total_mem_mb"] = plan_df["required_mem_mb"] + reserved_mem_mb
    plan_df = plan_df[
        ["array_name", "op_name", "required_mem_mb", "total_mem_mb", "num_tasks"]
    ]
    stats_df["peak_mem_start_mb"] = stats_df["peak_memory_start"] / 1_000_000
    stats_df["peak_mem_end_mb"] = stats_df["peak_memory_end"] / 1_000_000
    stats_df["peak_mem_delta_mb"] = (
        stats_df["peak_mem_end_mb"] - stats_df["peak_mem_start_mb"]
    )

    # find per-array stats
    df = stats_df.groupby("array_name", as_index=False).agg(
        {
            "peak_mem_start_mb": ["min", "mean", "max"],
            "peak_mem_end_mb": ["max"],
            "peak_mem_delta_mb": ["min", "mean", "max"],
        }
    )

    # flatten multi-index
    df.columns = ["_".join(a).rstrip("_") for a in df.columns.to_flat_index()]
    df = df.merge(plan_df, on="array_name")

    def utilization(row):
        return row["peak_mem_end_mb_max"] / row["total_mem_mb"]

    df["utilization"] = df.apply(lambda row: utilization(row), axis=1)
    df = df[
        [
            "array_name",
            "op_name",
            "num_tasks",
            "peak_mem_start_mb_max",
            "peak_mem_end_mb_max",
            "peak_mem_delta_mb_max",
            "required_mem_mb",
            "total_mem_mb",
            "utilization",
        ]
    ]

    return df
