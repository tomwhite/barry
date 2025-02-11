import numpy as np

from cubed.vendor.rechunker import algorithm
from cubed.vendor.rechunker.algorithm import rechunking_plan


def evaluate_stage_v2(shape, read_chunks, int_chunks, write_chunks):
    tasks = algorithm.calculate_single_stage_io_ops(shape, read_chunks, write_chunks)
    read_tasks = tasks if write_chunks != read_chunks else 0
    write_tasks = tasks if read_chunks != int_chunks else 0
    return read_tasks, write_tasks


def evaluate_plan(stages, shape, itemsize):
    total_reads = 0
    total_writes = 0
    for i, stage in enumerate(stages):
        read_chunks, int_chunks, write_chunks = stage
        read_tasks, write_tasks = evaluate_stage_v2(
            shape,
            read_chunks,
            int_chunks,
            write_chunks,
        )
        total_reads += read_tasks
        total_writes += write_tasks
    return total_reads, total_writes


def print_summary(stages, shape, itemsize):
    for i, stage in enumerate(stages):
        print(f"stage={i}: " + " -> ".join(map(str, stage)))
        read_chunks, int_chunks, write_chunks = stage
        read_tasks, write_tasks = evaluate_stage_v2(
            shape,
            read_chunks,
            int_chunks,
            write_chunks,
        )
        print(f"  Tasks: {read_tasks} reads, {write_tasks} writes")
        print(f"  Split chunks: {itemsize*np.prod(int_chunks)/1e6 :1.3f} MB")

    total_reads, total_writes = evaluate_plan(stages, shape, itemsize)
    print("Overall:")
    print(f"  Reads count: {total_reads:1.3e} ({total_reads})")
    print(f"  Write count: {total_writes:1.3e} ({total_writes})")


def rechunker_plan(shape, source_chunks, target_chunks, **kwargs):
    stages = algorithm.multistage_rechunking_plan(
        shape, source_chunks, target_chunks, **kwargs
    )
    return (
        [(source_chunks, source_chunks, stages[0][0])]
        + list(stages)
        + [(stages[-1][-1], target_chunks, target_chunks)]
    )


def test_rechunk_era5():
    # from https://github.com/pangeo-data/rechunker/pull/89

    itemsize = 4
    shape = (350640, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)

    print(f"Total size: {itemsize*np.prod(shape)/1e12:.3} TB")
    print(f"Source chunk count: {np.prod(shape)/np.prod(source_chunks):1.3e}")
    print(f"Target chunk count: {np.prod(shape)/np.prod(target_chunks):1.3e}")
    print(f"Source chunk size: {itemsize*np.prod(source_chunks)/1e6 :1.3f} MB")
    print(f"Target chunk size: {itemsize*np.prod(target_chunks)/1e6 :1.3f} MB")

    print()
    print("Rechunker plan (min_mem=0, max_mem=500 MB):")
    plan = rechunker_plan(
        shape, source_chunks, target_chunks, itemsize=4, min_mem=0, max_mem=int(500e6)
    )
    print_summary(plan, shape, itemsize=4)

    print()
    print("Rechunker plan (min_mem=10 MB, max_mem=500 MB):")
    plan = rechunker_plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize=4,
        min_mem=int(10e6),
        max_mem=int(500e6),
    )
    print_summary(plan, shape, itemsize=4)


def test_rechunk_era5_graph():
    # from https://github.com/pangeo-data/rechunker/pull/89

    itemsize = 4
    shape = (350640, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)

    print(f"Total size: {itemsize*np.prod(shape)/1e12:.3} TB")
    print(f"Source chunk count: {np.prod(shape)/np.prod(source_chunks):1.3e}")
    print(f"Target chunk count: {np.prod(shape)/np.prod(target_chunks):1.3e}")

    for m in range(0, 50):
        plan = rechunker_plan(
            shape,
            source_chunks,
            target_chunks,
            itemsize=4,
            min_mem=int(m * 1e6),
            max_mem=int(500e6),
        )
        total_reads, total_writes = evaluate_plan(plan, shape, itemsize)
        print(m, len(plan), total_reads, total_writes)


def test_rechunk_era5_2d():
    # from https://github.com/pangeo-data/rechunker/pull/89

    itemsize = 4
    shape = (350640, 721 * 1440)
    source_chunks = (31, 721 * 1440)
    target_chunks = (350640, 10 * 10)

    print(f"Total size: {itemsize*np.prod(shape)/1e12:.3} TB")
    print(f"Source chunk count: {np.prod(shape)/np.prod(source_chunks):1.3e}")
    print(f"Target chunk count: {np.prod(shape)/np.prod(target_chunks):1.3e}")

    print()
    print("Rechunker plan (min_mem=0, max_mem=500 MB):")
    plan = rechunker_plan(
        shape, source_chunks, target_chunks, itemsize=4, min_mem=0, max_mem=int(500e6)
    )
    print_summary(plan, shape, itemsize=4)

    print()
    print("Rechunker plan (min_mem=25 MB, max_mem=500 MB):")
    plan = rechunker_plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize=4,
        min_mem=int(25e6),
        max_mem=int(500e6),
    )
    print_summary(plan, shape, itemsize=4)


def test_rechunk_algo():
    shape = (40000,)
    source_chunks = (1000,)
    target_chunks = (999,)
    itemsize = 1
    max_mem = 20000
    plan = rechunking_plan(
        shape=shape,
        source_chunks=source_chunks,
        target_chunks=target_chunks,
        itemsize=itemsize,
        max_mem=max_mem,
    )
    print(plan)
