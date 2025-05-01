from src.install_import import install_if_missing

install_if_missing("time")
install_if_missing("psutil")
install_if_missing("functools")

import time
import psutil
import functools

def track_resources(func):
    """
    Decorator to measure and print CPU usage, memory consumption, and execution time
    of a function.
¡
    The following metrics are reported:
    - Execution time (in seconds)
    - Peak memory usage (in megabytes)
    - CPU usage percentage (sampled after execution)

    Output is printed to the console and flushed immediately, making it suitable for
    use in Jupyter Notebooks and scripts.

    Parameters:
    func (callable): The function whose performance is to be measured.

    Returns:
    callable: A wrapped version of the original function that includes resource tracking
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        children_before = process.children(recursive=True)

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Capture all children spawned during execution
        children_after = process.children(recursive=True)
        new_children = [p for p in children_after if p not in children_before]

        # Total memory usage
        memory_usage = process.memory_info().rss
        for child in new_children:
            try:
                memory_usage += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Convert to MB
        memory_usage_mb = memory_usage / (1024 ** 2)

        elapsed_time = end_time - start_time

        print(f"⏱️ Execution Time: {elapsed_time:.2f} seconds", flush=True)
        print(f"🧠 Memory Usage (incl. subprocesses): {memory_usage_mb:.2f} MB", flush=True)
        return result
    return wrapper