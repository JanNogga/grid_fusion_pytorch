import os

from torch.utils.cpp_extension import load

parent_dir = os.path.dirname(os.path.abspath(__file__))

# add_util_cuda = load(
#         name='add_util_cuda',
#         sources=[
#             os.path.join(parent_dir, path)
#             for path in ['cuda/dummy.cpp', 'cuda/dummy.cu']],
#         verbose=True)

increment_misses_util_cuda = load(
        name='increment_misses_util_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/miss_counter.cpp', 'cuda/miss_counter.cu']],
        verbose=True)


# def custom_add(A,B):
#     return add_util_cuda.vector_add_free_function(A,B)

def increment_misses(Grids, Origs, Dirs, Dists, RangeMin, RangeMax, n_steps=16):
    return increment_misses_util_cuda.increment_misses_free_function(Grids, Origs, Dirs, Dists, RangeMin, RangeMax, n_steps)
