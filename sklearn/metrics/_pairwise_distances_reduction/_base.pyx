cimport jax.numpy as cnp
import jax.numpy as np

from sklearn import get_config
from cython cimport final
from cython.parallel cimport parallel, prange

from ._datasets_pair cimport DatasetsPair
from ...utils._cython_blas cimport _dot
from ...utils._openmp_helpers cimport _openmp_thread_num
from ...utils._typedefs cimport ITYPE_t, DTYPE_t

from numbers import Integral
from sklearn.utils import check_scalar
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils._typedefs import ITYPE, DTYPE

cnp.import_array()

#####################

cpdef DTYPE_t[::1] _sqeuclidean_row_norms64(
    const DTYPE_t[:, ::1] X,
    ITYPE_t num_threads,
):
    """Compute the squared euclidean norm of the rows of X in parallel.

    This is faster than using np.einsum("ij, ij->i") even when using a single thread.
    """
    cdef:
        # Casting for X to remove the const qualifier is needed because APIs
        # exposed via scipy.linalg.cython_blas aren't reflecting the arguments'
        # const qualifier.
        # See: https://github.com/scipy/scipy/issues/14262
        DTYPE_t * X_ptr = <DTYPE_t *> &X[0, 0]
        ITYPE_t idx = 0
        ITYPE_t n = X.shape[0]
        ITYPE_t d = X.shape[1]
        DTYPE_t[::1] squared_row_norms = np.empty(n, dtype=DTYPE)

    for idx in prange(n, schedule='static', nogil=True, num_threads=num_threads):
        squared_row_norms[idx] = _dot(d, X_ptr + idx * d, 1, X_ptr + idx * d, 1)

    return squared_row_norms


cdef class PairwiseDistancesReduction64:
    """Base 64bit implementation of PairwiseDistancesReduction."""

    def __init__(
        self,
        DatasetsPair datasets_pair,
        chunk_size=None,
        strategy=None,
     ):
        cdef:
            ITYPE_t n_samples_chunk, X_n_full_chunks, Y_n_full_chunks

        if chunk_size is None:
            chunk_size = get_config().get("pairwise_dist_chunk_size", 256)

        self.chunk_size = check_scalar(chunk_size, "chunk_size", Integral, min_val=20)

        self.effective_n_threads = _openmp_effective_n_threads()

        self.datasets_pair = datasets_pair

        self.n_samples_X = datasets_pair.n_samples_X()
        self.X_n_samples_chunk = min(self.n_samples_X, self.chunk_size)
        X_n_full_chunks = self.n_samples_X // self.X_n_samples_chunk
        X_n_samples_remainder = self.n_samples_X % self.X_n_samples_chunk
        self.X_n_chunks = X_n_full_chunks + (X_n_samples_remainder != 0)

        if X_n_samples_remainder != 0:
            self.X_n_samples_last_chunk = X_n_samples_remainder
        else:
            self.X_n_samples_last_chunk = self.X_n_samples_chunk

        self.n_samples_Y = datasets_pair.n_samples_Y()
        self.Y_n_samples_chunk = min(self.n_samples_Y, self.chunk_size)
        Y_n_full_chunks = self.n_samples_Y // self.Y_n_samples_chunk
        Y_n_samples_remainder = self.n_samples_Y % self.Y_n_samples_chunk
        self.Y_n_chunks = Y_n_full_chunks + (Y_n_samples_remainder != 0)

        if Y_n_samples_remainder != 0:
            self.Y_n_samples_last_chunk = Y_n_samples_remainder
        else:
            self.Y_n_samples_last_chunk = self.Y_n_samples_chunk

        if strategy is None:
            strategy = get_config().get("pairwise_dist_parallel_strategy", 'auto')

        if strategy not in ('parallel_on_X', 'parallel_on_Y', 'auto'):
            raise RuntimeError(f"strategy must be 'parallel_on_X, 'parallel_on_Y', "
                               f"or 'auto', but currently strategy='{self.strategy}'.")

        if strategy == 'auto':
            # This is a simple heuristic whose constant for the
            # comparison has been chosen based on experiments.
            if 4 * self.chunk_size * self.effective_n_threads < self.n_samples_X:
                strategy = 'parallel_on_X'
            else:
                strategy = 'parallel_on_Y'

        self.execute_in_parallel_on_Y = strategy == "parallel_on_Y"

        # Not using less, not using more.
        self.chunks_n_threads = min(
            self.Y_n_chunks if self.execute_in_parallel_on_Y else self.X_n_chunks,
            self.effective_n_threads,
        )

    @final
    cdef void _parallel_on_X(self) nogil:
        """Compute the pairwise distances of each row vector of X on Y
        by parallelizing computation on the outer loop on chunks of X
        and reduce them.

        This strategy dispatches chunks of Y uniformly on threads.
        Each thread processes all the chunks of X in turn. This strategy is
        a sequence of embarrassingly parallel subtasks (the inner loop on Y
        chunks) with intermediate datastructures synchronisation at each
        iteration of the sequential outer loop on X chunks.

        Private datastructures are modified internally by threads.

        Private template methods can be implemented on subclasses to
        interact with those datastructures at various stages.
        """
        cdef:
            ITYPE_t Y_start, Y_end, X_start, X_end, X_chunk_idx, Y_chunk_idx
            ITYPE_t thread_num

        with nogil, parallel(num_threads=self.chunks_n_threads):
            thread_num = _openmp_thread_num()

            # Allocating thread datastructures
            self._parallel_on_X_parallel_init(thread_num)

            for X_chunk_idx in prange(self.X_n_chunks, schedule='static'):
                X_start = X_chunk_idx * self.X_n_samples_chunk
                if X_chunk_idx == self.X_n_chunks - 1:
                    X_end = X_start + self.X_n_samples_last_chunk
                else:
                    X_end = X_start + self.X_n_samples_chunk

                # Reinitializing thread datastructures for the new X chunk
                # If necessary, upcast X[X_start:X_end] to 64bit
                self._parallel_on_X_init_chunk(thread_num, X_start, X_end)

                for Y_chunk_idx in range(self.Y_n_chunks):
                    Y_start = Y_chunk_idx * self.Y_n_samples_chunk
                    if Y_chunk_idx == self.Y_n_chunks - 1:
                        Y_end = Y_start + self.Y_n_samples_last_chunk
                    else:
                        Y_end = Y_start + self.Y_n_samples_chunk

                    # If necessary, upcast Y[Y_start:Y_end] to 64bit
                    self._parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
                        X_start, X_end,
                        Y_start, Y_end,
                        thread_num,
                    )

                    self._compute_and_reduce_distances_on_chunks(
                        X_start, X_end,
                        Y_start, Y_end,
                        thread_num,
                    )

                # Adjusting thread datastructures on the full pass on Y
                self._parallel_on_X_prange_iter_finalize(thread_num, X_start, X_end)

            # end: for X_chunk_idx

            # Deallocating thread datastructures
            self._parallel_on_X_parallel_finalize(thread_num)

        # end: with nogil, parallel
        return

    @final
    cdef void _parallel_on_Y(self) nogil:
        """Compute the pairwise distances of each row vector of X on Y
        by parallelizing computation on the inner loop on chunks of Y
        and reduce them.

        This strategy dispatches chunks of Y uniformly on threads.
        Each thread processes all the chunks of X in turn. This strategy is
        a sequence of embarrassingly parallel subtasks (the inner loop on Y
        chunks) with intermediate datastructures synchronisation at each
        iteration of the sequential outer loop on X chunks.

        Private datastructures are modified internally by threads.

        Private template methods can be implemented on subclasses to
        interact with those datastructures at various stages.
        """
        cdef:
            ITYPE_t Y_start, Y_end, X_start, X_end, X_chunk_idx, Y_chunk_idx
            ITYPE_t thread_num

        # Allocating datastructures shared by all threads
        self._parallel_on_Y_init()

        for X_chunk_idx in range(self.X_n_chunks):
            X_start = X_chunk_idx * self.X_n_samples_chunk
            if X_chunk_idx == self.X_n_chunks - 1:
                X_end = X_start + self.X_n_samples_last_chunk
            else:
                X_end = X_start + self.X_n_samples_chunk

            with nogil, parallel(num_threads=self.chunks_n_threads):
                thread_num = _openmp_thread_num()

                # Initializing datastructures used in this thread
                # If necessary, upcast X[X_start:X_end] to 64bit
                self._parallel_on_Y_parallel_init(thread_num, X_start, X_end)

                for Y_chunk_idx in prange(self.Y_n_chunks, schedule='static'):
                    Y_start = Y_chunk_idx * self.Y_n_samples_chunk
                    if Y_chunk_idx == self.Y_n_chunks - 1:
                        Y_end = Y_start + self.Y_n_samples_last_chunk
                    else:
                        Y_end = Y_start + self.Y_n_samples_chunk

                    # If necessary, upcast Y[Y_start:Y_end] to 64bit
                    self._parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
                        X_start, X_end,
                        Y_start, Y_end,
                        thread_num,
                    )

                    self._compute_and_reduce_distances_on_chunks(
                        X_start, X_end,
                        Y_start, Y_end,
                        thread_num,
                    )
                # end: prange

                # Note: we don't need a _parallel_on_Y_finalize similarly.
                # This can be introduced if needed.

            # end: with nogil, parallel

            # Synchronizing the thread datastructures with the main ones
            self._parallel_on_Y_synchronize(X_start, X_end)

        # end: for X_chunk_idx
        # Deallocating temporary datastructures and adjusting main datastructures
        self._parallel_on_Y_finalize()
        return

    # Placeholder methods which have to be implemented

    cdef void _compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        """Compute the pairwise distances on two chunks of X and Y and reduce them.

        This is THE core computational method of PairwiseDistanceReductions64.
        This must be implemented in subclasses agnostically from the parallelization
        strategies.
        """
        return

    def _finalize_results(self, bint return_distance):
        """Callback adapting datastructures before returning results.

        This must be implemented in subclasses.
        """
        return None

    # Placeholder methods which can be implemented

    cdef void compute_exact_distances(self) nogil:
        """Convert rank-preserving distances to exact distances or recompute them."""
        return

    cdef void _parallel_on_X_parallel_init(
        self,
        ITYPE_t thread_num,
    ) nogil:
        """Allocate datastructures used in a thread given its number."""
        return

    cdef void _parallel_on_X_init_chunk(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        """Initialise datastructures used in a thread given its number."""
        return

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        """Initialise datastructures just before the _compute_and_reduce_distances_on_chunks.

        This is eventually used to upcast X[X_start:X_end] to 64bit.
        """
        return

    cdef void _parallel_on_X_prange_iter_finalize(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        """Interact with datastructures after a reduction on chunks."""
        return

    cdef void _parallel_on_X_parallel_finalize(
        self,
        ITYPE_t thread_num
    ) nogil:
        """Interact with datastructures after executing all the reductions."""
        return

    cdef void _parallel_on_Y_init(
        self,
    ) nogil:
        """Allocate datastructures used in all threads."""
        return

    cdef void _parallel_on_Y_parallel_init(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        """Initialise datastructures used in a thread given its number."""
        return

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        """Initialise datastructures just before the _compute_and_reduce_distances_on_chunks.

        This is eventually used to upcast Y[Y_start:Y_end] to 64bit.
        """
        return

    cdef void _parallel_on_Y_synchronize(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        """Update thread datastructures before leaving a parallel region."""
        return

    cdef void _parallel_on_Y_finalize(
        self,
    ) nogil:
        """Update datastructures after executing all the reductions."""
        return
