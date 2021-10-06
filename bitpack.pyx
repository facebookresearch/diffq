import cython
from libc.stdint cimport uint64_t, int16_t
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def pack(int16_t[::1] indexes, int nbits, int block_size=32):
    cdef int le = len(indexes)
    cdef int storage = 64
    assert le % (storage * block_size) == 0
    cdef int lines = le // (storage * block_size)
    out = np.zeros((lines, nbits, block_size), dtype=np.uint64)
    cdef uint64_t[:, :, ::1] out_view = out
    cdef int bit_in, bit_out, index, line
    cdef int16_t x
    cdef uint64_t tmp
    for line in range(lines):
        for bit_out in range(storage):
            for bit_in in range(nbits):
                for index in range(block_size):
                    x = indexes[line * block_size * storage + bit_out * block_size + index]
                    tmp = (x >> bit_in) & 1
                    out_view[line, bit_in, index] |= tmp << bit_out
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def unpack(uint64_t[:, :, ::1] packed):
    cdef int lines = packed.shape[0]
    cdef int nbits = packed.shape[1]
    cdef int block_size = packed.shape[2]
    cdef int storage = 64
    out = np.zeros((lines * block_size * storage), dtype=np.int16)
    cdef int16_t[::1] out_view = out
    cdef int bit_in, bit_out, index, line
    cdef int16_t x
    cdef uint64_t tmp
    for line in range(lines):
        for bit_in in range(storage):
            for bit_out in range(nbits):
                for index in range(block_size):
                    tmp = packed[line, bit_out, index]
                    x = (tmp >> bit_in) & 1
                    out_view[line * block_size * storage + bit_in * block_size + index] |= x << bit_out
    return out
