# -*- coding: utf-8 -*-
"""Check README example."""
import ndd

counts = [12, 4, 12, 4, 5, 3, 1, 5, 1, 2, 2, 2, 2, 11, 3, 4, 12, 12, 1, 2]
result = ndd.entropy(counts, k=100, return_std=True)
print(result)
assert result == (2.8400090835681375, 0.10884840411906187)
