import numpy as np
from collections import deque

class BayesRelEstimator:
    def __init__(self, memory_size=30, categories=3, threshold=0.01, target_category=2):
        self.categories = categories
        self.threshold = threshold
        self.target_category = target_category
        self.pe_records = deque(maxlen=memory_size)
        self.pe_counts = np.zeros(categories, dtype=np.int64)

    def _cat(self, pe):
        if pe < -self.threshold: return 1
        elif pe > +self.threshold: return 2
        else: return 0

    def add(self, pe):
        if len(self.pe_records) == self.pe_records.maxlen:
            self.pe_counts[self.pe_records[0]] -= 1
        c = self._cat(pe)
        self.pe_records.append(c)
        self.pe_counts[c] += 1

    def _post_mean(self, c):
        K = self.categories
        N = len(self.pe_records)
        return (1 + self.pe_counts[c]) / (K + N)

    def _post_var(self, c):
        K = self.categories
        N = len(self.pe_records)
        num = (1 + self.pe_counts[c]) * (K + N - (1 + self.pe_counts[c]))
        den = (K + N) ** 2 * (K + N + 1)
        return num / (den + 1e-12)

    def reliability(self):
        chis = []
        for c in range(self.categories):
            m = self._post_mean(c)
            v = self._post_var(c)
            chis.append(m / (v + 1e-12))
        rel = chis[self.target_category] / (np.sum(chis) + 1e-12)
        return float(np.clip(rel, 0.0, 1.0))
