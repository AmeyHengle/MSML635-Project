"""Query featurisation.

Two interchangeable backends sit behind one interface:

``SentenceTransformerBackend``
    The approach named in the proposal: a sentence-embedding model followed by
    PCA. Needs ``uv sync --extra embed`` (PyTorch).

``TfidfSvdBackend``
    Character-ngram TF-IDF followed by truncated SVD. No PyTorch, no network,
    runs in seconds. Character ngrams matter here because roughly 9% of
    RouterBench's ``eval_name`` tail is Chinese-language, where whitespace word
    tokenisation collapses.

Both are *fit on a calibration split only*. Fitting the reducer on the full
stream would leak post-shift distribution information backwards into the
features, which is precisely the non-stationarity the project is measuring.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureBackend",
    "SentenceTransformerBackend",
    "TfidfSvdBackend",
    "build_features",
]


class FeatureBackend(ABC):
    """Text -> fixed-width float matrix."""

    n_components: int

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> FeatureBackend:
        """Fit on calibration text only."""

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(len(texts), n_components)`` float32."""

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @property
    def name(self) -> str:
        return type(self).__name__


class TfidfSvdBackend(FeatureBackend):
    """Character-ngram TF-IDF reduced with truncated SVD.

    Parameters
    ----------
    n_components:
        Output dimensionality ``d``. The proposal sweeps ``{32, 64, 128}``.
    analyzer, ngram_range:
        Default to character ngrams so non-whitespace-delimited languages are
        handled. Switch to ``analyzer="word"`` for an English-only ablation.
    """

    def __init__(
        self,
        n_components: int = 64,
        *,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (2, 4),
        max_features: int = 100_000,
        min_df: int = 2,
        seed: int = 0,
    ) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.n_components = int(n_components)
        self._vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True,
            strip_accents=None,
            lowercase=True,
        )
        self._svd = TruncatedSVD(n_components=self.n_components, random_state=seed)
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> TfidfSvdBackend:
        mat = self._vec.fit_transform(list(texts))
        max_rank = min(mat.shape) - 1
        if self.n_components > max_rank:
            logger.warning(
                "n_components=%d exceeds achievable rank %d; reducing",
                self.n_components,
                max_rank,
            )
            self._svd.n_components = max(1, max_rank)
            self.n_components = self._svd.n_components
        self._svd.fit(mat)
        self._fitted = True
        logger.info(
            "TfidfSvdBackend fitted on %d docs -> vocab=%d, d=%d, explained_var=%.3f",
            len(texts),
            mat.shape[1],
            self.n_components,
            float(self._svd.explained_variance_ratio_.sum()),
        )
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfSvdBackend.fit must be called before transform")
        return self._svd.transform(self._vec.transform(list(texts))).astype(np.float32)


class SentenceTransformerBackend(FeatureBackend):
    """Sentence embeddings followed by PCA.

    Requires ``uv sync --extra embed``. Embedding is a one-time offline job:
    run it once, cache the result, and no other collaborator needs PyTorch.
    """

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        n_components: int = 64,
        *,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 128,
        device: str | None = None,
        max_chars: int = 2000,
        seed: int = 0,
    ) -> None:
        self.n_components = int(n_components)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.max_chars = int(max_chars)
        self._seed = seed
        self._model = None
        self._pca = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "sentence-transformers is not installed. Run "
                    "`uv sync --extra embed`, or use TfidfSvdBackend instead."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Raw sentence embeddings, before PCA. Cache these."""
        model = self._load_model()
        clipped = [str(t)[: self.max_chars] for t in texts]
        return np.asarray(
            model.encode(
                clipped,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def fit(self, texts: Sequence[str]) -> SentenceTransformerBackend:
        from sklearn.decomposition import PCA

        emb = self.embed(texts)
        n_comp = min(self.n_components, *emb.shape)
        self._pca = PCA(n_components=n_comp, random_state=self._seed).fit(emb)
        self.n_components = n_comp
        logger.info(
            "SentenceTransformerBackend fitted on %d docs -> d=%d, explained_var=%.3f",
            len(texts),
            self.n_components,
            float(self._pca.explained_variance_ratio_.sum()),
        )
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("SentenceTransformerBackend.fit must be called first")
        return self._pca.transform(self.embed(texts)).astype(np.float32)

    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the fitted PCA to pre-computed embeddings (skips re-encoding)."""
        if self._pca is None:
            raise RuntimeError("SentenceTransformerBackend.fit must be called first")
        return self._pca.transform(embeddings).astype(np.float32)


def build_features(
    texts: Sequence[str],
    fit_index: np.ndarray,
    backend: FeatureBackend,
) -> tuple[np.ndarray, FeatureBackend]:
    """Fit ``backend`` on ``texts[fit_index]`` and transform all of ``texts``.

    Separating the fit index from the transform set is the whole point: it is
    what keeps the reducer from seeing the post-shift stream.

    Returns
    -------
    (X, backend)
        ``X`` is ``(len(texts), backend.n_components)`` float32.
    """
    texts = list(texts)
    fit_index = np.asarray(fit_index, dtype=np.int64)
    if fit_index.size == 0:
        raise ValueError("fit_index is empty; the backend needs calibration data")
    if fit_index.max() >= len(texts) or fit_index.min() < 0:
        raise IndexError("fit_index out of bounds for texts")

    backend.fit([texts[i] for i in fit_index])
    features = backend.transform(texts)

    if features.shape[0] != len(texts):
        raise ValueError(f"backend returned {features.shape[0]} rows for {len(texts)} texts")
    if not np.isfinite(features).all():
        raise ValueError("feature matrix contains NaN or inf")
    return features, backend
