"""
TextCorrector: parallel sentence-level correction using asyncio + ThreadPoolExecutor.

Design for API service usage
------------------------------
Create ONE TextCorrector instance at application startup and reuse it across all
requests.  The underlying TermDictionary automaton is immutable after build(),
making it safe to share across concurrent threads and asyncio tasks.

    # at startup
    dictionary = TermDictionary()
    dictionary.load_json("data/terms.json")
    dictionary.build()
    corrector = TextCorrector(dictionary, max_workers=8)

    # per request (async handler)
    corrected = await corrector.correct_async(asr_text)

    # batch: multiple texts concurrently (e.g. from a batch API endpoint)
    results = await corrector.correct_batch_async([text1, text2, text3])

Parallelism model
-----------------
- Sentence splitting:  done synchronously in the calling thread (cheap regex).
- Per-sentence correction: offloaded to ThreadPoolExecutor so multiple sentences
  run in parallel.  The Aho-Corasick C extension releases the GIL, giving true
  CPU-level parallelism even in a multi-threaded Python process.
- Multiple concurrent requests: asyncio.gather() fans out naturally; each request
  submits its sentences to the shared thread pool.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .sentence_splitter import rejoin, split_sentences
from .term_dictionary import TermDictionary


class TextCorrector:
    """
    Applies rule-based corrections to ASR text with sentence-level parallelism.

    Parameters
    ----------
    dictionary:
        A TermDictionary instance.  Should already have build() called before
        the first request arrives to avoid a cold-start latency spike.
    max_workers:
        Thread pool size.  Defaults to min(32, cpu_count + 4), which matches
        the ThreadPoolExecutor default.  For I/O-light, CPU-bound correction
        tasks a value close to cpu_count is usually optimal.
    """

    def __init__(
        self,
        dictionary: TermDictionary,
        max_workers: Optional[int] = None,
    ) -> None:
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        self._dict = dictionary
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="corrector")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def correct_async(self, text: str) -> str:
        """
        Correct a single text asynchronously.

        Splits into sentences, processes them in parallel via the thread pool,
        then rejoins preserving the original separators.

        Safe to call concurrently from multiple async tasks.
        """
        if not text or not text.strip():
            return text

        sentences = split_sentences(text)
        if not sentences:
            return text

        loop = asyncio.get_running_loop()
        corrected: List[str] = list(
            await asyncio.gather(
                *[loop.run_in_executor(self._executor, self._dict.correct, s) for s in sentences]
            )
        )
        return rejoin(text, sentences, corrected)

    async def correct_batch_async(self, texts: List[str]) -> List[str]:
        """
        Correct multiple texts concurrently.

        All texts are processed in parallel — both across texts and within each
        text at the sentence level.  Ideal for batch API endpoints.
        """
        return list(await asyncio.gather(*[self.correct_async(t) for t in texts]))

    def correct(self, text: str) -> str:
        """
        Synchronous convenience wrapper for scripts and tests.

        Do NOT use this inside a running asyncio event loop (e.g. a FastAPI
        handler) — use correct_async() there instead.
        """
        return asyncio.run(self.correct_async(text))

    def correct_batch(self, texts: List[str]) -> List[str]:
        """Synchronous batch wrapper. See correct() note above."""
        return asyncio.run(self.correct_batch_async(texts))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool. Call at application teardown."""
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "TextCorrector":
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()
