"""
Example usage of the correction pipeline.

Run:
    python main.py

For a real API service, mount TextCorrector into a FastAPI app
and use correct_async() inside async route handlers:

    @app.post("/correct")
    async def correct_endpoint(body: CorrectRequest) -> CorrectResponse:
        corrected = await corrector.correct_async(body.text)
        return CorrectResponse(text=corrected)

    @app.post("/correct/batch")
    async def correct_batch_endpoint(body: BatchRequest) -> BatchResponse:
        results = await corrector.correct_batch_async(body.texts)
        return BatchResponse(texts=results)
"""

import asyncio
import time
from pathlib import Path

from correction import TermDictionary, TextCorrector

DATA_DIR = Path(__file__).parent / "data"


def build_corrector() -> TextCorrector:
    dictionary = TermDictionary()
    dictionary.load_json(str(DATA_DIR / "sample_terms.json"))
    dictionary.build()
    print(f"[init] Loaded {len(dictionary)} correction terms.")
    return TextCorrector(dictionary, max_workers=8)


# ---------------------------------------------------------------------------
# Demo: single text
# ---------------------------------------------------------------------------

async def demo_single(corrector: TextCorrector) -> None:
    text = (
        "오늘 미팅에서는 머신 러닝과 딥 러닝 기반의 인공 지능 서비스를 논의했습니다. "
        "파인 튜닝 전략과 하이퍼 파라미터 최적화 방안도 검토했으며, "
        "클라우드 컴퓨팅 인프라 위에 마이크로 서비스 아키텍처로 배포하기로 결정했습니다."
    )
    print("\n[단일 텍스트 교정]")
    print(f"입력: {text}")
    result = await corrector.correct_async(text)
    print(f"출력: {result}")


# ---------------------------------------------------------------------------
# Demo: batch (simulates concurrent API requests)
# ---------------------------------------------------------------------------

async def demo_batch(corrector: TextCorrector) -> None:
    texts = [
        "머신 러닝 파이프라인을 구축합니다.",
        "딥 러닝 모델을 파인 튜닝합니다.",
        "인공 지능 에이 피 아이를 개발합니다.",
        "데이터 베이스 스키마를 설계합니다.",
        "클라우드 컴퓨팅 환경에 배포합니다.",
        "오버 피팅 방지를 위해 드롭 아웃을 적용합니다.",
        "강화 학습 에이전트를 학습시킵니다.",
        "전이 학습으로 베이스 모델을 활용합니다.",
    ]

    print("\n[배치 교정 (동시 처리)]")
    start = time.perf_counter()
    results = await corrector.correct_batch_async(texts)
    elapsed = time.perf_counter() - start

    for original, corrected in zip(texts, results):
        print(f"  입력: {original}")
        print(f"  출력: {corrected}")
        print()
    print(f"  처리 시간: {elapsed * 1000:.1f}ms ({len(texts)}개 텍스트)")


# ---------------------------------------------------------------------------
# Demo: throughput under simulated concurrent load
# ---------------------------------------------------------------------------

async def demo_load(corrector: TextCorrector, n: int = 200) -> None:
    text = (
        "머신 러닝과 딥 러닝은 인공 지능의 핵심입니다. "
        "파인 튜닝과 하이퍼 파라미터 조정이 중요합니다. "
        "클라우드 컴퓨팅 환경에서 마이크로 서비스로 배포합니다."
    )
    print(f"\n[부하 테스트: {n}개 동시 요청]")
    start = time.perf_counter()
    results = await corrector.correct_batch_async([text] * n)
    elapsed = time.perf_counter() - start
    rps = n / elapsed
    print(f"  완료: {n}개 / {elapsed:.3f}s = {rps:.0f} req/s")
    print(f"  샘플 결과: {results[0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    corrector = build_corrector()
    await demo_single(corrector)
    await demo_batch(corrector)
    await demo_load(corrector, n=200)
    corrector.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
