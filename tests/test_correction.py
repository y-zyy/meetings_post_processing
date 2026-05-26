"""
Tests for the correction pipeline.
"""

import asyncio
import time

import pytest

from correction import TermDictionary, TextCorrector, split_sentences


# ---------------------------------------------------------------------------
# TermDictionary tests
# ---------------------------------------------------------------------------


class TestTermDictionary:
    def setup_method(self):
        self.d = TermDictionary()
        self.d.add("머신 러닝", "머신러닝")
        self.d.add("딥 러닝", "딥러닝")
        self.d.add("인공 지능", "인공지능")
        self.d.build()

    def test_exact_match(self):
        assert self.d.correct("머신 러닝 모델") == "머신러닝 모델"

    def test_multiple_matches_in_one_sentence(self):
        result = self.d.correct("머신 러닝과 딥 러닝은 인공 지능의 핵심입니다.")
        assert result == "머신러닝과 딥러닝은 인공지능의 핵심입니다."

    def test_no_match(self):
        text = "오늘 회의 잘 진행됐습니다."
        assert self.d.correct(text) == text

    def test_empty_string(self):
        assert self.d.correct("") == ""

    def test_overlap_longer_wins(self):
        d = TermDictionary()
        d.add("머신 러닝", "머신러닝")
        d.add("머신", "기계")
        d.build()
        # "머신 러닝" should win over "머신" at the same start position
        assert d.correct("머신 러닝 프레임워크") == "머신러닝 프레임워크"

    def test_non_overlapping_all_applied(self):
        result = self.d.correct("머신 러닝 딥 러닝")
        assert result == "머신러닝 딥러닝"

    def test_load_json_dict_format(self, tmp_path):
        import json
        p = tmp_path / "terms.json"
        p.write_text(json.dumps({"잘못": "올바름"}), encoding="utf-8")
        d = TermDictionary()
        d.load_json(str(p))
        d.build()
        assert d.correct("이건 잘못된 표기입니다.") == "이건 올바름된 표기입니다."

    def test_load_json_list_format(self, tmp_path):
        import json
        p = tmp_path / "terms.json"
        p.write_text(json.dumps([{"wrong": "잘못", "correct": "올바름"}]), encoding="utf-8")
        d = TermDictionary()
        d.load_json(str(p))
        d.build()
        assert d.correct("잘못 입력") == "올바름 입력"

    def test_load_csv(self, tmp_path):
        p = tmp_path / "terms.csv"
        p.write_text("wrong,correct\n에이 아이,AI\n", encoding="utf-8")
        d = TermDictionary()
        d.load_csv(str(p))
        d.build()
        assert d.correct("에이 아이 기술") == "AI 기술"

    def test_len_and_contains(self):
        assert len(self.d) == 3
        assert "머신 러닝" in self.d
        assert "없는단어" not in self.d

    def test_auto_build_on_first_correct(self):
        d = TermDictionary()
        d.add("딥 러닝", "딥러닝")
        # build() not called explicitly
        assert d.correct("딥 러닝 모델") == "딥러닝 모델"


# ---------------------------------------------------------------------------
# split_sentences tests
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_period(self):
        assert split_sentences("첫 번째 문장. 두 번째 문장.") == [
            "첫 번째 문장.",
            "두 번째 문장.",
        ]

    def test_question_and_exclamation(self):
        result = split_sentences("정말요? 맞습니다! 잘 됐네요.")
        assert result == ["정말요?", "맞습니다!", "잘 됐네요."]

    def test_newline(self):
        result = split_sentences("첫 줄\n두 번째 줄")
        assert result == ["첫 줄", "두 번째 줄"]

    def test_single_sentence(self):
        assert split_sentences("단일 문장입니다.") == ["단일 문장입니다."]

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []


# ---------------------------------------------------------------------------
# TextCorrector tests
# ---------------------------------------------------------------------------


class TestTextCorrector:
    def setup_method(self):
        d = TermDictionary()
        d.add("머신 러닝", "머신러닝")
        d.add("딥 러닝", "딥러닝")
        d.add("인공 지능", "인공지능")
        d.build()
        self.corrector = TextCorrector(d, max_workers=4)

    def test_single_text_sync(self):
        result = self.corrector.correct("머신 러닝과 딥 러닝은 인공 지능입니다.")
        assert result == "머신러닝과 딥러닝은 인공지능입니다."

    def test_multisentence_text(self):
        text = "오늘 머신 러닝 세션이 있습니다. 딥 러닝도 다룰 예정입니다."
        result = self.corrector.correct(text)
        assert "머신러닝" in result
        assert "딥러닝" in result

    def test_batch_sync(self):
        texts = [
            "머신 러닝 모델을 학습합니다.",
            "딥 러닝 네트워크를 구성합니다.",
            "인공 지능 연구를 진행합니다.",
        ]
        results = self.corrector.correct_batch(texts)
        assert len(results) == 3
        assert "머신러닝" in results[0]
        assert "딥러닝" in results[1]
        assert "인공지능" in results[2]

    def test_empty_input(self):
        assert self.corrector.correct("") == ""
        assert self.corrector.correct("   ") == "   "

    def test_no_match_unchanged(self):
        text = "변경사항이 없는 문장입니다."
        assert self.corrector.correct(text) == text

    def test_async_correct(self):
        async def run():
            return await self.corrector.correct_async("머신 러닝 파이프라인.")
        result = asyncio.run(run())
        assert result == "머신러닝 파이프라인."

    def test_concurrent_requests(self):
        """Simulate multiple concurrent API requests."""
        async def run():
            texts = [f"머신 러닝 태스크 {i}." for i in range(20)]
            tasks = [self.corrector.correct_async(t) for t in texts]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run())
        assert len(results) == 20
        assert all("머신러닝" in r for r in results)

    def test_context_manager(self):
        d = TermDictionary()
        d.add("테스트", "TEST")
        d.build()
        with TextCorrector(d) as c:
            assert c.correct("테스트 문장") == "TEST 문장"

    def test_throughput_baseline(self):
        """Sanity-check: 100 parallel corrections complete in under 5 seconds."""
        async def run():
            texts = ["머신 러닝과 딥 러닝은 인공 지능의 핵심입니다." * 10] * 100
            return await self.corrector.correct_batch_async(texts)

        start = time.perf_counter()
        asyncio.run(run())
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Throughput too slow: {elapsed:.2f}s for 100 texts"
