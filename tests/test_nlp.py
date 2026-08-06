"""Tests for the deterministic NLP layer."""

from __future__ import annotations

import pytest

from app.nlp.grammar import apply_corrections, check_grammar
from app.nlp.rewrite import rewrite
from app.nlp.scoring import score_text
from app.nlp.textstats import analyze_text
from app.nlp.tone import analyze_tone


class TestTextStats:
    def test_counts_words_and_sentences(self):
        stats = analyze_text("This is one. This is two! And three?")
        assert stats.sentence_count == 3
        assert stats.word_count == 8

    def test_detects_hedges_and_fillers(self):
        stats = analyze_text("I just wanted to maybe check if you could possibly help.")
        assert "just" in stats.filler_words
        assert any("maybe" in h or "wondering" in h for h in stats.hedges) or stats.hedges

    def test_detects_passive_voice(self):
        stats = analyze_text("The report was reviewed and the changes were approved.")
        assert len(stats.passive_phrases) >= 2

    def test_ignores_adjectival_participles(self):
        # "I am interested" is not passive voice.
        stats = analyze_text("I am interested in the role and I am excited to apply.")
        assert stats.passive_phrases == []

    def test_detects_structure(self):
        stats = analyze_text("Hi Sam,\n\nCould you send the file by Friday?\n\nThanks")
        assert stats.has_greeting
        assert stats.has_call_to_action
        assert stats.has_signoff

    def test_empty_text_does_not_crash(self):
        stats = analyze_text("")
        assert stats.word_count == 0
        assert stats.flesch_reading_ease == 0.0


class TestGrammar:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("I recieved the email.", "received"),
            ("We should of told them.", "could have|should have"),
            ("Their going to the meeting.", None),
            ("i went to the office.", "I"),
            ("He are going home.", "is"),
            ("They is late again.", "are"),
            ("She have finished it.", "has"),
            ("This is a apple.", "an apple"),
        ],
    )
    def test_detects_issues(self, text, expected):
        issues = check_grammar(text)
        assert issues, f"expected an issue in {text!r}"
        if expected:
            joined = " ".join(i.suggestion for i in issues)
            assert any(part in joined for part in expected.split("|"))

    def test_clean_text_has_no_major_issues(self):
        text = "Thank you for sending the report. I will review it before Friday."
        assert [i for i in check_grammar(text) if i.severity == "major"] == []

    def test_autofix_produces_corrected_text(self):
        text = "i recieved the report tommorow"
        corrected = apply_corrections(text, check_grammar(text))
        assert "received" in corrected
        assert "tomorrow" in corrected
        assert corrected.startswith("I ")
        assert corrected.endswith(".")

    def test_capitalization_issue_reports_the_word_not_the_sentence(self):
        issues = check_grammar("hey there, this is a fairly long opening sentence.")
        capitals = [i for i in issues if i.type == "capitalization"]
        assert capitals and capitals[0].original == "hey"

    def test_its_contraction(self):
        issues = check_grammar("its been a long week and its not over.")
        assert any(i.type == "its_contraction" for i in issues)


class TestTone:
    def test_flags_aggressive_language(self, ):
        text = "You never deliver on time. This is unacceptable and it's your fault."
        stats = analyze_text(text)
        tone = analyze_tone(text, stats)
        assert tone.primary_tone == "aggressive"
        assert tone.politeness < 50
        assert tone.risks

    def test_recognises_empathy(self):
        text = (
            "I understand how frustrating this is, and I'm sorry to hear the order "
            "was late. Thank you for your patience while I sort it out."
        )
        stats = analyze_text(text)
        tone = analyze_tone(text, stats)
        assert tone.warmth > 60
        assert tone.sentiment == "positive"

    def test_formality_dimension(self):
        formal = "Dear Dr Patel,\n\nI am writing to request an extension.\n\nSincerely,"
        casual = "hey! yeah gonna be late lol, no worries right?"
        assert (
            analyze_tone(formal, analyze_text(formal)).formality
            > analyze_tone(casual, analyze_text(casual)).formality
        )

    def test_hedging_lowers_assertiveness(self):
        hedged = "I was just wondering if maybe you could possibly take a look?"
        direct = "I recommend we ship on Tuesday. Could you review the draft by Monday?"
        assert (
            analyze_tone(hedged, analyze_text(hedged)).assertiveness
            < analyze_tone(direct, analyze_text(direct)).assertiveness
        )


class TestScoring:
    def test_scores_are_bounded(self, hostile_message):
        stats = analyze_text(hostile_message)
        tone = analyze_tone(hostile_message, stats)
        score = score_text(stats, tone, check_grammar(hostile_message)).score
        for value in score.model_dump().values():
            assert 0 <= value <= 100

    def test_good_writing_beats_bad_writing(self, sample_email):
        good = (
            "Hi John,\n\nHave you had a chance to review the report I sent on "
            "Monday? The deadline moved to Thursday, so I need your sign-off by "
            "Wednesday 5pm.\n\nThanks,\nPriya"
        )

        def overall(text, context="general"):
            stats = analyze_text(text)
            tone = analyze_tone(text, stats)
            return score_text(stats, tone, check_grammar(text), context=context).overall

        assert overall(good) > overall(sample_email)

    def test_deterministic(self, sample_email):
        def overall():
            stats = analyze_text(sample_email)
            tone = analyze_tone(sample_email, stats)
            return score_text(stats, tone, check_grammar(sample_email)).overall

        assert overall() == overall() == overall()

    def test_interview_context_does_not_demand_a_call_to_action(self):
        answer = (
            "At my last company I owned the payments service. I rewrote the retry "
            "logic and failed transactions dropped from 4% to 0.6% in two months."
        )
        stats = analyze_text(answer)
        tone = analyze_tone(answer, stats)
        weaknesses = score_text(stats, tone, [], context="interview").weaknesses
        assert not any("call to action" in w for w in weaknesses)


class TestRewrite:
    def test_removes_hedging_and_fillers(self, sample_email):
        improved, changes = rewrite(sample_email, "professional")
        assert "just wanted to" not in improved.lower()
        assert changes

    def test_de_escalation_stays_grammatical(self, hostile_message):
        improved, _ = rewrite(hostile_message, "diplomatic")
        assert "you never" not in improved.lower()
        assert "your fault" not in improved.lower()
        # The old phrase-swap produced "I haven't yet seen send the numbers".
        assert "seen send" not in improved.lower()

    def test_improves_the_score(self, hostile_message):
        improved, _ = rewrite(hostile_message, "diplomatic")

        def overall(text):
            stats = analyze_text(text)
            tone = analyze_tone(text, stats)
            return score_text(stats, tone, check_grammar(text)).overall

        assert overall(improved) > overall(hostile_message)

    def test_preserves_content_words(self):
        text = "Please send the Q3 budget spreadsheet to Dana before the audit."
        improved, _ = rewrite(text, "professional")
        for keyword in ("Q3", "budget", "Dana", "audit"):
            assert keyword in improved

    def test_formal_target_expands_contractions(self):
        improved, _ = rewrite("I can't make it, I'm gonna be late.", "formal")
        assert "cannot" in improved
        assert "gonna" not in improved
