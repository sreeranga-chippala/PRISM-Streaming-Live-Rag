from src.intelligence.intent_detector import (
    IntentDetector,
    IntentType,
)


def test_information_intent():
    detector = IntentDetector()

    result = detector.detect(
        "What is the cancellation policy?"
    )

    assert result.intent == IntentType.INFORMATION
    assert result.confidence >= 0.8


def test_action_intent():
    detector = IntentDetector()

    result = detector.detect(
        "I need to plan a customer workshop in Pune"
    )

    assert result.intent == IntentType.ACTION


def test_comparison_intent():
    detector = IntentDetector()

    result = detector.detect(
        "Compare the cancellation policies of these two venues"
    )

    assert result.intent == IntentType.COMPARISON


def test_summary_intent():
    detector = IntentDetector()

    result = detector.detect(
        "Summarize the key points"
    )

    assert result.intent == IntentType.SUMMARY


def test_clarification_intent():
    detector = IntentDetector()

    result = detector.detect(
        "Can you clarify what that means?"
    )

    assert result.intent == IntentType.CLARIFICATION


def test_empty_query_rejected():
    detector = IntentDetector()

    try:
        detector.detect("   ")
        assert False
    except ValueError:
        assert True