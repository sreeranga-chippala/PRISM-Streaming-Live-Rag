from __future__ import annotations

ASSESSMENT_QUESTIONS = [
    {
        "id": "q1",
        "topic": "work_from_home",
        "question": (
            "What does the company's work-from-home policy say about "
            "employee eligibility, and what conditions must be satisfied "
            "before working remotely?"
        ),
    },
    {
        "id": "q2",
        "topic": "work_from_home_security",
        "question": (
            "If an employee works from home using a personal laptop, "
            "what security and data-protection requirements should they "
            "follow according to company policy?"
        ),
    },
    {
        "id": "q3",
        "topic": "leave",
        "question": (
            "What types of leave are available to employees, and what "
            "does the company policy specify about applying for and "
            "managing leave?"
        ),
    },
    {
        "id": "q4",
        "topic": "performance",
        "question": (
            "How does the company conduct performance reviews, and what "
            "does the policy say about the factors considered during "
            "employee evaluation?"
        ),
    },
    {
        "id": "q5",
        "topic": "travel_expense",
        "question": (
            "If an employee travels for business, what expenses can be "
            "claimed and what requirements does the company policy place "
            "on submitting those expenses?"
        ),
    },
    {
        "id": "q6",
        "topic": "code_of_conduct",
        "question": (
            "What does the company's code of conduct say about conflicts "
            "of interest and the professional responsibilities of employees?"
        ),
    },
    {
        "id": "q7",
        "topic": "it_security",
        "question": (
            "What rules does the company provide for protecting "
            "confidential company information and using company IT resources?"
        ),
    },
    {
        "id": "q8",
        "topic": "sexual_harassment",
        "question": (
            "What does the company's policy say about reporting and "
            "handling workplace sexual-harassment concerns?"
        ),
    },
    {
        "id": "q9",
        "topic": "onboarding_separation",
        "question": (
            "What procedures does the company define for employee "
            "onboarding and separation from the organization?"
        ),
    },
    {
        "id": "q10",
        "topic": "compensation_benefits",
        "question": (
            "What benefits and compensation-related provisions are "
            "described in the company's policy documents?"
        ),
    },
]


def get_question_for_elapsed_time(elapsed_seconds: float) -> dict:
    """
    Select an assessment question according to elapsed session time.

    Each question is displayed for approximately 2 minutes.
    After the final question, the sequence wraps around.
    """
    interval_seconds = 120
    index = int(max(0, elapsed_seconds) // interval_seconds)
    return ASSESSMENT_QUESTIONS[index % len(ASSESSMENT_QUESTIONS)]
