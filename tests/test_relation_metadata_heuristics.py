from drg.extract import _infer_relation_metadata_heuristic


def test_negation_heuristic_marks_no_longer_produces():
    text = "Apple no longer produces Newton."
    relations = [("Apple", "produces", "Newton")]
    meta = _infer_relation_metadata_heuristic(text=text, relations=relations)
    assert meta["negations"] == [True]


def test_temporal_heuristic_extracts_single_year_start():
    text = "Steve Jobs founded Apple in 1976."
    relations = [("Steve Jobs", "founded", "Apple")]
    meta = _infer_relation_metadata_heuristic(text=text, relations=relations)
    assert meta["temporal_info"][0]["start"] == "1976"
    assert meta["temporal_info"][0]["precision"] == "year"


def test_negation_heuristic_abstains_if_negation_not_related_to_relation():
    text = "Apple is not in California. Apple produces Newton."
    relations = [("Apple", "produces", "Newton")]
    meta = _infer_relation_metadata_heuristic(text=text, relations=relations)
    assert meta["negations"] == [False]
