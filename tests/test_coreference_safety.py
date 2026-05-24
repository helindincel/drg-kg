from drg.coreference_resolution import resolve_coreferences


def test_coreference_ambiguous_pronoun_abstains():
    text = "Elon Musk and Tim Cook met. He spoke about iPhone."
    entities = [("Elon Musk", "Person"), ("Tim Cook", "Person"), ("iPhone", "Product")]
    relations = [("He", "spoke_about", "iPhone")]

    _, resolved = resolve_coreferences(
        text=text,
        entities=entities,
        relations=relations,
        use_nlp=False,  # force heuristic path for deterministic unit test
        use_neural_coref=False,
        embedding_provider=None,
    )
    # Must not guess between Elon vs Tim.
    assert ("He", "spoke_about", "iPhone") in resolved


def test_coreference_single_person_resolves():
    text = "Elon Musk founded Tesla. He founded SpaceX."
    entities = [("Elon Musk", "Person"), ("Tesla", "Company"), ("SpaceX", "Company")]
    relations = [("He", "founded", "SpaceX")]

    _, resolved = resolve_coreferences(
        text=text,
        entities=entities,
        relations=relations,
        use_nlp=False,
        use_neural_coref=False,
        embedding_provider=None,
    )
    assert ("Elon Musk", "founded", "SpaceX") in resolved


def test_coreference_turkish_o_single_person_resolves():
    text = "Elon Musk Tesla'yı kurdu. O daha sonra SpaceX'i de kurdu."
    entities = [("Elon Musk", "Person"), ("Tesla", "Company"), ("SpaceX", "Company")]
    relations = [("O", "founded", "SpaceX")]

    _, resolved = resolve_coreferences(
        text=text,
        entities=entities,
        relations=relations,
        use_nlp=False,
        use_neural_coref=False,
        embedding_provider=None,
        language="tr",
    )
    assert ("Elon Musk", "founded", "SpaceX") in resolved
