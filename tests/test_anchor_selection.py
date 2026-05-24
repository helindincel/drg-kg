from drg.extract import _select_anchor_entities


def test_select_anchor_entities_downweights_generic_high_df():
    # "Company" appears in many chunks (df high) while "Gigafactory" appears in few (df low).
    chunk_text = "The company built the Gigafactory. The company expanded the Gigafactory."
    chunk_entities = [("company", "Company"), ("Gigafactory", "Facility")]
    entity_to_chunks = {
        "company": list(range(10)),  # appears everywhere
        "gigafactory": [1, 2],  # appears in few chunks
    }

    anchors = _select_anchor_entities(
        chunk_text=chunk_text,
        chunk_entities=chunk_entities,
        entity_to_chunks=entity_to_chunks,
        total_chunks=10,
        min_anchor_len=3,
        max_anchors=1,
    )

    assert anchors == ["Gigafactory"]
