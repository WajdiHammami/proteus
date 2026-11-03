from src.domain import ExtractedRecord


def make_record(data=None) -> ExtractedRecord:
    if data is None:
        return ExtractedRecord(id="r1", document_id="d1", schema_id="sch1")
    return ExtractedRecord(id="r1", document_id="d1", schema_id="sch1", data=data)


def test_default_data_empty_and_not_shared():
    a, b = make_record(), make_record()
    assert a.data == {}
    a.data["x"] = 1
    assert b.data == {}


def test_is_complete_when_all_required_present():
    record = make_record({"title": "Shoe", "price": 9.99, "extra": 1})
    assert record.is_complete(["title", "price"])


def test_is_complete_false_when_missing():
    record = make_record({"title": "Shoe"})
    assert not record.is_complete(["title", "price"])


def test_is_complete_with_no_required_columns():
    assert make_record().is_complete([])


def test_is_complete_treats_none_value_as_missing():
    record = make_record({"title": None, "price": 0})
    assert not record.is_complete(["title"])


def test_is_complete_accepts_falsy_non_none_values():
    record = make_record({"price": 0, "in_stock": False, "name": ""})
    assert record.is_complete(["price", "in_stock", "name"])


def test_get_field():
    record = make_record({"title": "Shoe"})
    assert record.get_field("title") == "Shoe"
    assert record.get_field("missing") is None
