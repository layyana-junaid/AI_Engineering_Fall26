from src.tokenizer import UNK_ID, CharTokenizer


def test_roundtrip_and_determinism():
    text = "To be, or not to be:\nthat is the question."
    a, b = CharTokenizer.fit(text), CharTokenizer.fit(text[::-1])
    assert a.chars == b.chars                      # sorted -> independent of corpus order
    assert a.decode(a.encode(text)) == text
    assert a.vocab_size == len(set(text)) + 1      # + <unk>


def test_unknown_characters_map_to_unk():
    t = CharTokenizer.fit("abc")
    ids = t.encode("abz")
    assert ids[2] == UNK_ID and t.decode(ids) == "ab?"


def test_serialisation():
    t = CharTokenizer.fit("hello world")
    assert CharTokenizer.from_dict(t.to_dict()).encode("hello") == t.encode("hello")
