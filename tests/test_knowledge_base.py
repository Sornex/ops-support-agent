from ops_agent.knowledge_base import KnowledgeBase, tokenize


def test_loads_default_knowledge_base():
    kb = KnowledgeBase.from_file()
    assert len(kb) >= 5


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("How do I reset the VPN?") == {"reset", "vpn"}


def test_vpn_query_returns_vpn_entries_first():
    kb = KnowledgeBase.from_file()
    results = kb.search("my vpn will not connect")
    assert results
    assert results[0].entry.id.startswith("vpn-")


def test_password_query_finds_reset_entry():
    kb = KnowledgeBase.from_file()
    results = kb.search("I forgot my password")
    assert results[0].entry.id == "pwd-001"


def test_service_502_query_finds_service_entry():
    kb = KnowledgeBase.from_file()
    results = kb.search("internal website returns 502")
    assert results[0].entry.id == "svc-001"


def test_top_k_limits_results():
    kb = KnowledgeBase.from_file()
    assert len(kb.search("network connect vpn wifi", top_k=2)) <= 2


def test_gibberish_query_returns_nothing():
    kb = KnowledgeBase.from_file()
    assert kb.search("zzzqqq xxyyzz") == []


def test_empty_query_returns_nothing():
    kb = KnowledgeBase.from_file()
    assert kb.search("   ") == []
