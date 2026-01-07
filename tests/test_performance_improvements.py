"""Test performance improvements in citation modules."""
import sys
import os
import time

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_match_citation_cut_method():
    """Test the optimized cut() method in MatchCitation."""
    from trustrag.modules.citation.match_citation import MatchCitation
    
    mc = MatchCitation()
    
    # Test with a simple Chinese text
    test_text = "这是第一句话。这是第二句话！这是第三句话？"
    result = mc.cut(test_text)
    
    assert len(result) == 3, f"Expected 3 sentences, got {len(result)}"
    assert result[0] == "这是第一句话。"
    assert result[1] == "这是第二句话！"
    assert result[2] == "这是第三句话？"
    print("✓ MatchCitation.cut() test passed")


def test_source_citation_cut_method():
    """Test the optimized cut() method in SourceCitation."""
    from trustrag.modules.citation.source_citation import SourceCitation
    
    sc = SourceCitation()
    
    # Test with a simple Chinese text
    test_text = "这是第一句话。这是第二句话！这是第三句话？"
    result = sc.cut(test_text)
    
    assert len(result) == 3, f"Expected 3 sentences, got {len(result)}"
    assert result[0] == "这是第一句话。"
    assert result[1] == "这是第二句话！"
    assert result[2] == "这是第三句话？"
    print("✓ SourceCitation.cut() test passed")


def test_cut_with_quotes():
    """Test cut() method with quotes."""
    from trustrag.modules.citation.match_citation import MatchCitation
    
    mc = MatchCitation()
    
    # Test with quotes - should NOT split inside quotes
    test_text = '他说："这是一句话。包含句号。"然后继续说。'
    result = mc.cut(test_text)
    
    # The sentence should not be split inside quotes, so we expect 1 sentence
    # because the periods inside quotes don't trigger sentence splitting
    assert len(result) == 1, f"Expected 1 sentence (no split inside quotes), got {len(result)}: {result}"
    
    # Test without quotes - should split
    test_text2 = '这是第一句。这是第二句。'
    result2 = mc.cut(test_text2)
    assert len(result2) == 2, f"Expected 2 sentences, got {len(result2)}: {result2}"
    
    print("✓ Quote handling test passed")


def test_remove_stopwords():
    """Test the optimized remove_stopwords() method."""
    from trustrag.modules.citation.match_citation import MatchCitation
    
    mc = MatchCitation()
    
    test_text = "这是的一个的测试的"
    result = mc.remove_stopwords(test_text)
    
    # Should remove all instances of "的"
    assert "的" not in result or result.count("的") == 0, f"Stopwords not properly removed: {result}"
    print("✓ Stopwords removal test passed")


def test_convert_to_chinese():
    """Test the optimized convert_to_chinese() method."""
    from trustrag.modules.citation.source_citation import SourceCitation
    
    sc = SourceCitation()
    
    # Test various numbers
    assert sc.convert_to_chinese("0") == "零"
    assert sc.convert_to_chinese("1") == "一"
    assert sc.convert_to_chinese("10") == "十"
    assert sc.convert_to_chinese("11") == "十一"
    assert sc.convert_to_chinese("20") == "二十"
    assert sc.convert_to_chinese("25") == "二十五"
    assert sc.convert_to_chinese("99") == "九十九"
    print("✓ Chinese number conversion test passed")


def test_performance_cut_method():
    """Test the performance improvement of cut() method."""
    from trustrag.modules.citation.match_citation import MatchCitation
    
    mc = MatchCitation()
    
    # Create a large test text
    test_text = "这是一句话。" * 1000
    
    start_time = time.time()
    result = mc.cut(test_text)
    elapsed_time = time.time() - start_time
    
    assert len(result) == 1000, f"Expected 1000 sentences, got {len(result)}"
    print(f"✓ Performance test passed: cut() processed 1000 sentences in {elapsed_time:.4f}s")
    
    # Should complete in reasonable time (< 1 second for 1000 sentences)
    assert elapsed_time < 1.0, f"Performance issue: took {elapsed_time:.4f}s (expected < 1.0s)"


def test_excel_parser():
    """Test that ExcelParser still works correctly after optimization."""
    from trustrag.modules.document.excel_parser import ExcelParser
    
    parser = ExcelParser()
    
    # Test that the class can be instantiated
    assert parser is not None
    print("✓ ExcelParser initialization test passed")


if __name__ == "__main__":
    print("Running performance improvement tests...\n")
    
    try:
        test_match_citation_cut_method()
        test_source_citation_cut_method()
        test_cut_with_quotes()
        test_remove_stopwords()
        test_convert_to_chinese()
        test_performance_cut_method()
        test_excel_parser()
        
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
