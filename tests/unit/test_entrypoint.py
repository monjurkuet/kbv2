import knowledge_base
import inspect

def test_exported_main():
    """Verify that main is exported and is a function."""
    assert hasattr(knowledge_base, "main"), "knowledge_base should have 'main' attribute"
    assert callable(knowledge_base.main), "knowledge_base.main should be callable"
    print("✓ knowledge_base.main is exported and callable.")

def test_docstring_style():
    """Verify that the docstring follows Google style (has sections)."""
    doc = knowledge_base.main.__doc__
    assert doc is not None, "main() should have a docstring"
    assert "Returns:" in doc, "Docstring should have 'Returns:' section (Google style)"
    print("✓ main() docstring follows Google style requirements.")

if __name__ == "__main__":
    try:
        test_exported_main()
        test_docstring_style()
        print("\nAll entry point verification tests passed!")
    except AssertionError as e:
        print(f"\nVerification FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
