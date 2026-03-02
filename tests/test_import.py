
# tests/test_import.py
def test_import():
    import crymodel
    assert hasattr(crymodel, "__version__")
