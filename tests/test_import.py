
# tests/test_import.py
def test_import():
    import cryomodel
    assert hasattr(cryomodel, "__version__")
