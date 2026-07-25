def test_cli_import():
    try:
        import app.cli
        # If it imports without raising ModuleNotFoundError, it passes.
        assert app.cli is not None
    except ModuleNotFoundError as e:
        assert False, f"ModuleNotFoundError raised during import of app.cli: {e}"
