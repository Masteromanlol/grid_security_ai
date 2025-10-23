def test_compare_and_register(tmp_path):
    import numpy as np
    from sklearn.dummy import DummyClassifier

    from grid_ai.ml.registry import ModelRegistry
    from grid_ai.ml.model_selection import compare_models

    # Create toy data
    X = np.random.RandomState(0).rand(100, 5)
    y = np.random.RandomState(1).randint(0, 2, size=100)

    estimators = [("dummy", DummyClassifier(strategy="most_frequent"))]
    results = compare_models(estimators, X, y, cv=3)
    assert isinstance(results, list)
    assert results[0]["name"] == "dummy"

    # Test registry
    reg = ModelRegistry(repo_path=str(tmp_path / "reg"))
    model_path = reg.register(estimators[0][1], "dummy_model", metadata={"score": results[0]["mean_score"]})
    assert model_path.exists()
    loaded = reg.load(reg.list()[0]["id"])
    assert loaded is not None
