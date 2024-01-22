from zenml import step, pipeline

@step
def load_data() -> dict:
    """Loads data

    Returns:
        dict: Simulated training data alongside their labels
    """
    training_data = [[1,4], [4,7], [2,5], [9, 3]]
    labels = [2, 1, 7]

    return {"features": training_data, "labels": labels}

@step
def train_model(data: dict) -> None:
    """Simulates a 'training' model phase in a real ML scenario.
    """
    features = sum(map(sum, (data["features"])))
    labels = sum(data["labels"])

    print(f"{len(data['features'])} data points used to train the model.\nTotal features: {features}, Total labels: {labels}")


@pipeline
def simple_pipeline():
    """Simulates a real-world pipeline; how it combines various steps.
    """
    data = load_data()
    trained_model = train_model(data)


if __name__ == "__main__":
    run = simple_pipeline()
