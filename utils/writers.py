class StatsWriter:
    def __init__(self, filepath: str, keys: list[str]) -> None:
        self.filepath = filepath
        self.keys = keys
        self.file = None

    def __enter__(self):
        self.file = open(self.filepath, "w")
        self.file.write(",".join(self.keys) + "\n")
        self.file.flush()
        return self

    def write(self, values: dict[str, float]) -> None:
        self.file.write(",".join(str(values[key]) for key in self.keys) + "\n")
        self.file.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
