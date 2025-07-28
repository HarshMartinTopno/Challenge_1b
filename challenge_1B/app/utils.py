try:
    import orjson
    def read_json(path: str):
        from pathlib import Path
        return orjson.loads(Path(path).read_bytes())

    def write_json(data, path: str):
        from pathlib import Path
        Path(path).write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
except ImportError:
    import json
    def read_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(data, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)