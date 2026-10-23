"""Portable model archives. Load only model files from trusted sources."""
import io
import json
import tempfile
import zipfile
from pathlib import Path
import joblib


def export_bundle(bundle):
    output = io.BytesIO()
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory)
        meta = {k:bundle[k] for k in ['kind','columns','weather_columns']}
        if bundle['kind'] == 'sklearn':
            joblib.dump(bundle['model'],root/'pipeline.joblib')
        else:
            bundle['model'].save(root/'model.keras')
            joblib.dump(bundle['preprocess'],root/'preprocessing.joblib')
            meta['scale'] = bundle['scale']
        (root/'metadata.json').write_text(json.dumps(meta,indent=2))
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
            for file in root.iterdir():
                archive.write(file,file.name)
    return output.getvalue()


def load_bundle(directory):
    """Load an extracted, trusted archive for research.forecast."""
    root = Path(directory)
    meta = json.loads((root/'metadata.json').read_text())
    if meta['kind'] == 'sklearn':
        meta['model'] = joblib.load(root/'pipeline.joblib')
    else:
        import tensorflow as tf
        meta['model'] = tf.keras.models.load_model(root/'model.keras')
        meta['preprocess'] = joblib.load(root/'preprocessing.joblib')
    return meta
