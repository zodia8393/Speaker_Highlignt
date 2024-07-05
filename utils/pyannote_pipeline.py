import torch
from pyannote.audio import Pipeline

def load_pyannote_pipeline(token):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline.to(device)
    return pipeline
