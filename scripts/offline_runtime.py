"""Prefer prepared local resources without restricting API connections."""
import os


def configure(_urls=()):
    os.environ.pop('DATASET_SOURCE', None)
    # API routing, DNS, proxies and redirects remain under deployment control.
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE',
                'HF_HUB_DISABLE_TELEMETRY', 'DO_NOT_TRACK'):
        os.environ[key] = '1'
