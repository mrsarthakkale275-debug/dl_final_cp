import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_tmp_root = Path(tempfile.mkdtemp(prefix="adaptiveinspect-test-"))
_empty_rgb = _tmp_root / "empty_rgb"
_empty_xray = _tmp_root / "empty_xray"
_image_dir = _tmp_root / "inspection_images"
_report_dir = _tmp_root / "reports"
for directory in (_empty_rgb, _empty_xray, _image_dir, _report_dir):
    directory.mkdir(parents=True, exist_ok=True)

os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_root / 'test.db'}"
os.environ["IMAGE_STORAGE_DIR"] = str(_image_dir)
os.environ["REPORT_STORAGE_DIR"] = str(_report_dir)
os.environ["RGB_MODELS_DIR"] = str(PROJECT_ROOT / "models" / "rgb")
os.environ["XRAY_MODELS_DIR"] = str(_empty_xray)
os.environ["MAX_UPLOAD_BYTES"] = "1024"
os.environ["DATA_DIR"] = str(PROJECT_ROOT / "data")
os.environ["CORS_ORIGINS"] = "http://testserver"
os.environ["CORS_ALLOW_ALL"] = "false"
os.environ["APP_DEBUG"] = "false"

import backend.app.main as main_module

EMPTY_RGB_MODELS_DIR = _empty_rgb


@pytest.fixture
def client(tmp_path):
    # Each test still uses a fresh temp directory by overriding runtime paths.
    import backend.app.core.config as config_module
    import backend.app.services.model_registry as registry_module

    db_path = tmp_path / "test.db"
    image_dir = tmp_path / "inspection_images"
    report_dir = tmp_path / "reports"
    image_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config_module.settings.database_url = f"sqlite:///{db_path}"
    config_module.settings.image_storage_dir = image_dir
    config_module.settings.report_storage_dir = report_dir
    config_module.settings.rgb_models_dir = PROJECT_ROOT / "models" / "rgb"
    config_module.settings.xray_models_dir = _empty_xray
    config_module.settings.max_upload_bytes = 1024
    config_module.settings.ensure_directories()

    # Reconnect the global engine/session to the temporary database.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        config_module.settings.database_url,
        connect_args={"check_same_thread": False},
        future=True,
    )
    config_module.settings.database_url = config_module.settings.database_url
    import backend.app.core.database as database_module
    database_module.engine = engine
    database_module.SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    database_module.Base.metadata.create_all(bind=engine)
    from backend.app.models import inspection as inspection_models
    for table in (inspection_models.Inspection.__table__, inspection_models.InspectionResult.__table__):
        table.create(bind=engine, checkfirst=True)

    # Ensure registered model state matches configured directories.
    if (PROJECT_ROOT / "models" / "rgb" / "anomaly_model.npz").exists():
        from backend.app.ml.rgb_engine import load_rgb_engine

        rgb_engine = load_rgb_engine()
        if rgb_engine is not None:
            registry_module.registry.register_rgb_engine(rgb_engine)
        else:
            registry_module.registry.rgb.ready = False
            registry_module.registry.rgb.status = "untrained"
            registry_module.registry.rgb.weights = []
            registry_module.registry.rgb_engine = None
    else:
        registry_module.registry.rgb.ready = False
        registry_module.registry.rgb.status = "untrained"
        registry_module.registry.rgb.weights = []
        registry_module.registry.rgb_engine = None
    registry_module.registry.xray.ready = False
    registry_module.registry.xray.status = "untrained"
    registry_module.registry.xray.weights = []
    registry_module.registry.xray_engine = None

    return TestClient(main_module.app)


@pytest.fixture
def rgb_client(tmp_path):
    client = client_fixture = None
    # Build a client using the same mechanism; RGB artifacts are available.
    import backend.app.core.config as config_module
    import backend.app.services.model_registry as registry_module
    from fastapi.testclient import TestClient

    db_path = tmp_path / "test.db"
    image_dir = tmp_path / "inspection_images"
    report_dir = tmp_path / "reports"
    image_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config_module.settings.database_url = f"sqlite:///{db_path}"
    config_module.settings.image_storage_dir = image_dir
    config_module.settings.report_storage_dir = report_dir
    config_module.settings.rgb_models_dir = PROJECT_ROOT / "models" / "rgb"
    config_module.settings.xray_models_dir = _empty_xray
    config_module.settings.max_upload_bytes = 1024
    config_module.settings.ensure_directories()

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    import backend.app.core.database as database_module
    database_module.engine = engine
    database_module.SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    database_module.Base.metadata.create_all(bind=engine)
    from backend.app.models import inspection as inspection_models
    for table in (inspection_models.Inspection.__table__, inspection_models.InspectionResult.__table__):
        table.create(bind=engine, checkfirst=True)

    from backend.app.ml.rgb_engine import load_rgb_engine
    rgb_engine = load_rgb_engine()
    if rgb_engine is not None:
        registry_module.registry.register_rgb_engine(rgb_engine)
    else:
        registry_module.registry.rgb.ready = False
        registry_module.registry.rgb.status = "untrained"
        registry_module.registry.rgb.weights = []
        registry_module.registry.rgb_engine = None

    return TestClient(main_module.app)
