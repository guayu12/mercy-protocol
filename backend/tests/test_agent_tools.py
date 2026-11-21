from core.agent.graph import create_local_directory


def test_create_local_directory_rejects_paths_outside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    monkeypatch.setenv("AURA_WORKSPACE_DIR", str(workspace))

    result = create_local_directory.invoke({"dir_name": "../outside"})

    assert result.startswith("Failed to create directory")
    assert workspace.is_dir()
    assert not outside.exists()
