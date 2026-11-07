import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def composer(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    package = "k2cr_composer_test"
    spec = importlib.util.spec_from_file_location(
        package, root / "__init__.py", submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package, module)
    spec.loader.exec_module(module)
    node_module = sys.modules[f"{package}.nodes"]
    management = ModuleType("comfy.model_management")
    management.intermediate_device = lambda: torch.device("cpu")
    management.intermediate_dtype = lambda: torch.float32
    comfy = ModuleType("comfy")
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    sessions = []
    monkeypatch.setattr(node_module, "install_router_wrapper", lambda model, session: sessions.append(session))
    monkeypatch.setattr(
        node_module, "load_region_adapters",
        lambda model, region, strict: SimpleNamespace(adapters={}, skipped_keys=[], unsupported=[]),
    )
    monkeypatch.setattr(
        node_module, "find_krea2_concept_positions",
        lambda clip, tokens, concepts, fallbacks: (
            [[index] if text else [] for index, text in enumerate(concepts)], concepts,
        ),
    )
    model = SimpleNamespace(clone=lambda: object())
    clip = SimpleNamespace(tokenize=lambda prompt: prompt, encode_from_tokens_scheduled=lambda tokens: [])

    def apply(canvases, characters=None, **extra_inputs):
        return node_module.Krea2CharacterRouter().apply(
            model=model, clip=clip, scene_prompt="A city", width=256, height=256,
            regions_json=json.dumps(
                [{"lora": "person", "trigger": "PERSON", "x": 0, "w": 0.5}]
                if characters is None else characters
            ),
            feather=0.0, overlap_policy="nearest", schedule_softness=0.0,
            strict=True, debug=False, supersample_scale=1.0,
            canvas_lora_json=json.dumps(canvases),
            **extra_inputs,
        )

    return apply, sessions, node_module


@pytest.mark.parametrize("canvases", [
    {"enabled": True, "lora": "legacy", "coverage": "global"},
    [
        {"enabled": True, "lora": "paint", "trigger": "PAINT", "coverage": "global", "start": 0.2},
        {"enabled": False, "lora": "disabled", "trigger": "DISABLED"},
        {"enabled": True, "lora": "zero", "trigger": "ZERO", "strength": 0},
        {"enabled": True, "lora": "grain", "prompt": "paper grain", "coverage": "unboxed", "end": 0.8},
        {"enabled": True, "lora": "lighting", "coverage": "global", "strength": -0.5},
    ],
    [{"enabled": True, "lora": f"style-{index}", "coverage": "global"} for index in range(8)],
    [],
])
def test_composer_routes_and_exports_all_canvas_loras(composer, canvases):
    apply, sessions, node_module = composer
    output = apply(canvases)
    parsed, _ = node_module.parse_canvas_loras(json.dumps(canvases))
    active = [(index, canvas) for index, canvas in enumerate(parsed)
              if canvas.enabled and canvas.lora != "None" and canvas.strength != 0]
    session = sessions[0]
    assert [region.lora for region in session.regions] == ["person", *[canvas.lora for _, canvas in active]]
    assert session.mask_modes == ["region", *[canvas.coverage for _, canvas in active]]
    assert len(session.loaded) == len(session.token_positions) == 1 + len(active)
    assert [(region.start, region.end) for region in session.regions[1:]] == [
        (canvas.start, canvas.end) for _, canvas in active
    ]
    assert len({region.name for region in session.regions}) == len(session.regions)
    diagnostics = json.loads(output[4])
    assert [entry["mask_mode"] for entry in diagnostics["regions"]] == session.mask_modes
    assert "DISABLED" not in output[2]
    assert "ZERO" not in output[2]
    shared = json.loads(output[6])
    restored, _ = node_module.parse_canvas_loras(json.dumps(shared["canvas_loras"]))
    assert restored == parsed
    assert shared["format"] == "krea2_character_router_share_v1"
    assert shared["canvas_lora"] == (shared["canvas_loras"][0] if parsed else {"enabled": False})


def test_composer_supports_multiple_canvas_loras_without_active_characters(composer):
    apply, sessions, _ = composer
    output = apply([
        {"enabled": True, "lora": "paint", "coverage": "global"},
        {"enabled": True, "lora": "texture", "coverage": "unboxed"},
    ], characters=[])
    assert sessions[0].mask_modes == ["global", "unboxed"]
    assert sessions[0].token_positions == [[], []]
    assert output[2] == "A city."


def test_connected_scene_prompt_text_replaces_the_scene_prompt_field(composer):
    apply, _, _ = composer
    output = apply([], scene_prompt_text="A beach at sunset")
    assert output[2].startswith("A beach at sunset")
    assert "A city" not in output[2]
    assert json.loads(output[6])["scene_prompt"] == "A beach at sunset"


def test_blank_scene_prompt_text_falls_back_to_the_field_with_a_warning(composer):
    apply, _, _ = composer
    output = apply([], scene_prompt_text="   ")
    assert output[2].startswith("A city")
    assert any("scene_prompt_text" in warning for warning in json.loads(output[4])["warnings"])


def test_scene_prompt_text_is_an_optional_socket_that_keeps_widget_order(composer):
    _, _, node_module = composer
    inputs = node_module.Krea2CharacterRouter.INPUT_TYPES()
    assert list(inputs["required"])[-1] == "canvas_lora_json"
    assert inputs["optional"]["scene_prompt_text"] == ("STRING", {"forceInput": True, "tooltip": (
        "Optional. When connected and not empty, replaces the Scene prompt field "
        "(e.g. from a prompt enhancer or Prompt Manager)."
    )})


def test_connected_scene_prompt_text_keeps_its_line_breaks_in_the_share_json(composer):
    apply, _, _ = composer
    output = apply([], scene_prompt_text="  A beach at sunset.\n\nWarm rim light.  ")
    assert json.loads(output[6])["scene_prompt"] == "A beach at sunset.\n\nWarm rim light."


TWO_ROWS = [
    {"name": "A", "lora": "row-a", "trigger": "AAA", "x": 0, "w": 0.5},
    {"name": "B", "lora": "row-b", "trigger": "BBB", "x": 0.5, "w": 0.5, "strength": 0.7},
]


def test_connected_lora_inputs_replace_each_rows_lora_and_strength(composer):
    apply, sessions, _ = composer
    output = apply([], characters=TWO_ROWS, lora_2={"lora": "picked/b.safetensors", "strength": 1.2})
    assert [(r.lora, r.strength) for r in sessions[0].regions] == [("row-a", 1.0), ("picked/b.safetensors", 1.2)]
    shared = json.loads(output[6])["characters"]
    assert [(c["lora"], c["strength"]) for c in shared] == [("row-a", 1.0), ("picked/b.safetensors", 1.2)]


def test_a_lora_input_without_a_matching_row_is_reported(composer):
    apply, sessions, _ = composer
    output = apply([], lora_3={"lora": "extra", "strength": 1.0})
    assert [r.lora for r in sessions[0].regions] == ["person"]
    assert any("lora_3" in warning for warning in json.loads(output[4])["warnings"])


def test_lora_inputs_are_optional_sockets_after_scene_prompt_text(composer):
    _, _, node_module = composer
    optional = node_module.Krea2CharacterRouter.INPUT_TYPES()["optional"]
    assert list(optional) == ["scene_prompt_text", "lora_1", "lora_2", "lora_3", "lora_4", "lora_5"]
    assert all(optional[f"lora_{i}"][0] == "KREA2_COMPOSER_LORA" for i in range(1, 6))


def test_composer_lora_node_lists_loras_and_outputs_the_selection(composer, monkeypatch):
    _, _, node_module = composer
    folder_paths = ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda kind: ["Krea/a.safetensors", "Krea/b.safetensors"] if kind == "loras" else []
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    node = node_module.Krea2ComposerLoRA
    required = node.INPUT_TYPES()["required"]
    assert required["lora_name"][0] == ["None", "Krea/a.safetensors", "Krea/b.safetensors"]
    assert required["strength"][1]["default"] == 1.0
    assert node().select("Krea/b.safetensors", 0.8) == ({"lora": "Krea/b.safetensors", "strength": 0.8},)


def test_row_warnings_describe_the_connected_lora_not_the_rows_own(composer):
    apply, _, _ = composer
    rows = [{"name": "A", "lora": "None", "trigger": "AAA", "x": 0, "w": 0.5}]
    warnings = json.loads(apply([], characters=rows, lora_1={"lora": "real", "strength": 3.0})[4])["warnings"]
    assert not any("no LoRA selected" in warning for warning in warnings)
    assert any("high LoRA strength (3.00)" in warning for warning in warnings)


def test_a_lora_input_for_a_disabled_row_is_reported(composer):
    apply, _, _ = composer
    rows = [*TWO_ROWS[:1], {**TWO_ROWS[1], "enabled": False}]
    warnings = json.loads(apply([], characters=rows, lora_2={"lora": "x", "strength": 1.0})[4])["warnings"]
    assert any("lora_2" in warning and "disabled" in warning for warning in warnings)
