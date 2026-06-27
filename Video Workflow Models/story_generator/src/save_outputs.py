from __future__ import annotations
import json
import os



def _char_names(scene: dict) -> list:
    result = []
    for entry in scene.get("characters_present", []):
        if isinstance(entry, dict):
            result.append(entry["name"])
        elif isinstance(entry, str):
            result.append(entry)
    return result


def _scene_positions(scene: dict) -> dict:
    positions = {}
    for entry in scene.get("characters_present", []):
        if isinstance(entry, dict):
            positions[entry["name"]] = entry.get("position", {})
        elif isinstance(entry, str):
            positions[entry] = {}
    return positions


def _write(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 3YNO voice description (kept in sync with generator.py constants)
# ---------------------------------------------------------------------------
_ZYNO_VOICE_DESCRIPTION = (
    "Orus A warm and wise male voice with a friendly, enthusiastic tone, "
    "speaking with clear storytelling energy and calm confidence. "
    "The voice is inviting and encouraging, making young children feel "
    "excited and safe to learn."
)

_ZYNO_VIDEO_PROMPT = (
    "A pure white background with a friendly cartoon black crow character standing at center frame. "
    "The crow is the only element in the scene -- no environment, no props, no scenery. "
    "The crow speaks directly to camera, its beak opening and closing in perfect synchronization "
    "with the voice audio. The crow's golden eyes blink expressively and its body animates "
    "naturally and warmly as it talks. "
    "No text, no words, no letters, no captions, no labels of any kind appear anywhere in the scene."
)

_ZYNO_NEGATIVE_PROMPT = (
    "text, words, letters, captions, subtitles, typography, font, label, title, watermark, "
    "writing, inscription, alphabets, numbers, digits, overlay text, on-screen text, "
    "speech bubble, dialogue box, low quality, blurry, pixelated, distorted, deformed, "
    "ugly, bad anatomy, duplicate, error, cropped, out of frame, worst quality, "
    "jpeg artifacts, overexposed, underexposed, background scenery, landscape, nature, "
    "sky, ground, grass, trees, buildings, clouds"
)


def _make_host_scene_id(host: dict) -> str:
    """Return a unique, filesystem-safe scene-id string for a 3YNO host scene."""
    if host["type"] == "intro":
        return "host_intro"
    elif host["type"] == "outro":
        return "host_outro"
    else:  # transition
        return f"host_trans_{host['after_scene']}_{host['insert_before']}"


def save_all(result: dict, out_dir: str = "output"):
    characters   = result["characters"]
    backgrounds  = result["backgrounds"]
    scripts      = result["voice_scripts"]
    host_scenes  = result.get("host_scenes", [])   # list of 3YNO host-scene dicts

    chars_folder  = os.path.join(out_dir, "characters")
    bgs_folder    = os.path.join(out_dir, "backgrounds")
    voices_folder = os.path.join(out_dir, "voices")

    os.makedirs(chars_folder,  exist_ok=True)
    os.makedirs(bgs_folder,    exist_ok=True)
    os.makedirs(voices_folder, exist_ok=True)

    # ── characters.json ──────────────────────────────────────────────
    characters_json = []
    for c in characters:
        name = c["name"]
        characters_json.append({
            "name":        name,
            "description": c.get("visual_description", ""),
            "output_path": f"characters/{name}.png",
        })

    # NOTE: 3YNO is intentionally NOT added to characters.json.
    # The fixed 3YNO.png is bundled directly into the output zip by the API
    # (via shutil.copy in story_generator/API/API.py), so the image generation
    # service (flux) never sees it and never tries to generate it.

    _write(os.path.join(out_dir, "characters.json"), characters_json)
    # ── backgrounds.json ─────────────────────────────────────────────
    backgrounds_json = []
    for b in backgrounds:
        name = b["name"]
        backgrounds_json.append({
            "name":        name,
            "description": b.get("visual_description", ""),
            "output_path": f"backgrounds/{name}.png",
        })

    _write(os.path.join(out_dir, "backgrounds.json"), backgrounds_json)

    # ── voices.json ──────────────────────────────────────────────────
    # Regular scene voices
    voices_json = []
    for scene in scripts:
        scene_id = scene["scene_number"]
        for shot_number, line in enumerate(scene["script"], start=1):
            voice_name  = f"{shot_number}-{scene_id}"
            output_path = f"voices/{voice_name}.wav"
            voices_json.append({
                "shot_id":     shot_number,
                "scene_id":    scene_id,
                "text":        line.get("text", ""),
                "description": line.get("voice_description", ""),
                "name":        voice_name,
                "output_path": output_path,
            })

    # 3YNO host voices (one per host scene)
    host_voice_map: dict[str, str] = {}   # scene_id -> output_path
    for host in host_scenes:
        hid          = _make_host_scene_id(host)
        voice_name   = hid.replace("_", "-")      # e.g. "host-intro"
        output_path  = f"voices/{voice_name}.wav"
        voices_json.append({
            "shot_id":     1,
            "scene_id":    hid,
            "text":        host.get("text", ""),
            "description": _ZYNO_VOICE_DESCRIPTION,
            "name":        voice_name,
            "output_path": output_path,
            "is_host":     True,
        })
        host_voice_map[hid] = output_path

    _write(os.path.join(out_dir, "voices.json"), voices_json)

    # ── shots_flow.json ──────────────────────────────────────────────
    # Build lookup: scene_number -> voice_path for regular scenes
    regular_voice_lookup: dict[tuple, str] = {}
    for scene in scripts:
        scene_id = scene["scene_number"]
        for shot_number, _ in enumerate(scene["script"], start=1):
            regular_voice_lookup[(scene_id, shot_number)] = (
                f"voices/{shot_number}-{scene_id}.wav"
            )

    # Build regular scenes list (keyed by scene_number)
    regular_scene_by_num: dict[int, dict] = {}
    for scene in scripts:
        scene_id  = scene["scene_number"]
        bg_name   = scene["background"]
        scene_pos = _scene_positions(scene)

        chars_in_scene = []
        for name in _char_names(scene):
            pos = scene_pos.get(name, {})
            chars_in_scene.append({
                "name":     name,
                "path":     f"characters/{name}.png",
                "position": {
                    "x": pos.get("x", 0.5),
                    "y": pos.get("y", 0.5),
                },
            })

        shots_list = []
        for shot_number, line in enumerate(scene["script"], start=1):
            vpath = regular_voice_lookup.get(
                (scene_id, shot_number),
                f"voices/{shot_number}-{scene_id}.wav"
            )
            shots_list.append({
                "shot_id":         shot_number,
                "name":            f"{shot_number}-{scene_id}",
                "speaker":         line.get("speaker", ""),
                "voice_path":      vpath,
                "video_prompt":    line.get("video_prompt", ""),
                "negative_prompt": line.get("negative_prompt", ""),
            })

        regular_scene_by_num[scene_id] = {
            "scene_id":      scene_id,
            "title":         scene["title"],
            "background":    f"backgrounds/{bg_name}.png",
            "is_host_scene": False,
            "characters":    chars_in_scene,
            "shots":         shots_list,
        }

    # Helper: build a 3YNO host scene entry for shots_flow
    def _build_host_entry(host: dict) -> dict:
        hid        = _make_host_scene_id(host)
        voice_path = host_voice_map.get(hid, f"voices/{hid.replace('_', '-')}.wav")
        label      = {
            "intro":      "3YNO: Introduction",
            "transition": f"3YNO: Transition (scene {host.get('after_scene')} \u2192 {host.get('insert_before')})",
            "outro":      "3YNO: Farewell",
        }.get(host["type"], "3YNO: Host Scene")

        return {
            "scene_id":      hid,
            "title":         label,
            "background":    None,          # 3YNO uses white background (fixed image)
            "is_host_scene": True,
            "characters": [
                {
                    "name":     "3YNO",
                    "path":     "characters/3YNO.png",
                    "position": {"x": 0.5, "y": 0.5},
                }
            ],
            "shots": [
                {
                    "shot_id":         1,
                    "name":            hid.replace("_", "-"),
                    "speaker":         "3YNO",
                    "voice_path":      voice_path,
                    "video_prompt":    _ZYNO_VIDEO_PROMPT,
                    "negative_prompt": _ZYNO_NEGATIVE_PROMPT,
                    "is_host_scene":   True,
                }
            ],
        }

    # Interleave host scenes with regular scenes in the correct order:
    #   intro → scene 1 → transition(1→2) → scene 2 → ... → scene N → outro
    regular_scene_nums = sorted(regular_scene_by_num.keys())

    # Index host scenes by type for easy lookup
    intro_host      = next((h for h in host_scenes if h["type"] == "intro"),      None)
    outro_host      = next((h for h in host_scenes if h["type"] == "outro"),      None)
    transition_hosts = {
        h["insert_before"]: h
        for h in host_scenes
        if h["type"] == "transition"
    }

    scenes_flow = []

    # 1. Intro
    if intro_host:
        scenes_flow.append(_build_host_entry(intro_host))

    # 2. Regular scenes, each preceded by its transition (except the first)
    for idx, scene_num in enumerate(regular_scene_nums):
        # Transition before this scene (if any)
        if idx > 0 and scene_num in transition_hosts:
            scenes_flow.append(_build_host_entry(transition_hosts[scene_num]))
        # Regular scene
        scenes_flow.append(regular_scene_by_num[scene_num])

    # 3. Outro
    if outro_host:
        scenes_flow.append(_build_host_entry(outro_host))

    _write(os.path.join(out_dir, "shots_flow.json"), {"scenes": scenes_flow})

    # ── Summary ──────────────────────────────────────────────────────
    total_shots  = sum(len(s["script"]) for s in scripts)
    total_shots += len(host_scenes)   # each host scene has exactly 1 shot

    print(f"\nTotal regular scenes : {len(scripts)}")
    print(f"Total host scenes    : {len(host_scenes)}")
    print(f"Total scenes in flow : {len(scenes_flow)}")
    print(f"Total shots          : {total_shots}")
    print(f"Output dir           : {out_dir}/")
    print("\nFiles created:")
    print(f"  {out_dir}/characters.json  ({len(characters_json)} entries, incl. 3YNO)")
    print(f"  {out_dir}/backgrounds.json ({len(backgrounds_json)} backgrounds)")
    print(f"  {out_dir}/voices.json      ({len(voices_json)} voice entries)")
    print(f"  {out_dir}/shots_flow.json  ({len(scenes_flow)} scenes)")
    print("\nFolders ready:")
    print(f"  {out_dir}/characters/")
    print(f"  {out_dir}/backgrounds/")
    print(f"  {out_dir}/voices/")
    print("\nNOTE: Place your fixed 3YNO image at characters/3YNO.png before running the video generator.")