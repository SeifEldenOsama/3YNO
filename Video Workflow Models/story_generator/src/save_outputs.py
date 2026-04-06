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



def save_all(result: dict, out_dir: str = "output"):
    characters   = result["characters"]
    backgrounds  = result["backgrounds"]
    scripts      = result["voice_scripts"]

    chars_folder  = os.path.join(out_dir, "characters")
    bgs_folder    = os.path.join(out_dir, "backgrounds")
    voices_folder = os.path.join(out_dir, "voices")

    os.makedirs(chars_folder,  exist_ok=True)
    os.makedirs(bgs_folder,    exist_ok=True)
    os.makedirs(voices_folder, exist_ok=True)

    characters_json = []
    for c in characters:
        name = c["name"]
        characters_json.append({
            "name":        name,
            "description": c.get("visual_description", ""),
            "output_path": f"characters/{name}.png",
        })

    _write(os.path.join(out_dir, "characters.json"), characters_json)

    backgrounds_json = []
    for b in backgrounds:
        name = b["name"]
        backgrounds_json.append({
            "name":        name,
            "description": b.get("visual_description", ""),
            "output_path": f"backgrounds/{name}.png",
        })

    _write(os.path.join(out_dir, "backgrounds.json"), backgrounds_json)

    voices_json = []
    for scene in scripts:
        scene_id = scene["scene_number"]
        for shot_number, line in enumerate(scene["script"], start=1):
            voice_name  = f"{shot_number}-{scene_id}"
            output_path = f"voices/{voice_name}.mp3"
            voices_json.append({
                "shot_id":     shot_number,
                "scene_id":    scene_id,
                "text":        line.get("text", ""),
                "description": line.get("voice_description", ""),
                "name":        voice_name,
                "output_path": output_path,
            })

    _write(os.path.join(out_dir, "voices.json"), voices_json)

    voice_path_lookup = {}
    for v in voices_json:
        voice_path_lookup[(v["scene_id"], v["shot_id"])] = v["output_path"]

    scenes_flow = []
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
            vpath = voice_path_lookup.get(
                (scene_id, shot_number),
                f"voices/{shot_number}-{scene_id}.mp3"
            )
            shots_list.append({
                "shot_id":         shot_number,
                "name":            f"{shot_number}-{scene_id}",
                "speaker":         line.get("speaker", ""),
                "voice_path":      vpath,
                "video_prompt":    line.get("video_prompt", ""),
                "negative_prompt": line.get("negative_prompt", ""),
            })

        scenes_flow.append({
            "scene_id":   scene_id,
            "title":      scene["title"],
            "background": f"backgrounds/{bg_name}.png",
            "characters": chars_in_scene,
            "shots":      shots_list,
        })

    _write(os.path.join(out_dir, "shots_flow.json"), {"scenes": scenes_flow})

    total_shots = sum(len(s["script"]) for s in scripts)
    print(f"\nTotal scenes : {len(scripts)}")
    print(f"Total shots  : {total_shots}")
    print(f"Output dir   : {out_dir}/")
    print("\nFiles created:")
    print(f"  {out_dir}/characters.json  ({len(characters_json)} characters)")
    print(f"  {out_dir}/backgrounds.json ({len(backgrounds_json)} backgrounds)")
    print(f"  {out_dir}/voices.json      ({len(voices_json)} voice entries)")
    print(f"  {out_dir}/shots_flow.json  ({len(scenes_flow)} scenes)")
    print("\nFolders ready:")
    print(f"  {out_dir}/characters/")
    print(f"  {out_dir}/backgrounds/")
    print(f"  {out_dir}/voices/")