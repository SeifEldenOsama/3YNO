from __future__ import annotations
import json
import os


def _scene_folder(scene_number: int, title: str) -> str:
    safe = (
        title.replace(" ", "_")
             .replace("'", "")
             .replace("/", "-")
             .replace(":", "")
    )
    return f"scene_{scene_number:02d}_{safe}"


def _estimate_duration(text: str) -> float:
    return round(len(text.split()) / 2.5, 1)


def _scene_positions(scene: dict) -> dict:
    positions = {}
    for entry in scene.get("characters_present", []):
        if isinstance(entry, dict):
            positions[entry["name"]] = entry.get("position", {})
        elif isinstance(entry, str):
            positions[entry] = {}
    return positions


def _char_names(scene: dict) -> list:
    result = []
    for entry in scene.get("characters_present", []):
        if isinstance(entry, dict):
            result.append(entry["name"])
        elif isinstance(entry, str):
            result.append(entry)
    return result


def _write(out_dir: str, filename: str, data: dict):
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")


def save_all(result: dict, out_dir: str = "outputs"):
    characters  = result["characters"]
    backgrounds = result["backgrounds"]
    scripts     = result["voice_scripts"]

    char_lookup = {c["name"]: c for c in characters}
    bg_lookup   = {b["name"]: b for b in backgrounds}

    os.makedirs(os.path.join(out_dir, "assets", "characters"),  exist_ok=True)
    os.makedirs(os.path.join(out_dir, "assets", "backgrounds"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "scenes"),                exist_ok=True)

    all_shots    = []
    global_order = 0

    for scene in scripts:
        sn        = scene["scene_number"]
        sf        = _scene_folder(sn, scene["title"])
        bg_name   = scene["background"]
        n_lines   = len(scene["script"])
        scene_pos = _scene_positions(scene)

        for i, line in enumerate(scene["script"], 1):
            global_order += 1
            speaker   = line.get("speaker", "")
            text      = line.get("text", "")
            is_first  = (i == 1)
            is_last   = (i == n_lines)
            shot_rel  = f"scenes/{sf}/shots/shot_{i:02d}_{speaker}"
            prev_shot = (
                f"scenes/{sf}/shots/shot_{i-1:02d}_{scene['script'][i-2].get('speaker','')}/clip.mp4"
                if i > 1 else None
            )

            all_shots.append({
                "order":             global_order,
                "shot_id":           f"s{sn:02d}_shot{i:02d}",
                "scene_number":      sn,
                "scene_title":       scene["title"],
                "shot_number":       i,
                "is_first_in_scene": is_first,
                "is_last_in_scene":  is_last,
                "background_name":   bg_name,
                "background_image":  f"assets/backgrounds/{bg_name}.png",
                "speaker":           speaker,
                "speaker_position":  scene_pos.get(speaker, {}),
                "speaker_image":     f"assets/characters/{speaker}.png",
                "text":              text,
                "voice_description": line.get("voice_description", ""),
                "video_prompt":      line.get("video_prompt", ""),
                "estimated_duration_sec": _estimate_duration(text),
                "voice_file":  f"{shot_rel}/voice.mp3",
                "frame_file":  f"{shot_rel}/frame.png",
                "clip_file":   f"{shot_rel}/clip.mp4",
                "frame_source":   "composite" if is_first else "previous_clip",
                "previous_clip":  prev_shot,
                "transition_hint": (
                    None
                    if is_first
                    else "extract last frame of previous clip → use as frame.png"
                ),
            })

    frames_to_composite = []
    frames_from_clip    = []

    for s in all_shots:
        if s["frame_source"] == "composite":
            frames_to_composite.append({
                "order":            s["order"],
                "shot_id":          s["shot_id"],
                "background_image": s["background_image"],
                "characters": [
                    {
                        "name":       name,
                        "image_file": f"assets/characters/{name}.png",
                        "position":   _scene_positions(
                                          scripts[s["scene_number"] - 1]
                                      ).get(name, {}),
                    }
                    for name in _char_names(scripts[s["scene_number"] - 1])
                    if name in char_lookup
                ],
                "output_file": s["frame_file"],
            })
        else:
            frames_from_clip.append({
                "order":         s["order"],
                "shot_id":       s["shot_id"],
                "previous_clip": s["previous_clip"],
                "instruction":   "extract last frame of previous_clip → save as output_file",
                "output_file":   s["frame_file"],
            })

    manifest = {
        "_description": (
            "Step 1: generate character PNGs. "
            "Step 2: generate background PNGs. "
            "Step 3: generate voice MP3s. "
            "Step 4a: composite frame.png for shot 1 of each scene (frames_to_composite). "
            "Step 4b: for all other shots, extract the last frame of the previous clip (frames_from_clip). "
            "Step 5: run LTX-2 video generation per shot using frame.png + voice.mp3."
        ),
        "characters_to_generate": [
            {
                "name":              c["name"],
                "gender":            c["gender"],
                "image_prompt":      c["visual_description"],
                "voice_description": c.get("voice_description", ""),
                "output_file":       f"assets/characters/{c['name']}.png",
            }
            for c in characters
        ],
        "backgrounds_to_generate": [
            {
                "name":           b["name"],
                "mood":           b["mood"],
                "lesson_context": b.get("lesson_context", ""),
                "image_prompt":   b["visual_description"],
                "output_file":    f"assets/backgrounds/{b['name']}.png",
            }
            for b in backgrounds
        ],
        "voices_to_generate": [
            {
                "order":             s["order"],
                "shot_id":           s["shot_id"],
                "speaker":           s["speaker"],
                "text":              s["text"],
                "voice_description": s["voice_description"],
                "estimated_duration_sec": s["estimated_duration_sec"],
                "output_file":       s["voice_file"],
            }
            for s in all_shots
        ],
        "frames_to_composite": frames_to_composite,
        "frames_from_clip":    frames_from_clip,
    }
    _write(out_dir, "00_generation_manifest.json", manifest)

    _write(out_dir, "video_timeline.json", {
        "_description": (
            "Flat ordered shot list. For each shot: "
            "if frame_source=composite → run frames_to_composite step. "
            "if frame_source=previous_clip → extract last frame of previous_clip into frame_file. "
            "Then run LTX-2 with frame_file + voice_file."
        ),
        "total_scenes": len(scripts),
        "total_shots":  len(all_shots),
        "shots":        all_shots,
    })

    story_scenes = []
    for scene in scripts:
        sn          = scene["scene_number"]
        bg_name     = scene["background"]
        bg_data     = bg_lookup.get(bg_name, {})
        scene_shots = [s for s in all_shots if s["scene_number"] == sn]
        scene_pos   = _scene_positions(scene)

        story_scenes.append({
            "scene_number":   sn,
            "title":          scene["title"],
            "folder":         f"scenes/{_scene_folder(sn, scene['title'])}",
            "lesson_element": scene["lesson_element"],
            "background": {
                "name":       bg_name,
                "mood":       bg_data.get("mood", ""),
                "image_file": f"assets/backgrounds/{bg_name}.png",
            },
            "characters_present": [
                {
                    "name":       name,
                    "image_file": f"assets/characters/{name}.png",
                    "position":   scene_pos.get(name, {}),
                }
                for name in _char_names(scene)
            ],
            "total_shots": len(scene_shots),
            "shots": scene_shots,
        })

    _write(out_dir, "story_index.json", {
        "total_scenes": len(story_scenes),
        "total_shots":  len(all_shots),
        "scenes":       story_scenes,
    })

    for scene in scripts:
        sn        = scene["scene_number"]
        sf        = _scene_folder(sn, scene["title"])
        bg_name   = scene["background"]
        bg_data   = bg_lookup.get(bg_name, {})
        scene_dir = os.path.join(out_dir, "scenes", sf)
        shots_dir = os.path.join(scene_dir, "shots")
        os.makedirs(shots_dir, exist_ok=True)

        scene_shots = [s for s in all_shots if s["scene_number"] == sn]
        scene_pos   = _scene_positions(scene)

        scene_json = {
            "scene_number":   sn,
            "title":          scene["title"],
            "lesson_element": scene["lesson_element"],
            "background": {
                "name":       bg_name,
                "mood":       bg_data.get("mood", ""),
                "image_file": f"../../assets/backgrounds/{bg_name}.png",
            },
            "characters_present": [
                {
                    "name":       name,
                    "image_file": f"../../assets/characters/{name}.png",
                    "position":   scene_pos.get(name, {}),
                }
                for name in _char_names(scene)
            ],
            "total_shots": len(scene_shots),
            "shots": scene_shots,
        }
        with open(os.path.join(scene_dir, "scene.json"), "w", encoding="utf-8") as f:
            json.dump(scene_json, f, indent=2, ensure_ascii=False)

        for i, line in enumerate(scene["script"], 1):
            speaker  = line.get("speaker", "")
            text     = line.get("text", "")
            loc      = scene_pos.get(speaker, {})
            shot_dir = os.path.join(shots_dir, f"shot_{i:02d}_{speaker}")
            os.makedirs(shot_dir, exist_ok=True)
            is_first = (i == 1)

            with open(os.path.join(shot_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(f"SHOT ID      : s{sn:02d}_shot{i:02d}\n")
                f.write(f"SCENE        : {scene['title']}\n")
                f.write(f"SPEAKER      : {speaker} (x={loc.get('x','?')}, y={loc.get('y','?')})\n")
                f.write(f"FRAME SOURCE : {'composite background + characters' if is_first else 'last frame of previous clip'}\n")
                f.write(f"\nVIDEO PROMPT:\n{line.get('video_prompt', '')}\n")

            with open(os.path.join(shot_dir, "voice.txt"), "w", encoding="utf-8") as f:
                f.write(f"SPEAKER: {speaker}\n")
                f.write(f"VOICE  : {line.get('voice_description', '')}\n")
                f.write(f"TEXT   : {text}\n")
                f.write(f"OUT    : voice.mp3\n")

            for ph in ["voice.mp3", "frame.png", "clip.mp4"]:
                open(os.path.join(shot_dir, ph), "w").close()

        print(f"Scene {sn:02d} [{bg_name}] -> scenes/{sf}/ ({len(scene['script'])} shots)")

    print(f"\nTotal scenes : {len(scripts)}")
    print(f"Total shots  : {len(all_shots)}")
    print(f"Output dir   : {out_dir}/")