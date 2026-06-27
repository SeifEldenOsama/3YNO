from __future__ import annotations
import json
import re
import random
import os

GEMINI_VOICES = {
    "Puck":      "male",
    "Charon":    "male",
    "Orus":      "male",
    "Achird":    "male",
    "Enceladus": "male",
    "Zephyr":    "female",
    "Leda":      "female",
    "Kore":      "female",
    "Aoede":     "female",
    "Gacrux":    "female",
    "Sulafat":   "female",
}

# ---------------------------------------------------------------------------
# 3YNO — Fixed host character (crow). Never AI-generated; user supplies PNG.
# ---------------------------------------------------------------------------
ZYNO_VOICE_DESCRIPTION = (
    "Orus A warm and wise male voice with a friendly, enthusiastic tone, "
    "speaking with clear storytelling energy and calm confidence. "
    "The voice is inviting and encouraging, making young children feel "
    "excited and safe to learn."
)

ZYNO_CHARACTER = {
    "name":                 "3YNO",
    "gender":               "male",
    "visual_noun":          "the wise black crow with gleaming golden eyes",
    "voice_description":    ZYNO_VOICE_DESCRIPTION,
    "visual_description":   (
        "FIXED IMAGE — DO NOT GENERATE. "
        "Place the fixed 3YNO crow PNG at characters/3YNO.png before running the video generator."
    ),
    "what_they_represent":  "the host and guide of the educational show",
    "role":                 "educational show host",
    "personality":          "wise, friendly, enthusiastic, encouraging",
    "is_host":              True,
    "skip_generation":      True,
}

ZYNO_VIDEO_PROMPT = (
    "A pure white background with a friendly cartoon black crow character standing at center frame. "
    "The crow is the only element in the scene — no environment, no props, no scenery. "
    "The crow speaks directly to camera, its beak opening and closing in perfect synchronization "
    "with the voice audio. The crow's golden eyes blink expressively and its body animates "
    "naturally and warmly as it talks. "
    "No text, no words, no letters, no captions, no labels of any kind appear anywhere in the scene."
)

ZYNO_NEGATIVE_PROMPT = (
    "text, words, letters, captions, subtitles, typography, font, label, title, watermark, "
    "writing, inscription, alphabets, numbers, digits, overlay text, on-screen text, "
    "speech bubble, dialogue box, low quality, blurry, pixelated, distorted, deformed, "
    "ugly, bad anatomy, duplicate, error, cropped, out of frame, worst quality, "
    "jpeg artifacts, overexposed, underexposed, background scenery, landscape, nature, "
    "sky, ground, grass, trees, buildings, clouds"
)


class StoryGenerator:

    def load_model(
        self,
        model_id: str = "Qwen/Qwen2.5-32B-Instruct",
        cache_dir: str = "/model-cache",
        hf_token: str = "",
        **kwargs,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=hf_token or None,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=hf_token or None,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print(f"Model loaded locally — model: {self.model_id}", flush=True)


    def _ask(self, prompt: str, max_new_tokens: int = 4000, temperature: float = 0.7) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _extract_json(self, raw: str):
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = raw.find(start_char)
            if start == -1:
                continue
            end = raw.rfind(end_char)
            if end == -1:
                continue
            candidate = raw[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        for pattern in [r'\{.*\}', r'\[.*\]']:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        return None

    def _ask_json(self, prompt: str, max_new_tokens: int = 4000, temperature: float = 0.6):
        full_prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY valid JSON. "
            + "No extra text, no markdown, no code fences. Start directly with { or [."
        )
        last_raw = ""
        for attempt in range(5):
            raw = self._ask(full_prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            last_raw = raw
            print(f"JSON attempt {attempt + 1}, raw (first 200 chars): {raw[:200]}", flush=True)
            result = self._extract_json(raw)
            if result is not None:
                return result
            print(f"JSON parse attempt {attempt + 1} failed. Retrying...", flush=True)
        raise RuntimeError(
            f"Model did not return valid JSON after 5 attempts.\nLast output: {last_raw[:500]}"
        )


    def _normalize_gender(self, value: str) -> str:
        v = str(value).strip().lower()
        if v in ("male", "m", "boy", "man", "he", "him"):
            return "male"
        return "female"

    def _filter_voice_lines(self, lines: list, allowed_speakers: list) -> list:
        allowed_set = {s.lower() for s in allowed_speakers}
        filtered = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            speaker = line.get("speaker", "")
            if speaker.lower() not in allowed_set:
                print(f"Skipping unknown speaker '{speaker}'", flush=True)
                continue
            filtered.append(line)
        return filtered


    def analyze_lesson(self, lesson: str) -> dict:
        prompt = f"""You are an educational content designer for a children's animated show.
Your job is to break down a lesson into a story structure that teaches it clearly and accurately.

LESSON:
{lesson}

First, identify:
- How many distinct concepts or steps are in this lesson?
- Does the lesson involve interaction between things (e.g. sun + water + plant)?
- How many different locations or settings would naturally appear in this lesson?

Based on this, decide:
- num_characters: one character PER key concept or actor in the lesson (min 2, max 6)
- num_backgrounds: one background PER distinct setting in the lesson (min 2, max 6)
- num_scenes: one scene PER learning step (min 2, max 4)

CRITICAL: Every scene must teach a real, specific part of the lesson.

Return ONLY this JSON:
{{"num_characters": <int>, "num_backgrounds": <int>, "num_scenes": <int>,
  "reasoning": "<1 sentence why>",
  "lesson_steps": ["<step 1>", "<step 2>", "..."]}}"""
        return self._ask_json(prompt, max_new_tokens=600, temperature=0.3)

    def generate_characters(self, lesson: str, num_characters: int, lesson_steps: list) -> list:
        steps_text = "\n".join([f"- {s}" for s in lesson_steps])
        prompt = f"""You are creating characters for a children's educational animated story.

LESSON:
{lesson}

KEY LESSON STEPS:
{steps_text}

Generate EXACTLY {num_characters} characters.
Each character REPRESENTS a key concept or element from the lesson.

CRITICAL RULES:
- Characters MUST be objects or nature things from the lesson world (e.g. a sun, a raindrop, a cloud, a plant, a wind gust).
- Characters must NEVER be human or have a human face, human body, or human skin. NO people. NO humans. NO boys. NO girls.
- Each character's role must directly relate to what they represent in the lesson.
- Each character MUST have a different image_location so they do not overlap.

For each CHARACTER provide:
- name: ONE single word, fun and kid-friendly. ONE WORD ONLY.
- what_they_represent: the exact lesson concept this character embodies (1 sentence).
- gender: "male" or "female" only, lowercase.
- role: their specific educational role in the story (1 sentence).
- personality: what they are like (1 sentence).
- visual_description: A high-quality TTI image generation prompt for this character ONLY.
  Rules:
  * The character MUST be a cartoon version of a NON-HUMAN object or nature element (e.g. a cartoon sun with a smiley face, a cartoon water droplet, a cartoon cloud). NEVER a human, person, boy, or girl.
  * Pure white background — character only, no scene, no environment, no shadow, no ground line.
  * Art style: premium 2D cartoon illustration rendered at 4K resolution, crisp bold black outlines with variable line weight (thicker outer contour, thinner inner detail lines), vibrant flat colors with subtle cel-shading highlights and soft inner glow to give depth without losing the flat look. Every detail razor-sharp with maximum color fidelity. Inspired by top-tier children's TV animation (Pixar short style meets classic Saturday-morning cartoon).
  * Character body: describe the exact shape, dominant color, secondary accent colors, any texture or pattern details (spots, stripes, sheen, sparkle), and overall silhouette clearly. Be specific about size proportions (e.g. "oversized round head, tiny stubby limbs").
  * Facial features: large round expressive eyes with glossy white specular highlights and colored irises, thick upper eyelashes, a wide cheerful open smile showing small rounded teeth, rosy circular cheek blush marks. NO human nose. NO human ears. NO human skin tone.
  * Add ONE signature detail that makes this character instantly recognizable and memorable (e.g. a tiny crown, glowing aura, sparkle trail, bouncy antenna, leaf hat).
  * Lighting: soft front-facing studio light with a subtle rim highlight on the upper-left edge to make the character pop off the white background.
  * End with: "pure white background, isolated character, no background elements, ultra-high-detail 2D cartoon illustration, 8K resolution, maximum sharpness, sharp crisp lines, vibrant color accuracy, children's premium animated series style, no humans, no people, no realistic textures, no 3D rendering, highest possible quality and resolution, every detail fully rendered and visible".
  * 5-6 sentences total.
    - image_location: Where this character will be placed on the scene image.
      Return as a JSON object: {{"x": <float>, "y": <float>}}
      where (0.0, 0.0) = top-left, (1.0, 1.0) = bottom-right.
      CRITICAL POSITION RULES BASED ON CHARACTER TYPE:
      * SKY / AERIAL characters (sun, cloud, moon, star, bird, raindrop falling, wind, lightning, balloon, kite, rainbow, snowflake, etc.):
        Use y values between 0.15 and 0.40 (upper portion of the frame = sky area).
      * GROUND / SURFACE characters (animals, plants, rocks, mushrooms, insects on ground, water in a river, etc.):
        Use y values between 0.55 and 0.80 (lower portion of the frame = ground area).
      * x values must always be between 0.2 and 0.8 for both types.
      Each character MUST have a DIFFERENT position.
      CRITICAL: No character may be placed at the center. Avoid x between 0.4–0.6 combined with y between 0.4–0.6.
      Clearly separated horizontal zones: left side (x: 0.2–0.35) or right side (x: 0.65–0.8).
- visual_noun: A SHORT 2-5 word phrase starting with "the" that describes HOW THIS CHARACTER PHYSICALLY LOOKS IN THE IMAGE.
  This is pasted directly into video prompts so the AI video model must be able to identify the character from this description alone.
  Describe its COLOR and SHAPE — what would someone see if they looked at it: "the brown spiky ball", "the small green tree", "the red mushroom", "the blue water droplet", "the yellow sun with rays", "the white cloud".
  CRITICAL rules:
  * Include the dominant COLOR and SHAPE — never just a type name like "the tree" or "the fungus"
  * Never use the character name
  * Never use abstract concepts
  * Must match the visual_description of the character exactly
- voice_description: A full natural voice description for Gemini TTS.
  AVAILABLE VOICES (choose based on character gender and personality):
    Male voices: Puck, Charon, Orus, Achird, Enceladus
    Female voices: Zephyr, Leda, Kore, Aoede, Gacrux, Sulafat
  CRITICAL RULES:
  * The VERY FIRST WORD must be the voice name and nothing else — no punctuation before it, no other words before it.
  * Pick ONE voice name that best matches this character's gender and personality — this voice will be used for ALL lines this character speaks.
  * After the voice name, write a space then a full natural description of tone, mood, energy, and speaking style (2-3 sentences).
  * Do NOT use the old "gender, emotion" format.
  Example: "Aoede A warm and enthusiastic female voice with a gentle encouraging tone, speaking with calm clarity and a nurturing energy perfect for teaching young children."

Return a JSON array of EXACTLY {num_characters} objects."""
        chars = self._ask_json(prompt, max_new_tokens=1500 * num_characters)
        if isinstance(chars, list):
            for c in chars:
                if isinstance(c, dict):
                    c["gender"] = self._normalize_gender(c.get("gender", "female"))
        return chars

    def generate_backgrounds(self, lesson: str, num_backgrounds: int, lesson_steps: list) -> list:
        steps_text = "\n".join([f"- {s}" for s in lesson_steps])
        prompt = f"""You are creating background scenes for a children's educational animated story.

LESSON:
{lesson}

KEY LESSON STEPS:
{steps_text}

Generate EXACTLY {num_backgrounds} background scenes.
Each background must reflect a REAL setting where part of this lesson takes place.
Do NOT include any characters in the background.

For each BACKGROUND provide:
- name: short 2-3 word label
- lesson_context: which part of the lesson happens here (1 sentence)
- visual_description: A high-quality TTI image generation prompt for this background ONLY. NO characters. NO humans. NO people anywhere in the scene.
  Style rules:
  * Premium 2D cartoon illustration style rendered at 4K resolution — crisp bold outlines with variable weight, vibrant flat color fills with subtle cel-shading to add depth, rich layered composition (clear foreground details, midground subject, distant background elements) creating a strong sense of depth without 3D rendering. Maximum sharpness, ultra-fine environmental detail, and cinematic color grading throughout.
  * Describe the scene in layers: (1) SKY / CEILING layer — exact colors, cloud shapes, sun/moon position, light rays, stars, or weather effects; (2) MIDGROUND — main environmental features (hills, trees, water, buildings, rocks) with specific colors, shapes, and textures; (3) FOREGROUND — close-up ground details (grass tufts, pebbles, flowers, sand, roots, puddles) that frame the bottom of the image.
  * Lighting & atmosphere: specify the time of day, direction of light, color temperature (warm golden, cool blue, soft pink dawn, etc.), any atmospheric effects (mist, sparkle particles, light shafts, glowing embers, bubbles, falling leaves).
  * Color palette: name 3-4 dominant colors and describe how they interact (e.g. "deep teal sky gradients into soft mint at the horizon, contrasted by warm amber ground tones").
  * Mood details: include small environmental storytelling elements that reinforce the scene's mood (fireflies dotting the air, rippling water reflections, swaying grass, dappled sunlight through leaves).
  * Must end with: "2D cartoon background, premium children's animated series style, 8K resolution, ultra-sharp detail, vibrant color grading, cinematic composition, no characters, no people, no humans, no text, highly detailed environment illustration, highest possible quality and resolution, every background element fully visible and richly detailed, no detail omitted or blurred".
  * 6-8 sentences total.
- mood: one word (cheerful, cozy, adventurous, mysterious, warm, bright, etc.)

Return a JSON array of EXACTLY {num_backgrounds} background objects:
[{{"name":"","lesson_context":"","visual_description":"","mood":""}}]"""
        return self._ask_json(prompt, max_new_tokens=1200 * num_backgrounds)

    def _normalize_outline_names(self, outline: list, characters: list) -> list:
        canonical = {c['name'].lower(): c['name'] for c in characters}
        for scene in outline:
            # Normalize names
            for entry in scene.get('characters_present', []):
                if isinstance(entry, dict):
                    key = entry['name'].lower()
                    if key in canonical:
                        entry['name'] = canonical[key]
                    else:
                        for ckey, cname in canonical.items():
                            if key in ckey or ckey in key:
                                print(f"Normalizing character name '{entry['name']}' -> '{cname}'", flush=True)
                                entry['name'] = cname
                                break
        return outline

    def generate_outline(self, lesson: str, characters: list, backgrounds: list,
                         num_scenes: int, lesson_steps: list) -> list:
        chars_summary = "\n".join([
            f"- {c['name']} ({c['gender']}): represents {c.get('what_they_represent', c['role'])}"
            for c in characters
        ])
        bgs_summary = "\n".join([
            f"- {b['name']}: {b.get('lesson_context', b['mood'])}"
            for b in backgrounds
        ])
        steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(lesson_steps)])
        bg_names   = ", ".join([b['name'] for b in backgrounds])
        char_names = ", ".join([c['name'] for c in characters])

        prompt = f"""You are writing a children's educational story outline.
The story exists ONLY to teach this lesson - every scene must serve the lesson.

LESSON:
{lesson}

LESSON STEPS TO COVER (every step must appear in at least one scene):
{steps_text}

CHARACTERS:
{chars_summary}

AVAILABLE BACKGROUNDS:
{bgs_summary}

Create EXACTLY {num_scenes} scenes. Rules:
- Map each lesson step to one or more scenes - cover ALL steps, in order
- Every scene must have a clear "lesson_element" that is a specific accurate fact from the lesson
- Only use character names from: {char_names}
- Only use background names from: {bg_names}
- CHARACTER LIMIT PER SCENE: EXACTLY 1 or 2 characters per scene. NEVER 3 or more. This is a hard rule — any scene with more than 2 characters_present entries is invalid.
- CHARACTER REUSE ACROSS SCENES: The same character MUST appear in multiple scenes. You have {len(characters)} characters and {num_scenes} scenes — distribute them so every character appears at least once, and key characters recur across scenes to build continuity. Do NOT assign a unique set of characters to each scene.
- POSITION RULE: x values must always be between 0.2 and 0.8. y values depend on what the character physically IS:
  * SKY / AERIAL characters (sun, cloud, moon, star, bird, raindrop, wind, lightning, balloon, rainbow, snowflake, etc.): y between 0.15 and 0.40.
  * GROUND / SURFACE characters (animals, plants, rocks, mushrooms, insects on ground, water bodies, etc.): y between 0.55 and 0.80.
  NO character may be at the center (avoid x 0.4–0.6 combined with y 0.4–0.6). Separate characters horizontally: left side (x: 0.2–0.35) or right side (x: 0.65–0.8).

Return a JSON array of exactly {num_scenes} scene objects:
[
  {{
    "scene_number": 1,
    "title": "",
    "background": "",
    "characters_present": [
      {{"name": "<character name>", "position": {{"x": 0.0, "y": 0.0}}}}
    ],
    "event": "",
    "lesson_element": "<specific accurate fact from the lesson>",
    "lesson_step_covered": "<which lesson step this scene covers>"
  }}
]"""
        return self._ask_json(prompt, max_new_tokens=4000)

    def generate_passages(self, lesson: str, outline: list,
                          characters: list, backgrounds: list,
                          lesson_steps: list) -> list:

        def get_char_names(characters_present):
            names = []
            for entry in characters_present:
                if isinstance(entry, dict):
                    names.append(entry['name'])
                elif isinstance(entry, str):
                    names.append(entry)
            return names

        def get_bg(name):
            for b in backgrounds:
                if b['name'].lower().strip() == name.lower().strip():
                    return b['visual_description']
            return name

        def get_profiles(characters_present):
            names = get_char_names(characters_present)
            lines = []
            for n in names:
                for c in characters:
                    if c['name'] == n:
                        lines.append(
                            f"{c['name']} ({c['gender']}): "
                            f"represents {c.get('what_they_represent', c['role'])}. "
                            f"Personality: {c['personality']}."
                        )
            return "\n".join(lines)

        steps_text   = "\n".join([f"- {s}" for s in lesson_steps])
        passages     = []
        story_so_far = ""

        for scene in outline:
            bg_detail       = get_bg(scene['background'])
            char_profiles   = get_profiles(scene['characters_present'])
            char_names_list = get_char_names(scene['characters_present'])
            num_chars       = len(char_names_list)
            interact_rule   = (
                "- Characters MUST talk to each other and their dialogue must explain "
                "how their concepts relate to each other in the lesson"
                if num_chars > 1 else ""
            )
            prompt = f"""Write one passage of a children's educational story.

FULL LESSON BEING TAUGHT:
{lesson}

ALL LESSON STEPS:
{steps_text}

THIS SCENE TEACHES: {scene['lesson_element']}
LESSON STEP COVERED: {scene.get('lesson_step_covered', '')}

Story so far:
{story_so_far if story_so_far else "(This is the first scene.)"}

Scene event: {scene['event']}
Background: {bg_detail}

Characters in this scene: {", ".join(char_names_list)}
{char_profiles}

CRITICAL WRITING RULES:
- The passage MUST clearly teach: "{scene['lesson_element']}"
- Characters speak and act AS their lesson concept
- Use simple words and short sentences (age 5-8)
- Cheerful and warm tone
- Length: 2-3 short paragraphs
- ONLY use characters listed above
- Return ONLY the story text, no titles or labels
{interact_rule}"""
            text = self._ask(prompt, max_new_tokens=2000, temperature=0.6)
            passages.append({
                "scene_number":           scene['scene_number'],
                "title":                  scene['title'],
                "background":             scene['background'],
                "background_description": bg_detail,
                "characters_present":     scene['characters_present'],
                "lesson_element":         scene['lesson_element'],
                "lesson_step_covered":    scene.get('lesson_step_covered', ''),
                "passage":                text,
            })
            story_so_far += f"\n\n[Scene {scene['scene_number']} - taught: {scene['lesson_element']}]\n{text}"
            print(f"Scene {scene['scene_number']} written.", flush=True)
        return passages

    def generate_voice_scripts(self, passages: list, characters: list) -> list:
        all_scripts = []
        char_lookup = {c['name']: c for c in characters}

        for sp in passages:
            scene_char_entries = sp['characters_present']
            scene_pos          = {}
            char_names_list    = []

            for entry in scene_char_entries:
                if isinstance(entry, dict):
                    char_names_list.append(entry['name'])
                    scene_pos[entry['name']] = entry.get('position', {})
                elif isinstance(entry, str):
                    char_names_list.append(entry)
                    scene_pos[entry] = {}

            char_voice_info = "\n".join([
                f"- {name} (gender: {char_lookup[name]['gender']}): {char_lookup[name]['voice_description']}"
                for name in char_names_list if name in char_lookup
            ])

            # Build a lookup of what each character represents (their real-world identity)
            def _identity_line(name):
                vn  = char_lookup[name].get('visual_noun', 'the character')
                wtr = char_lookup[name].get('what_they_represent', char_lookup[name].get('role', name))
                vd  = char_lookup[name].get('voice_description', '')
                return '- ' + name + ": visual_noun = '" + vn + "', what_they_represent = '" + wtr + "', voice_description = '" + vd + "'"
            char_identity_map = "\n".join([
                _identity_line(name)
                for name in char_names_list if name in char_lookup
            ])

            location_map = "\n".join([
                f"- {name}: x={scene_pos[name].get('x','?')}, y={scene_pos[name].get('y','?')}"
                for name in char_names_list
            ])

            # Build identity descriptions for use in video_prompt context
            identity_lines = [
                f"{name} (who represents {char_lookup[name].get('what_they_represent', char_lookup[name].get('role', name))})"
                for name in char_names_list if name in char_lookup
            ]

            others_context = (
                f"Characters in this scene: {', '.join(identity_lines)}. "
                "For each shot, one character speaks while the others are still present and visible. "
                "The video_prompt MUST clearly identify the speaking character by BOTH their name AND what they represent "
                "(e.g. 'Sunshine the sun', 'Droplet the water droplet', 'Breezy the wind'). "
                "It must also state their role (e.g. 'who provides energy to plants'). "
                "CRITICAL: To prevent the AI from animating the wrong character, you MUST explicitly state that all other characters are 'completely static and unmoving'. "
                "Describe what EVERY other character in the scene is doing (e.g. 'watching attentively', 'listening', 'looking curious') but always include the phrase 'completely static and frozen' for them. "
                "Identify each of them by name and what they represent."
                if len(char_names_list) > 1
                else
                f"The only character in this scene is {char_names_list[0]}, who represents "
                f"{char_lookup[char_names_list[0]].get('what_they_represent', char_lookup[char_names_list[0]].get('role', char_names_list[0])) if char_names_list[0] in char_lookup else char_names_list[0]}. "
                "The video_prompt should identify this character by both their name and what they represent, "
                "mention their role, and focus entirely on this character speaking."
            )

            # Build position hints from x,y coords
            # Keywords that indicate a character naturally lives in the sky/air
            AERIAL_KEYWORDS = {
                "sun", "cloud", "moon", "star", "rainbow", "sky", "bird",
                "butterfly", "bee", "dragonfly", "balloon", "kite", "lightning",
                "thunder", "wind", "breeze", "gust", "comet", "meteor", "planet",
                "rocket", "airplane", "aircraft", "ufo", "fairy", "angel", "fly",
                "flying", "hawk", "eagle", "owl", "bat", "snowflake", "hail",
                "rain", "raindrop", "droplet", "haze", "fog", "mist", "aurora",
            }

            def _is_aerial(name):
                """Return True if the character naturally belongs in the sky."""
                char = char_lookup.get(name, {})
                # Check visual_noun, what_they_represent, and name itself
                text_to_check = " ".join([
                    str(char.get("visual_noun", "")),
                    str(char.get("what_they_represent", "")),
                    str(char.get("visual_description", "")),
                    name,
                ]).lower()
                return any(kw in text_to_check for kw in AERIAL_KEYWORDS)

            def pos_hint(name):
                pos = scene_pos.get(name, {})
                x = float(pos.get("x", 0.5))
                y = float(pos.get("y", 0.5))
                multi = len(char_names_list) > 1
                aerial = _is_aerial(name)
                h_word = "left" if x < 0.4 else ("right" if x > 0.6 else ("left" if multi else "center"))
                # Vertical label: only use sky/ground language when it makes physical sense
                if aerial:
                    # Aerial characters: y < 0.5 = high in the sky, y >= 0.5 = lower sky
                    v_word = "high in the sky on the" if y < 0.5 else "in the lower sky on the"
                    return f"{v_word} {h_word}"
                else:
                    # Ground characters: ignore y, just use horizontal placement
                    return f"on the {h_word} side"

            pos_hints = "\n".join([
                f"- {name}: {pos_hint(name)} of the scene"
                for name in char_names_list
            ])

            prompt = f"""You are writing a voice script for a children's educational animated video.

THIS SCENE TEACHES: {sp['lesson_element']}
BACKGROUND: {sp['background_description']}

CHARACTER POSITIONS IN FRAME:
{pos_hints}

CHARACTER VOICES:
{char_voice_info}

{others_context}

CRITICAL RULES:
- Every line of dialogue MUST relate to the lesson fact being taught
- NO narration - only character dialogue
- Each line must move the lesson understanding forward
- Keep dialogue natural and fun

LINE LENGTH RULES:
- If there is ONLY ONE character in this scene: return EXACTLY ONE item in the JSON array. The single "text" field must be 80 to 120 words — a full, rich monologue covering the entire lesson element. Do NOT split it into multiple shots.
- If there are MULTIPLE characters in this scene: each character takes turns speaking. Every "text" field MUST be between 38 and 50 words. NEVER write fewer than 38 words or more than 50 words per line. This ensures each line takes between 15 and 20 seconds to speak aloud.

Story passage to adapt into dialogue:
{sp["passage"]}

Return a JSON array. Each item must have EXACTLY these 4 fields:
1. speaker: MUST be exactly one of: {", ".join(char_names_list)}
2. text: EXACTLY 38 to 50 words of natural dialogue (this must take 15-20 seconds to speak).
3. voice_description: Copy EXACTLY the voice_description from this character's profile in the CHARACTER IDENTITY MAP below. Do NOT change it or generate a new one.
4. video_prompt: MUST follow this EXACT template, filling in the bracketed placeholders based on the scene:

   "A beautiful landscape with [SPEAKING character visual_noun] and [each STATIC character visual_noun, joined with 'and']. [SPEAKING character visual_noun, capitalized] in the [speaker position] is the only character moving. [SPEAKING character visual_noun, capitalized] speaks the words in the voice, its mouth opening and closing in perfect synchronization. [SPEAKING character visual_noun, capitalized]'s eyes blink and its body animates as it talks. [For each STATIC character: '[Static character visual_noun, capitalized] in the [static character position] remains completely static, its eyes wide and its smile frozen.'] No text, no words, no letters, no captions, no labels of any kind appear anywhere in the scene."

   STRICT RULES:
   - SPEAKING CHARACTER ALWAYS FIRST: In "A beautiful landscape with ...", the SPEAKING character visual_noun MUST be written first, before all static characters. If the green jellyfish speaks, write "...with the green jellyfish and the yellow firefly..." NOT "...with the yellow firefly and the green jellyfish...".
   - Use visual_noun for EVERY character reference — NEVER use character names.
   - Use POSITION MAP for frame positions EXACTLY as written (e.g. "on the left side", "high in the sky on the right", "on the right side"). Copy the position label word-for-word from the CHARACTER POSITION MAP below.
   - NEVER say a ground character is "floating", "in the upper", "in the lower", or imply it is airborne if its position says "on the left/right side".
   - NEVER say a sky character is "on the ground" or "standing" if its position says "in the sky".
   - After the opening sentence, describe the SPEAKING character moving first, then write one frozen description sentence per static character last.
   - If there is only one character in the scene, skip the static character sentences entirely.
   - Keep the sentence structure close to the template — only swap character nouns and positions.

5. negative_prompt: Use ONLY this exact fixed string for every line:

   "text, words, letters, captions, subtitles, typography, font, label, title, watermark, writing, inscription, characters, alphabets, numbers, digits, overlay text, on-screen text, speech bubble, dialogue box, low quality, blurry, pixelated, distorted, deformed, ugly, bad anatomy, duplicate, error, cropped, out of frame, worst quality, jpeg artifacts, overexposed, underexposed"

   STRICT RULES:
   - The negative_prompt is ALWAYS this fixed string — never add character names, visual nouns, or anything else.
   - Do NOT change it based on which character is speaking or which are static.
   - If there is only one character in the scene, still use the same fixed string.
CHARACTER IDENTITY MAP (name → what they represent):
{char_identity_map}

CHARACTER POSITION MAP:
{pos_hints}

Return ONLY the JSON array."""

            lines = self._ask_json(prompt, max_new_tokens=5000, temperature=0.4)
            lines = self._filter_voice_lines(lines, char_names_list)

            all_scripts.append({
                "scene_number":       sp['scene_number'],
                "title":              sp['title'],
                "background":         sp['background'],
                "characters_present": scene_char_entries,
                "lesson_element":     sp['lesson_element'],
                "script":             lines,
            })
            print(f"Scene {sp['scene_number']} scripted ({len(lines)} lines).", flush=True)
        return all_scripts

    # ------------------------------------------------------------------
    # 3YNO host scenes
    # ------------------------------------------------------------------

    def generate_3yno_scenes(self, lesson: str, scripts: list) -> list:
        """
        Generate 3YNO host scenes to wrap and bridge the regular story scenes:
          - 1 intro  : welcome + full lesson brief + preview of Scene 1
          - N-1 transitions : recap previous scene + preview next scene
          - 1 outro  : congratulate kids + full lesson recap + goodbye

        Returns a list of host-scene dicts ordered by insertion position.
        Each dict has:
          type          : "intro" | "transition" | "outro"
          insert_before : scene_number of the regular scene that follows (None for outro)
          after_scene   : scene_number of the scene just finished (transitions only)
          text          : the spoken dialogue for 3YNO
        """
        scene_meta = [
            (s["scene_number"], s["title"], s["lesson_element"])
            for s in scripts
        ]
        num_scenes = len(scene_meta)
        host_scenes = []

        # INTRO
        first_num, first_title, _ = scene_meta[0]
        scene_list_text = "\n".join(
            f"  * Scene {n}: {t} (teaches: {e})" for n, t, e in scene_meta
        )
        intro_prompt = (
            "You are 3YNO, a friendly and wise crow who hosts a children's educational show.\n"
            "You are about to introduce today's episode to young children (ages 5-8).\n\n"
            f"TODAY'S LESSON:\n{lesson}\n\n"
            f"SCENES THAT WILL FOLLOW:\n{scene_list_text}\n\n"
            "Write 3YNO's INTRO speech. It MUST:\n"
            "1. Warmly welcome the children to the show (1-2 sentences).\n"
            "2. Give an exciting, fast brief of the FULL lesson and its main concepts (2-3 sentences).\n"
            f"3. Tease what will happen in the FIRST scene \"{first_title}\" (1-2 sentences).\n\n"
            "RULES:\n"
            "- Speak DIRECTLY to kids (use 'you', 'we', 'let us').\n"
            "- Simple vocabulary for ages 5-8.\n"
            "- Cheerful, warm, enthusiastic tone.\n"
            "- 60 to 90 words total.\n\n"
            'Return ONLY a JSON object: {"text": "<speech here>"}'
        )
        raw = self._ask_json(intro_prompt, max_new_tokens=500, temperature=0.5)
        intro_text = raw.get("text", "") if isinstance(raw, dict) else str(raw)
        host_scenes.append({
            "type":          "intro",
            "insert_before": first_num,
            "after_scene":   None,
            "text":          intro_text,
        })
        print("3YNO intro scene generated.", flush=True)

        # TRANSITIONS (between every pair of regular scenes)
        for i in range(num_scenes - 1):
            cur_num,  cur_title,  cur_elem  = scene_meta[i]
            next_num, next_title, next_elem = scene_meta[i + 1]
            trans_prompt = (
                "You are 3YNO, a friendly and wise crow hosting a children's educational show.\n"
                "You appear between two scenes to bridge them.\n\n"
                f"PREVIOUS SCENE: \"{cur_title}\" -- taught: {cur_elem}\n"
                f"NEXT SCENE:     \"{next_title}\" -- will teach: {next_elem}\n\n"
                "Write 3YNO's TRANSITION speech. It MUST:\n"
                "1. Give a brief, exciting recap of the PREVIOUS scene (2-3 sentences).\n"
                "2. Build anticipation for the NEXT scene (1-2 sentences).\n\n"
                "RULES:\n"
                "- Speak DIRECTLY to kids (use 'you', 'we', 'let us').\n"
                "- Simple vocabulary for ages 5-8.\n"
                "- Cheerful, warm, enthusiastic tone.\n"
                "- 45 to 65 words total.\n\n"
                'Return ONLY a JSON object: {"text": "<speech here>"}'
            )
            raw = self._ask_json(trans_prompt, max_new_tokens=400, temperature=0.5)
            trans_text = raw.get("text", "") if isinstance(raw, dict) else str(raw)
            host_scenes.append({
                "type":          "transition",
                "insert_before": next_num,
                "after_scene":   cur_num,
                "text":          trans_text,
            })
            print(
                f"3YNO transition scene generated (scene {cur_num} -> {next_num}).",
                flush=True,
            )

        # OUTRO
        last_num, last_title, _ = scene_meta[-1]
        outro_prompt = (
            "You are 3YNO, a friendly and wise crow hosting a children's educational show.\n"
            "The episode is over and you appear at the very end to say goodbye.\n\n"
            f"FULL LESSON TAUGHT TODAY:\n{lesson}\n\n"
            f"ALL SCENES COVERED:\n{scene_list_text}\n\n"
            "Write 3YNO's OUTRO speech. It MUST:\n"
            "1. Congratulate the kids for watching and learning (1-2 sentences).\n"
            "2. Briefly summarize the most important things they learned today (2-3 sentences).\n"
            "3. Say a warm, friendly goodbye and invite them to come back (1-2 sentences).\n\n"
            "RULES:\n"
            "- Speak DIRECTLY to kids (use 'you', 'we').\n"
            "- Simple vocabulary for ages 5-8.\n"
            "- Warm, proud, encouraging tone.\n"
            "- 60 to 90 words total.\n\n"
            'Return ONLY a JSON object: {"text": "<speech here>"}'
        )
        raw = self._ask_json(outro_prompt, max_new_tokens=500, temperature=0.5)
        outro_text = raw.get("text", "") if isinstance(raw, dict) else str(raw)
        host_scenes.append({
            "type":          "outro",
            "insert_before": None,
            "after_scene":   last_num,
            "text":          outro_text,
        })
        print("3YNO outro scene generated.", flush=True)

        print(
            f"3YNO host scenes complete: {len(host_scenes)} "
            f"(1 intro + {num_scenes - 1} transition(s) + 1 outro).",
            flush=True,
        )
        return host_scenes